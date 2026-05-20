"""Verify welfare_shadow integrator on 9x10 using RunGTAP's solved inputs.

Pipes RunGTAP's cumulative `u`, `dpsave`, `dppriv`, `dpgov` per region back
into `welfare_shadow.integrate()` with the matching 9x10 baseline coefs
(from basedata.har + default.prm). The output EV should match RunGTAP's
sl4 EV header to within the same tolerance as NUS333 (~0.3% USA, 0.4% ROW)
— this validates the chain scales to 10 regions × 10 commodities.

This is a STRUCTURAL test of the shadow demand chain, not of equilibria's
MCP solve. The MCP-side validation requires running equilibria 9x10 with
capFix closure (~15 min on this laptop) — done separately.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import xlrd

from equilibria.templates.gtap.welfare_shadow import ShadowBaseline, integrate

HERE = Path(__file__).resolve().parent
BASEDATA_XLS = HERE / "rungtap" / "basedata.xls"
DEFAULT_XLS  = HERE / "rungtap" / "default.xls"
SL4DUMP_XLS  = HERE / "rungtap" / "sl4dump.xls"

REGIONS = (
    "Oceania", "EastAsia", "SEAsia", "SouthAsia", "NAmerica",
    "LatinAmer", "EU_28", "MENA", "SSA", "RestofWorld",
)
COMMS = (
    "c_Crops", "c_MeatLstk", "c_Extraction", "c_ProcFood", "c_TextWapp",
    "c_LightMnfc", "c_HeavyMnfc", "c_Util_Cons", "c_TransComm", "c_OthService",
)


def read_reg(wb, sheet):
    s = wb.sheet_by_name(sheet)
    out = {}
    for r in range(s.nrows):
        try: int(s.cell_value(r, 0))
        except (ValueError, TypeError): continue
        out[str(s.cell_value(r, 1)).strip()] = float(s.cell_value(r, 2))
    return out


def read_comm_reg(wb, sheet):
    s = wb.sheet_by_name(sheet)
    reg_labels = []
    data_start = None
    for r in range(s.nrows):
        if s.ncols > 2 and str(s.cell_value(r, 2)).strip() in REGIONS:
            reg_labels = [
                str(s.cell_value(r, c)).strip()
                for c in range(2, s.ncols)
                if str(s.cell_value(r, c)).strip() in REGIONS
            ]
            data_start = r + 1
            break
    out = {}
    if data_start is None:
        return out
    for r in range(data_start, s.nrows):
        try: int(s.cell_value(r, 0))
        except (ValueError, TypeError): continue
        comm = str(s.cell_value(r, 1)).strip()
        for j, reg in enumerate(reg_labels):
            try:
                out[(comm, reg)] = float(s.cell_value(r, 2 + j))
            except (ValueError, TypeError):
                pass
    return out


def build_baseline(region: str, wb_base, wb_prm) -> ShadowBaseline:
    VDPP = read_comm_reg(wb_base, "VDPP")
    VMPP = read_comm_reg(wb_base, "VMPP")
    VDGP = read_comm_reg(wb_base, "VDGP")
    VMGP = read_comm_reg(wb_base, "VMGP")
    SAVE = read_reg(wb_base, "SAVE")
    INCPAR = read_comm_reg(wb_prm, "INCP")
    SUBPAR = read_comm_reg(wb_prm, "SUBP")
    DPSM = read_reg(wb_base, "DPSM")

    VPP = {c: VDPP.get((c, region), 0.0) + VMPP.get((c, region), 0.0) for c in COMMS}
    VGP_total = sum(VDGP.get((c, region), 0.0) + VMGP.get((c, region), 0.0) for c in COMMS)
    PRIVEXP = sum(VPP.values())
    return ShadowBaseline(
        region=region, commodities=COMMS, PRIVEXP=PRIVEXP, GOVEXP=VGP_total,
        SAVE=SAVE[region], INCOME=PRIVEXP + VGP_total + SAVE[region],
        VPP=VPP,
        INCPAR={c: INCPAR[(c, region)] for c in COMMS},
        ALPHA={c: SUBPAR[(c, region)] for c in COMMS},
        DPARSUM=DPSM[region],
    )


def main():
    wb_base = xlrd.open_workbook(str(BASEDATA_XLS))
    wb_prm  = xlrd.open_workbook(str(DEFAULT_XLS))
    wb_sl4  = xlrd.open_workbook(str(SL4DUMP_XLS))

    u_rg       = read_reg(wb_sl4, "0094")
    EV_rg      = read_reg(wb_sl4, "0208")
    dpsave_rg  = read_reg(wb_sl4, "0087")
    dpgov_rg   = read_reg(wb_sl4, "0088")
    dppriv_rg  = read_reg(wb_sl4, "0089")
    yev_rg     = read_reg(wb_sl4, "0202")

    print("=" * 88)
    print("9x10 — welfare_shadow integrator vs RunGTAP (Euler N=25 default)")
    print("=" * 88)
    print(f"{'region':<12}{'u_pct':>10}{'dpsave':>10}{'EV integ':>14}{'EV RG':>14}{'gap %':>10}")

    total_eq = 0.0
    total_rg = 0.0
    for region in REGIONS:
        base = build_baseline(region, wb_base, wb_prm)
        res = integrate(
            base,
            u_pct=u_rg[region],
            dppriv_pct=dppriv_rg.get(region, 0.0),
            dpgov_pct=dpgov_rg.get(region, 0.0),
            dpsave_pct=dpsave_rg.get(region, 0.0),
        )
        rg = EV_rg[region]
        gap_pct = (res.EV_USDm - rg) / rg * 100 if rg else float('nan')
        print(
            f"{region:<12}{u_rg[region]:>10.4f}{dpsave_rg.get(region,0):>10.4f}"
            f"{res.EV_USDm:>14,.0f}{rg:>14,.0f}{gap_pct:>+9.2f}%"
        )
        total_eq += res.EV_USDm
        total_rg += rg

    world_gap = (total_eq - total_rg) / total_rg * 100 if total_rg else float('nan')
    print("-" * 88)
    print(
        f"{'WORLD':<12}{'':<10}{'':<10}"
        f"{total_eq:>14,.0f}{total_rg:>14,.0f}{world_gap:>+9.2f}%"
    )


if __name__ == "__main__":
    main()
