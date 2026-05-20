"""Verify the shadow integrator using RunGTAP's ACTUAL non-linear trajectory
of `dpsave`, `u`, etc. across shock magnitudes 1% → 10%.

If the integrator now matches RunGTAP's final EV exactly, that proves the
shadow demand chain is correct and the remaining 2.5% gap was entirely due
to the linear-trajectory assumption inside [0,1]. The fix for equilibria-
native parity is then: run the model at multiple homotopy points and feed
the per-segment cumulatives to this integrator.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import xlrd
from equilibria.templates.gtap.welfare_shadow import (
    ShadowBaseline, _make_state, _step, _apply_updates,
)

HERE = Path(__file__).resolve().parent
TRAJ_JSON   = HERE / "trajectory.json"
BASEDATA_XLS = HERE / "rungtap" / "basedata.xls"
DEFAULT_XLS  = HERE / "rungtap" / "default.xls"
SL4DUMP_XLS  = HERE / "rungtap" / "sl4dump.xls"
REGIONS = ("USA", "ROW")
COMMS   = ("AGR", "MFG", "SER")


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
    VPP = {c: VDPP[(c, region)] + VMPP[(c, region)] for c in COMMS}
    VGP_total = sum(VDGP[(c, region)] + VMGP[(c, region)] for c in COMMS)
    PRIVEXP = sum(VPP.values())
    return ShadowBaseline(
        region=region, commodities=COMMS, PRIVEXP=PRIVEXP, GOVEXP=VGP_total,
        SAVE=SAVE[region], INCOME=PRIVEXP + VGP_total + SAVE[region],
        VPP=VPP,
        INCPAR={c: INCPAR[(c, region)] for c in COMMS},
        ALPHA={c: SUBPAR[(c, region)] for c in COMMS},
        DPARSUM=DPSM[region],
    )


def integrate_with_trajectory(base: ShadowBaseline, trajectory: dict[str, dict[str, dict[str, float]]],
                              region: str, sub_n: int = 1) -> dict[str, float]:
    """Integrate the shadow demand chain using the ACTUAL trajectory of
    inputs from RunGTAP solves at multiple shock magnitudes. Between each
    pair of consecutive shock magnitudes (1%→2%, 2%→3%, …), use `sub_n`
    Euler substeps with the segment's per-step input increments.
    """
    st = _make_state(base)
    # The cumulative percent changes at each shock magnitude (1, 2, …, 10).
    shocks = sorted(int(k) for k in trajectory.keys())
    cum = {0: {"u": 0.0, "dpsave": 0.0, "dppriv": 0.0, "dpgov": 0.0}}
    for s in shocks:
        cum[s] = {
            "u":       trajectory[str(s)]["u"][region],
            "dpsave":  trajectory[str(s)]["dpsave"][region],
            "dppriv":  trajectory[str(s)].get("dppriv", {}).get(region, 0.0),
            "dpgov":   trajectory[str(s)].get("dpgov",  {}).get(region, 0.0),
        }

    acc = {k: 0.0 for k in (
        "yev", "ypev", "ygev", "ysaveev", "ueprivev", "uelasev",
        "qsaveev", "upev", "ugev", "EV",
    )}

    prev_s = 0
    for s in shocks:
        # Increments across this segment, applied via `sub_n` Euler substeps.
        seg = {k: cum[s][k] - cum[prev_s][k] for k in cum[s]}
        for _ in range(sub_n):
            du = seg["u"] / sub_n
            ddpsave = seg["dpsave"] / sub_n
            ddppriv = seg["dppriv"] / sub_n
            ddpgov  = seg["dpgov"]  / sub_n
            step = _step(st, base, du, 0.0, ddppriv, ddpgov, ddpsave, 0.0)
            for k in ("yev", "ypev", "ygev", "ysaveev", "ueprivev",
                      "uelasev", "qsaveev", "upev", "ugev"):
                acc[k] += step[k]
            acc["EV"] += (st.INCOMEEV / 100.0) * step["yev"]
            _apply_updates(st, base, step)
        prev_s = s

    return acc


def main():
    trajectory = json.loads(TRAJ_JSON.read_text())
    wb_base = xlrd.open_workbook(str(BASEDATA_XLS))
    wb_prm  = xlrd.open_workbook(str(DEFAULT_XLS))
    wb_sl4  = xlrd.open_workbook(str(SL4DUMP_XLS))

    EV_rg     = read_reg(wb_sl4, "0208")
    yev_rg    = read_reg(wb_sl4, "0202")
    upev_rg   = read_reg(wb_sl4, "0200")
    ugev_rg   = read_reg(wb_sl4, "0199")
    qsave_rg  = read_reg(wb_sl4, "0201")
    uelas_rg  = read_reg(wb_sl4, "0197")

    print("=" * 96)
    print("Etapa 3 — verificación con TRAYECTORIA REAL de RunGTAP (1%→2%→…→10%)")
    print("=" * 96)
    for sub_n in (1, 2, 3, 4, 5, 8, 16, 32):
        print(f"\n--- sub_n = {sub_n} Euler substeps per 1% segment ---")
        print(f"{'reg':<5}{'var':<10}{'integ':>14}{'RunGTAP':>14}{'gap':>14}{'gap_pct':>10}")
        for region in REGIONS:
            base = build_baseline(region, wb_base, wb_prm)
            res = integrate_with_trajectory(base, trajectory, region, sub_n=sub_n)
            for vn, ours, theirs in (
                ("yev",     res["yev"],     yev_rg[region]),
                ("upev",    res["upev"],    upev_rg[region]),
                ("ugev",    res["ugev"],    ugev_rg[region]),
                ("qsaveev", res["qsaveev"], qsave_rg[region]),
                ("uelasev", res["uelasev"], uelas_rg[region]),
                ("EV ($M)", res["EV"],      EV_rg[region]),
            ):
                gap = ours - theirs
                gap_pct = (gap / theirs * 100) if theirs else float("nan")
                print(f"{region:<5}{vn:<10}{ours:>14.4f}{theirs:>14.4f}{gap:>+14.4f}{gap_pct:>+9.2f}%")


if __name__ == "__main__":
    main()
