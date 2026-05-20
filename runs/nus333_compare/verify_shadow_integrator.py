"""Etapa 2 — verificación del integrador shadow demand system.

Carga los coeficientes baseline de NUS333 desde RunGTAP basedata.xls +
default.xls, toma los valores cumulativos `u, dppriv, dpgov, dpsave` que
RunGTAP produjo en el solve con cierre capFix (vía swap), corre el
integrador `welfare_shadow.integrate` y verifica que reproduce el
`yev` (y por lo tanto el `EV`) de RunGTAP a alta precisión.

Si esto funciona, el shadow demand chain de equilibria queda probado
contra RunGTAP. El siguiente paso (Etapa 3) sería extraer los inputs
(`u, dpsave` etc.) directamente de la solve de equilibria.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import xlrd
from equilibria.templates.gtap.welfare_shadow import ShadowBaseline, integrate

BASEDATA_XLS = ROOT / "runs" / "nus333_compare" / "rungtap" / "basedata.xls"
DEFAULT_XLS  = ROOT / "runs" / "nus333_compare" / "rungtap" / "default.xls"
SL4DUMP_XLS  = ROOT / "runs" / "nus333_compare" / "rungtap" / "sl4dump.xls"
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
        region=region,
        commodities=COMMS,
        PRIVEXP=PRIVEXP,
        GOVEXP=VGP_total,
        SAVE=SAVE[region],
        INCOME=PRIVEXP + VGP_total + SAVE[region],
        VPP=VPP,
        INCPAR={c: INCPAR[(c, region)] for c in COMMS},
        ALPHA={c: SUBPAR[(c, region)] for c in COMMS},
        DPARSUM=DPSM[region],
    )


def main():
    wb_base = xlrd.open_workbook(str(BASEDATA_XLS))
    wb_prm  = xlrd.open_workbook(str(DEFAULT_XLS))
    wb_sl4  = xlrd.open_workbook(str(SL4DUMP_XLS))

    # RunGTAP cumulative percent changes (from sl4dump)
    u_rg       = read_reg(wb_sl4, "0094")
    yev_rg     = read_reg(wb_sl4, "0202")
    EV_rg      = read_reg(wb_sl4, "0208")
    dpsave_rg  = read_reg(wb_sl4, "0087")
    dpgov_rg   = read_reg(wb_sl4, "0088")
    dppriv_rg  = read_reg(wb_sl4, "0089")
    upev_rg    = read_reg(wb_sl4, "0200")
    ugev_rg    = read_reg(wb_sl4, "0199")
    qsaveev_rg = read_reg(wb_sl4, "0201")
    uelasev_rg = read_reg(wb_sl4, "0197")
    ueprivev_rg= read_reg(wb_sl4, "0198")
    ypev_rg    = read_reg(wb_sl4, "0203")
    ygev_rg    = read_reg(wb_sl4, "0204")
    ysaveev_rg = read_reg(wb_sl4, "0206")
    # Main-model income trajectories (real-economy, drive XSHR* evolution)
    y_rg       = read_reg(wb_sl4, "0036")   # regional income
    yg_rg      = read_reg(wb_sl4, "0039")   # gov expenditure
    yp_rg      = read_reg(wb_sl4, "0042")   # private expenditure
    qsave_rg   = read_reg(wb_sl4, "0028")   # real savings
    psave_rg   = read_reg(wb_sl4, "0027")   # savings price
    # ysave = qsave + psave (nominal savings percent change, linearised form)
    ysave_rg = {r: qsave_rg[r] + psave_rg[r] for r in REGIONS}

    print("=" * 102)
    print("Etapa 2 — verificación del shadow demand integrator vs RunGTAP")
    print("=" * 102)

    # The "+MAIN" runs feed real-economy ypriv/ygov/ysave/y from RunGTAP's
    # sl4 into the shadow integrator so XSHRPRIV/GOV/SAVE evolve along the
    # main-model path. Empirically this WORSENS the USA EV gap (the evolved
    # shadow shares amplify the log(UTILSAVEEV) × dpsave contribution beyond
    # what RunGTAP actually computes), so the best Python-side approximation
    # is to keep main shares frozen — the implicit nonlinear trajectory of
    # `dpsave` in RunGTAP's full MCP solve provides cancellation that linear
    # interpolation between baseline and shock cannot.
    schemes = [
        ("euler",          {"n_steps": 10},  False),
        ("euler",          {"n_steps": 20},  False),
        ("euler",          {"n_steps": 25},  False),
        ("euler",          {"n_steps": 30},  False),
        ("euler",          {"n_steps": 50},  False),
        ("bulirsch_stoer", {"bs_ladder": (8, 16, 32)}, False),
    ]
    for method, kwargs, with_main in schemes:
        ladder_label = kwargs.get("bs_ladder") or kwargs.get("n_steps")
        main_label = " +MAIN" if with_main else " (frozen)"
        print(f"\n--- method={method}({ladder_label}){main_label} ---")
        print(f"{'reg':<5}{'var':<10}{'integ':>14}{'RunGTAP':>14}{'gap':>14}{'gap_pct':>12}")
        for region in REGIONS:
            base = build_baseline(region, wb_base, wb_prm)
            main_inputs = (
                dict(
                    ypriv_pct=yp_rg[region],
                    ygov_pct=yg_rg[region],
                    ysave_pct=ysave_rg[region],
                    y_pct=y_rg[region],
                ) if with_main else {}
            )
            res = integrate(
                base,
                u_pct=u_rg[region],
                dppriv_pct=dppriv_rg[region],
                dpgov_pct=dpgov_rg[region],
                dpsave_pct=dpsave_rg[region],
                method=method,
                **kwargs,
                **main_inputs,
            )
            for name, ours, theirs in (
                ("yev",       res.yev_pct,       yev_rg[region]),
                ("ypev",      res.ypev_pct,      ypev_rg[region]),
                ("ygev",      res.ygev_pct,      ygev_rg[region]),
                ("ysaveev",   res.ysaveev_pct,   ysaveev_rg[region]),
                ("upev",      res.upev_pct,      upev_rg[region]),
                ("ugev",      res.ugev_pct,      ugev_rg[region]),
                ("qsaveev",   res.qsaveev_pct,   qsaveev_rg[region]),
                ("uelasev",   res.uelasev_pct,   uelasev_rg[region]),
                ("ueprivev",  res.ueprivev_pct,  ueprivev_rg[region]),
                ("EV ($M)",   res.EV_USDm,       EV_rg[region]),
            ):
                gap = ours - theirs
                gap_pct = (gap / theirs * 100) if theirs else float("nan")
                print(f"{region:<5}{name:<10}{ours:>14.4f}{theirs:>14.4f}{gap:>+14.4f}{gap_pct:>+11.2f}%")


if __name__ == "__main__":
    main()
