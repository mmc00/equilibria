"""Etapa 1 — verificación numérica de la fórmula EV de RunGTAP/gtapv7.tab.

Reproduce el cálculo de `EV(r) = (INCOME(r)/100) * UTILELASEV(r) * u(r)` con
los coeficientes baseline de NUS333, y lo compara contra los valores `EV` que
RunGTAP escribe en el sl4 después del shock 10% tm.

Si el predicho coincide con RunGTAP a <1%, queda confirmada la hipótesis
del gap USA: equilibria estaba implícitamente usando `UTILELAS=1` (formula
Cobb-Douglas linealizada) en vez del `UTILELAS<1` que produce el descuento
de elasticidad-ingreso CDE.
"""
from __future__ import annotations
import xlrd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASEDATA_XLS = ROOT / "rungtap" / "basedata.xls"
DEFAULT_XLS  = ROOT / "rungtap" / "default.xls"
SL4DUMP_XLS  = ROOT / "rungtap" / "sl4dump.xls"
REGIONS = ("USA", "ROW")
COMMS   = ("AGR", "MFG", "SER")


def read_reg_var(wb: xlrd.Book, sheet: str) -> dict[str, float]:
    s = wb.sheet_by_name(sheet)
    out = {}
    for r in range(s.nrows):
        try: int(s.cell_value(r, 0))
        except (ValueError, TypeError): continue
        out[str(s.cell_value(r, 1)).strip()] = float(s.cell_value(r, 2))
    return out


def read_comm_reg(wb: xlrd.Book, sheet: str) -> dict[tuple[str,str], float]:
    """Read a (COMM, REG) shaped header from a har2xls dump.

    Layout: 6-7 metadata rows, then a row with REG labels at col 2..,
    then numeric data rows where col 0 = int index, col 1 = COMM name.
    """
    s = wb.sheet_by_name(sheet)
    reg_labels: list[str] = []
    data_start = None
    for r in range(s.nrows):
        # The labels row has REG names ('USA', 'ROW') in cols 2..
        c2 = str(s.cell_value(r, 2)).strip() if s.ncols > 2 else ""
        if c2 in REGIONS:
            reg_labels = [
                str(s.cell_value(r, c)).strip()
                for c in range(2, s.ncols)
                if str(s.cell_value(r, c)).strip() in REGIONS
            ]
            data_start = r + 1
            break
    if data_start is None:
        return {}
    out = {}
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


def main():
    wb_base = xlrd.open_workbook(str(BASEDATA_XLS))
    wb_prm  = xlrd.open_workbook(str(DEFAULT_XLS))
    wb_sl4  = xlrd.open_workbook(str(SL4DUMP_XLS))

    # --- Baseline absorption: VDPP + VMPP = private consumption (purchaser)
    VDPP = read_comm_reg(wb_base, "VDPP")
    VMPP = read_comm_reg(wb_base, "VMPP")
    VDGP = read_comm_reg(wb_base, "VDGP")
    VMGP = read_comm_reg(wb_base, "VMGP")
    SAVE = read_reg_var(wb_base, "SAVE")

    VPP = {k: VDPP.get(k, 0.0) + VMPP.get(k, 0.0) for k in set(list(VDPP) + list(VMPP))}
    VGP = {k: VDGP.get(k, 0.0) + VMGP.get(k, 0.0) for k in set(list(VDGP) + list(VMGP))}

    PRIVEXP = {r: sum(VPP[(c, r)] for c in COMMS if (c, r) in VPP) for r in REGIONS}
    GOVEXP  = {r: sum(VGP[(c, r)] for c in COMMS if (c, r) in VGP) for r in REGIONS}
    SAVING  = {r: SAVE.get(r, 0.0) for r in REGIONS}
    INCOME  = {r: PRIVEXP[r] + GOVEXP[r] + SAVING[r] for r in REGIONS}

    # --- CDE income elasticity: INCP(c,r) from default.prm
    INCPAR = read_comm_reg(wb_prm, "INCP")

    # --- Compute UELASPRIVEV = sum_c CONSHR · INCPAR, then UTILELASEV
    CONSHR = {(c, r): VPP[(c, r)] / PRIVEXP[r] for c in COMMS for r in REGIONS if (c,r) in VPP and PRIVEXP[r] > 0}
    UELASPRIVEV = {
        r: sum(CONSHR[(c, r)] * INCPAR[(c, r)] for c in COMMS if (c, r) in CONSHR and (c, r) in INCPAR)
        for r in REGIONS
    }
    XSHRPRIV = {r: PRIVEXP[r] / INCOME[r] for r in REGIONS}
    XSHRGOV  = {r: GOVEXP[r]  / INCOME[r] for r in REGIONS}
    XSHRSAVE = {r: SAVING[r]  / INCOME[r] for r in REGIONS}

    # Linearized init: UTILELASEV = XSHRPRIV·UELASPRIVEV + XSHRGOV + XSHRSAVE
    # (gov+save sub-utilities are Cobb-Douglas → expenditure elasticity = 1)
    UTILELASEV = {
        r: XSHRPRIV[r] * UELASPRIVEV[r] + XSHRGOV[r] * 1.0 + XSHRSAVE[r] * 1.0
        for r in REGIONS
    }

    # --- RunGTAP outputs (from sl4 cumulative dump, capFix run)
    u_pct = read_reg_var(wb_sl4, "0094")    # u, % change
    EV    = read_reg_var(wb_sl4, "0208")    # EV, USD millions (change var)
    yev   = read_reg_var(wb_sl4, "0202")    # yev, % change in equivalent income

    # --- Predicted EV
    # E_yev: yev = pop + UTILELAS * u   (with pop=0, dppriv=dpgov=dpsave=0 by closure)
    yev_predicho = {r: UTILELASEV[r] * u_pct[r] for r in REGIONS}
    # E_EV:  EV  = (INCOME / 100) * yev
    EV_predicho_via_yev = {r: (INCOME[r] / 100.0) * yev[r] for r in REGIONS}
    EV_predicho_full    = {r: (INCOME[r] / 100.0) * yev_predicho[r] for r in REGIONS}

    # ---------- Reporte ----------
    print("=" * 96)
    print("Etapa 1 — verificación de la fórmula EV de RunGTAP/gtapv7.tab")
    print("=" * 96)
    print(f"{'Región':<8}{'INCOME':>14}{'PRIVEXP':>12}{'GOVEXP':>12}{'SAVE':>12}{'SHRpriv':>10}{'SHRgov':>10}{'SHRsave':>10}")
    for r in REGIONS:
        print(f"{r:<8}{INCOME[r]:>14,.2f}{PRIVEXP[r]:>12,.2f}{GOVEXP[r]:>12,.2f}{SAVING[r]:>12,.2f}"
              f"{XSHRPRIV[r]:>10.4f}{XSHRGOV[r]:>10.4f}{XSHRSAVE[r]:>10.4f}")

    print(f"\n{'Región':<8}{'CONSHR(AGR)':>14}{'CONSHR(MFG)':>14}{'CONSHR(SER)':>14}"
          f"{'INCPAR(AGR)':>14}{'INCPAR(MFG)':>14}{'INCPAR(SER)':>14}")
    for r in REGIONS:
        print(f"{r:<8}"
              f"{CONSHR[('AGR',r)]:>14.4f}{CONSHR[('MFG',r)]:>14.4f}{CONSHR[('SER',r)]:>14.4f}"
              f"{INCPAR[('AGR',r)]:>14.4f}{INCPAR[('MFG',r)]:>14.4f}{INCPAR[('SER',r)]:>14.4f}")

    print(f"\n{'Región':<8}{'UELASPRIV':>14}{'UTILELAS':>14}{'u_pct':>12}{'yev_pred':>12}{'yev_rungtap':>14}"
          f"{'EV_pred_full':>16}{'EV_pred_yev':>16}{'EV_RunGTAP':>14}{'gap_full %':>12}{'gap_yev %':>12}")
    for r in REGIONS:
        gap_full = (EV_predicho_full[r] - EV[r]) / EV[r] * 100 if EV[r] else float('nan')
        gap_yev  = (EV_predicho_via_yev[r] - EV[r]) / EV[r] * 100 if EV[r] else float('nan')
        print(f"{r:<8}{UELASPRIVEV[r]:>14.4f}{UTILELASEV[r]:>14.4f}{u_pct[r]:>12.4f}"
              f"{yev_predicho[r]:>12.4f}{yev[r]:>14.4f}"
              f"{EV_predicho_full[r]:>16,.2f}{EV_predicho_via_yev[r]:>16,.2f}{EV[r]:>14,.2f}"
              f"{gap_full:>+11.2f}%{gap_yev:>+11.2f}%")
    print("\nLeyenda:")
    print("  yev_pred       = UTILELAS · u_pct       (predicho con coeficientes baseline)")
    print("  yev_rungtap    = % cambio que escribe RunGTAP en el sl4")
    print("  EV_pred_full   = (INCOME/100) · UTILELAS · u_pct      (predicción independiente)")
    print("  EV_pred_yev    = (INCOME/100) · yev_rungtap           (sustituye yev de RunGTAP en E_EV)")
    print("  EV_RunGTAP     = valor reportado por RunGTAP en sl4")


if __name__ == "__main__":
    main()
