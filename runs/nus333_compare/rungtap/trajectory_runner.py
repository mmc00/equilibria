"""Drive RunGTAP at multiple shock magnitudes to map the non-linear
trajectory of `dpsave` (and `u`, `yev`, etc.) under the capFix-with-swap
closure. Outputs a JSON file with the cumulative percent changes at each
shock magnitude — fed downstream to the shadow integrator as a
piecewise-linear trajectory to close the EV residual gap.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
import xlrd

HERE = Path(__file__).resolve().parent
GTAP_EXE = Path(r"C:\runGTAP375\gtapv7.exe")
HAR2XLS  = Path(r"C:\GP\har2xls.exe")
SLTOHT   = Path(r"C:\GP\sltoht.exe")

SHOCK_MAGNITUDES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
VARS_TO_EXTRACT = {
    "u":         "0094",
    "yev":       "0202",
    "EV":        "0208",
    "dpsave":    "0087",
    "upev":      "0200",
    "ugev":      "0199",
    "qsaveev":   "0201",
    "uelasev":   "0197",
    "ueprivev":  "0198",
    "ypev":      "0203",
    "ygev":      "0204",
    "ysaveev":   "0206",
    "y":         "0036",
    "yg":        "0039",
    "yp":        "0042",
    "qsave":     "0028",
    "psave":     "0027",
}
REGIONS = ("USA", "ROW")


def write_cmf(shock_pct: float, tag: str) -> Path:
    cmf = HERE / f"tm_traj_{tag}.cmf"
    cmf.write_text(
        f"""! NUS333 uniform {shock_pct:g}% shock to tm — trajectory step
Auxiliary files = C:\\runGTAP375\\gtapv7 ;
file GTAPSETS = sets.har ;
file GTAPDATA = basedata.har ;
file GTAPPARM = default.prm ;
file GTAPSUM  = summary_{tag}.har ;
file WELVIEW  = decomp_{tag}.har ;
file GTAPVOL  = volume_{tag}.har ;
Updated file GTAPDATA = updated_{tag}.har ;
Method = Gragg ;
Steps  = 8 16 32 ;
Automatic accuracy = no ;
Subintervals = 1 ;
Verbal Description =
NUS333 trajectory step {shock_pct:g}pct ;
Exogenous
          pop
          psaveslack pfactwld
          profitslack incomeslack endwslack
          cgdslack
          tradslack
          ams atm atf ats atd
          aosec aoreg avasec avareg
          aintsec aintreg aintall
          afcom afsec afreg afecom afesec afereg
          aoall afall afeall
          au dppriv dpgov dpsave
          to tinc
          tpreg tm tms tx txs
          qe
          qesf ;
Rest Endogenous ;
swap dpsave("USA") = del_tbalry("USA") ;
Shock tm = uniform {shock_pct:g} ;
CPU = yes ;
NDS = yes ;
log file = yes ;
Extrapolation accuracy file = NO ;
"""
    )
    return cmf


def write_sltoht_sti(tag: str) -> Path:
    sti = HERE / f"sl4dump_{tag}.sti"
    sti.write_text(
        f"\ntm_traj_{tag}\nc\nn\nhar\nsl4dump_{tag}.har\ne\n"
    )
    return sti


def read_reg_var(wb, sheet_id: str) -> dict[str, float]:
    s = wb.sheet_by_name(sheet_id)
    out = {}
    for r in range(s.nrows):
        try: int(s.cell_value(r, 0))
        except (ValueError, TypeError): continue
        out[str(s.cell_value(r, 1)).strip()] = float(s.cell_value(r, 2))
    return out


def run_one(shock_pct: float) -> dict[str, dict[str, float]]:
    tag = f"{int(round(shock_pct * 10)):03d}"  # 010, 020, ..., 100
    cmf = write_cmf(shock_pct, tag)
    print(f"  [{shock_pct:5.2f}%] running gtapv7…", end=" ", flush=True)
    subprocess.run(
        [str(GTAP_EXE), "-cmf", str(cmf.name)],
        cwd=str(HERE), check=True, capture_output=True,
    )
    sti = write_sltoht_sti(tag)
    subprocess.run(
        [str(SLTOHT), "-sti", str(sti.name)],
        cwd=str(HERE), check=True, capture_output=True,
    )
    har = HERE / f"sl4dump_{tag}.har"
    xls = HERE / f"sl4dump_{tag}.xls"
    subprocess.run(
        [str(HAR2XLS), str(har.name), str(xls.name)],
        cwd=str(HERE), check=True, capture_output=True,
    )
    wb = xlrd.open_workbook(str(xls))
    out: dict[str, dict[str, float]] = {}
    for vname, sid in VARS_TO_EXTRACT.items():
        try:
            out[vname] = read_reg_var(wb, sid)
        except Exception as e:
            print(f"!!! missing {vname}({sid}): {e}", end=" ")
            out[vname] = {}
    print("ok")
    return out


def main():
    trajectory = {}
    print(f"Mapping trajectory at shocks {SHOCK_MAGNITUDES}…")
    for s in SHOCK_MAGNITUDES:
        trajectory[s] = run_one(float(s))
    out_path = HERE.parent / "trajectory.json"
    out_path.write_text(json.dumps(trajectory, indent=2))
    print(f"\nWrote {out_path}")

    # Quick table: u, dpsave, EV at each shock magnitude
    print(f"\n{'shock':>6} | {'u USA':>8} {'dpsave USA':>11} {'EV USA':>10} | {'u ROW':>8} {'dpsave ROW':>11} {'EV ROW':>12}")
    for s in SHOCK_MAGNITUDES:
        t = trajectory[s]
        print(
            f"{s:>5}% | {t['u']['USA']:>8.4f} {t['dpsave']['USA']:>11.4f} {t['EV']['USA']:>10.1f} | "
            f"{t['u']['ROW']:>8.4f} {t['dpsave']['ROW']:>11.4f} {t['EV']['ROW']:>12.1f}"
        )


if __name__ == "__main__":
    main()
