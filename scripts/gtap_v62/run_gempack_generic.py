"""Run GEMPACK Gragg-multi on an arbitrary v6.2 dataset.

Generalises run_gempack_exp1a_multistep.py to accept arbitrary
dataset, shock commodity/source/destination.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.gtap_v62.run_gempack_oracle import (  # noqa: E402
    DEFAULT_RUNGTAP_DIR,
    GTAP_EXE_NAME,
    SLTOHTA_EXE_NAME,
    _locate_executable,
)


def _build_cmf(
    *,
    exp_name: str,
    shock_comm: str,
    shock_src: str,
    shock_dst: str,
    method: str = "Gragg",
    steps: str = "2 4 6",
    auto_acc: str = "no",
) -> str:
    return f"""\
! GraggBulirsch multi-step variant ({steps})
file GTAPSETS = SETS.HAR ;
file GTAPDATA = basedata.har ;
file GTAPPARM = Default.prm ;
Auxiliary files = gtap ;
solution file = {exp_name} ;
Updated File GTAPDATA = {exp_name}-upd.har ;

Verbal Description =
10% tariff cut: tms({shock_comm},{shock_src},{shock_dst}) = -10 ;
Method = {method};
Steps = {steps};
automatic accuracy = {auto_acc};
subintervals = 1;

! Standard GE closure: psave varies by region, pfactwld is numeraire
 Exogenous
          pop
          psaveslack pfactwld
          profitslack incomeslack endwslack
          cgdslack tradslack
          ams atm atf ats atd
          aosec aoreg avasec avareg
          afcom afsec afreg afecom afesec afereg
          aoall afall afeall
          au dppriv dpgov dpsave
          to tp tm tms tx txs
          qo(ENDW_COMM,REG) ;
 Rest Endogenous ;
Shock tms("{shock_comm}","{shock_src}","{shock_dst}") = -10 ;
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", required=True, type=Path)
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--shock-comm", required=True)
    p.add_argument("--shock-src", required=True)
    p.add_argument("--shock-dst", required=True)
    p.add_argument("--exp-name", default="Shock1")
    p.add_argument("--steps", default="2 4 6")
    p.add_argument("--rungtap-dir", type=Path, default=DEFAULT_RUNGTAP_DIR)
    args = p.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    # Stage dataset files (try both cases for filenames).
    for fname in [
        "SETS.HAR", "sets.har", "basedata.har",
        "Default.prm", "default.prm", "CMFSTART",
    ]:
        src = args.dataset_dir / fname
        if src.exists():
            shutil.copy2(src, args.workdir / fname)

    # Ensure SETS.HAR / Default.prm exist with the standard casing GEMPACK uses.
    for canonical, alts in [("SETS.HAR", ["sets.har"]), ("Default.prm", ["default.prm"])]:
        canon_path = args.workdir / canonical
        if not canon_path.exists():
            for alt in alts:
                alt_path = args.workdir / alt
                if alt_path.exists():
                    shutil.copy2(alt_path, canon_path)
                    break

    cmf = _build_cmf(
        exp_name=args.exp_name,
        shock_comm=args.shock_comm,
        shock_src=args.shock_src,
        shock_dst=args.shock_dst,
        steps=args.steps,
    )
    cmf_path = args.workdir / f"{args.exp_name}.cmf"
    cmf_path.write_text(cmf, encoding="latin-1")
    print(f"Wrote CMF: {cmf_path}")

    gtap_exe = _locate_executable(GTAP_EXE_NAME, args.rungtap_dir)
    sltohta_exe = _locate_executable(SLTOHTA_EXE_NAME, args.rungtap_dir)

    print(f"Running {gtap_exe.name} -cmf {cmf_path.name}...")
    proc = subprocess.run(
        [str(gtap_exe), "-cmf", cmf_path.name],
        cwd=args.workdir, capture_output=True, text=True,
        timeout=900, check=False,
    )
    log_path = args.workdir / f"{args.exp_name}.gempack.log"
    log_path.write_text(
        f"$ {gtap_exe.name} -cmf {cmf_path.name}\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}\n"
        f"--- rc: {proc.returncode}\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        print(f"gtap.exe failed (rc={proc.returncode}); see {log_path}")
        print("\n--- last 30 stdout lines ---")
        for line in proc.stdout.splitlines()[-30:]:
            print(line)
        return proc.returncode

    sl4 = next(args.workdir.glob(f"{args.exp_name}*.sl4"), None)
    if sl4 is None:
        print(f"No .sl4 produced; see {log_path}")
        return 1

    print(f"Solved. .sl4: {sl4.name}")
    sol_har = args.workdir / f"{args.exp_name}_sol.har"
    proc = subprocess.run(
        [str(sltohta_exe), sl4.name, sol_har.name],
        cwd=args.workdir, capture_output=True, text=True,
        timeout=60, check=False,
    )
    if proc.returncode != 0:
        print(f"sltohta failed: {proc.stdout}\n{proc.stderr}")
        return proc.returncode
    print(f"Done.\n  Updated HAR: {args.workdir / (args.exp_name + '-upd.har')}")
    print(f"  Solution HAR: {sol_har}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
