"""Re-run BOOK3X3 Exp1a under GEMPACK with multi-step extrapolation.

The default Exp1a.exp specifies ``Method = Johansen; Steps = 1`` — a
single linearization step at the SAM. For shocks > 5% this systematically
underestimates the non-linear response. To recover the same answer a
levels-MCP solver produces, GEMPACK needs Gragg-Bulirsch (or Euler)
multi-step with Richardson extrapolation, typically ``Steps = 2 4 6``.

This script:
  1. Stages a workdir with the BOOK3X3 dataset.
  2. Writes Exp1a_GB246.cmf with Method=GraggBulirsch + Steps=2 4 6.
  3. Runs gtap.exe and converts the .sl4 to .har.
  4. Compares the multi-step result to the Johansen-1 result.

Usage::

    python scripts/gtap_v62/run_gempack_exp1a_multistep.py \\
        --workdir runs/gtap_v62_oracle/BOOK3X3_Exp1a_GB246
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.gtap_v62.run_gempack_oracle import (  # noqa: E402
    DEFAULT_RUNGTAP_DIR,
    GTAP_EXE_NAME,
    SLTOHTA_EXE_NAME,
    _locate_executable,
)

logger = logging.getLogger(__name__)


def _build_body(method_block: str, verbal: str) -> str:
    return f"""\
file gtapPARM = Default.prm;
Verbal Description =
Experiment 1: 10% cut of tariff on US food exports to the EU ({verbal});
{method_block}
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
Shock tms("food","usa","eu")= -10 ;
"""


# Predefined method blocks (callable via --variant).
METHOD_BLOCKS = {
    "GB246": (
        "Method = Gragg;\nSteps = 2 4 6;\n"
        "automatic accuracy = no;\nsubintervals = 1;",
        "Gragg-Bulirsch multi-step 2-4-6",
    ),
    "GB48-12": (
        "Method = Gragg;\nSteps = 4 8 12;\n"
        "automatic accuracy = no;\nsubintervals = 1;",
        "Gragg-Bulirsch multi-step 4-8-12",
    ),
    "GB_auto": (
        "Method = Gragg;\nSteps = 2 4 6;\n"
        "automatic accuracy = yes;\nsubintervals = 1;",
        "Gragg-Bulirsch with automatic accuracy",
    ),
    "GB_12_24_48": (
        "Method = Gragg;\nSteps = 12 24 48;\n"
        "automatic accuracy = no;\nsubintervals = 1;",
        "Gragg-Bulirsch multi-step 12-24-48 (high-stencil convergence test)",
    ),
    "GB_24_48_96": (
        "Method = Gragg;\nSteps = 24 48 96;\n"
        "automatic accuracy = no;\nsubintervals = 1;",
        "Gragg-Bulirsch multi-step 24-48-96 (very high stencil)",
    ),
}


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", required=True, type=Path,
        help="Output directory for the GraggBulirsch run.",
    )
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=Path("C:/runGTAP375/BOOK3X3"),
    )
    parser.add_argument(
        "--rungtap-dir", type=Path, default=DEFAULT_RUNGTAP_DIR,
    )
    parser.add_argument(
        "--variant", choices=list(METHOD_BLOCKS), default="GB246",
        help="Stepping variant (see METHOD_BLOCKS).",
    )
    args = parser.parse_args(argv)

    args.workdir.mkdir(parents=True, exist_ok=True)
    exp_name = f"Exp1a_{args.variant}"
    method_block, verbal = METHOD_BLOCKS[args.variant]
    body = _build_body(method_block, verbal)

    # 1. Stage dataset files.
    for fname in ("SETS.HAR", "basedata.har", "Default.prm", "CMFSTART"):
        src = args.dataset_dir / fname
        if src.exists():
            shutil.copy2(src, args.workdir / fname)

    # 2. Write the CMF with explicit file declarations + GB method.
    header = (
        f"! GraggBulirsch multi-step variant of Exp1a (non-linear correction)\n"
        f"file GTAPSETS = SETS.HAR ;\n"
        f"file GTAPDATA = basedata.har ;\n"
        f"file GTAPPARM = Default.prm ;\n"
        f"Auxiliary files = gtap ;\n"
        f"solution file = {exp_name} ;\n"
        f"Updated File GTAPDATA = {exp_name}-upd.har ;\n"
    )
    # Strip the body's local "file gtapPARM = ..." line (we set it in header).
    body_lines = [
        line for line in body.splitlines()
        if not line.strip().lower().startswith("file gtapparm")
    ]
    cmf_path = args.workdir / f"{exp_name}.cmf"
    cmf_path.write_text(header + "\n".join(body_lines) + "\n", encoding="latin-1")
    print(f"Wrote CMF: {cmf_path}")

    # 3. Run gtap.exe.
    gtap_exe = _locate_executable(GTAP_EXE_NAME, args.rungtap_dir)
    sltohta_exe = _locate_executable(SLTOHTA_EXE_NAME, args.rungtap_dir)

    print(f"Running {gtap_exe.name} -cmf {cmf_path.name}...")
    proc = subprocess.run(
        [str(gtap_exe), "-cmf", cmf_path.name],
        cwd=args.workdir, capture_output=True, text=True,
        timeout=600, check=False,
    )
    log_path = args.workdir / f"{exp_name}.gempack.log"
    log_path.write_text(
        f"$ {gtap_exe.name} -cmf {cmf_path.name}\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}\n"
        f"--- rc: {proc.returncode}\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        print(f"gtap.exe failed (rc={proc.returncode}); see {log_path}")
        print("--- last 30 lines of stdout ---")
        print("\n".join(proc.stdout.splitlines()[-30:]))
        return proc.returncode

    sl4_candidates = list(args.workdir.glob(f"{exp_name}*.sl4"))
    if not sl4_candidates:
        print(f"No .sl4 produced; see {log_path}")
        return 1
    sl4 = sl4_candidates[0]
    print(f"Solved. .sl4 produced: {sl4.name}")

    # 4. Convert .sl4 -> .har.
    sol_har = args.workdir / f"{exp_name}_sol.har"
    proc = subprocess.run(
        [str(sltohta_exe), sl4.name, sol_har.name],
        cwd=args.workdir, capture_output=True, text=True,
        timeout=60, check=False,
    )
    if proc.returncode != 0:
        print(f"sltohta.exe failed (rc={proc.returncode})")
        print(proc.stdout)
        print(proc.stderr)
        return proc.returncode

    print(f"\nDone. Updated HAR: {args.workdir / (exp_name + '-upd.har')}")
    print(f"      Solution HAR: {sol_har}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
