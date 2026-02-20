#!/usr/bin/env python3
"""
Compare pep2 GAMS Results.gdx BASE values against Python PEP initialization.

This script is designed for the pep2 workflow in:
  src/equilibria/templates/reference/pep2
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_model_solver import PEPModelSolver


def _gdxdump_value(gdxdump_bin: str, gdx_file: Path, symbol: str, label: str = "BASE") -> float | None:
    """Read one labeled scalar from a GDX symbol dump."""
    out = subprocess.check_output(
        [gdxdump_bin, str(gdx_file), f"symb={symbol}"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    pat = re.compile(r"'([^']+)'\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)")
    values = {m.group(1): float(m.group(2)) for m in pat.finditer(out)}
    return values.get(label)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare pep2 GAMS Results.gdx with Python PEP initialization."
    )
    parser.add_argument(
        "--pep2-root",
        type=Path,
        default=Path("/Users/marmol/proyectos/equilibria/src/equilibria/templates/reference/pep2"),
        help="Path to pep2 root directory.",
    )
    parser.add_argument(
        "--init-mode",
        choices=["gams", "excel"],
        default="excel",
        help="Python initialization mode for comparison.",
    )
    parser.add_argument(
        "--gdxdump",
        default="/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump",
        help="Path to gdxdump binary.",
    )
    parser.add_argument(
        "--gams-results-slice",
        choices=["base", "sim1"],
        default="base",
        help="Slice from Results.gdx for gams levels.",
    )
    args = parser.parse_args()

    pep2_root = args.pep2_root.resolve()
    sam_file = pep2_root / "data" / "SAM-V2_0.gdx"
    val_file = pep2_root / "data" / "VAL_PAR.xlsx"
    results_gdx = pep2_root / "scripts" / "Results.gdx"

    if not sam_file.exists():
        print(f"Missing SAM file: {sam_file}")
        return 1
    if not val_file.exists():
        print(f"Missing VAL_PAR file: {val_file}")
        return 1
    if not results_gdx.exists():
        print(f"Missing GAMS results file: {results_gdx}")
        return 1

    state = PEPModelCalibrator(sam_file=sam_file, val_par_file=val_file).calibrate()
    solver = PEPModelSolver(
        calibrated_state=state,
        init_mode=args.init_mode,
        gams_results_gdx=results_gdx,
        gams_results_slice=args.gams_results_slice,
        sam_file=sam_file,
        val_par_file=val_file,
        gdxdump_bin=args.gdxdump,
    )
    vars_ = solver._create_initial_guess()

    rows = [
        ("valGDP_BP", vars_.GDP_BP, "GDP_BP"),
        ("valGDP_FD", vars_.GDP_FD, "GDP_FD"),
        ("valGDP_IB", vars_.GDP_IB, "GDP_IB"),
        ("valGDP_MP", vars_.GDP_MP, "GDP_MP"),
        ("valIT", vars_.IT, "IT"),
        ("valCAB", vars_.CAB, "CAB"),
        ("valYG", vars_.YG, "YG"),
        ("valTPRODN", vars_.TPRODN, "TPRODN"),
        ("valTIWT", vars_.TIWT, "TIWT"),
        ("valTIKT", vars_.TIKT, "TIKT"),
    ]

    print("PEP2 GAMS vs Python")
    print(f"  pep2_root: {pep2_root}")
    print(f"  init_mode: {args.init_mode}")
    print("-" * 92)
    print(f"{'symbol':<12} {'gams_BASE':>14} {'python':>14} {'diff(py-gams)':>18} {'py_var':<12}")
    print("-" * 92)

    for symbol, py_val, py_name in rows:
        gams_val = _gdxdump_value(args.gdxdump, results_gdx, symbol, "BASE")
        if gams_val is None:
            print(f"{symbol:<12} {'NA':>14} {py_val:>14.6f} {'NA':>18} {py_name:<12}")
            continue
        diff = py_val - gams_val
        print(f"{symbol:<12} {gams_val:>14.6f} {py_val:>14.6f} {diff:>18.6f} {py_name:<12}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
