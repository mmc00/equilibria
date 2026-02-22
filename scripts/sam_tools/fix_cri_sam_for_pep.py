#!/usr/bin/env python3
"""
Transform and rebalance a CRI SAM into a PEP-compatible structure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from equilibria.templates.pep_calibration_unified_excel import PEPModelCalibratorExcel
from equilibria.templates.pep_model_solver import PEPModelSolver
from equilibria.templates.pep_sam_compat import transform_sam_to_pep_compatible


def run_optional_pep_check(
    sam_file: Path,
    max_iterations: int,
) -> dict[str, Any]:
    try:
        state = PEPModelCalibratorExcel(sam_file=sam_file).calibrate()
        solver = PEPModelSolver(
            calibrated_state=state,
            tolerance=1e-6,
            max_iterations=max_iterations,
            init_mode="excel",
            sam_file=sam_file,
        )
        solution = solver.solve(method="simple_iteration")
        vars_ = solution.variables
        return {
            "ok": True,
            "converged": bool(solution.converged),
            "iterations": int(solution.iterations),
            "final_residual": float(solution.final_residual),
            "LEON": float(vars_.LEON),
            "GDP_MP": float(vars_.GDP_MP),
            "GDP_FD": float(vars_.GDP_FD),
            "GDP_gap_FD_minus_MP": float(vars_.GDP_FD - vars_.GDP_MP),
        }
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"ok": False, "error": str(exc)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform and rebalance CRI SAM into PEP-compatible structure.",
    )
    parser.add_argument(
        "--input-sam",
        type=Path,
        default=Path("src/equilibria/templates/reference/pep2/data/SAM-CRI-gams-fixed.xlsx"),
        help="Input SAM Excel path",
    )
    parser.add_argument(
        "--output-sam",
        type=Path,
        default=Path("output/SAM-CRI-pep-compatible.xlsx"),
        help="Output transformed SAM Excel path",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("output/SAM-CRI-pep-compatible-report.json"),
        help="Path to save JSON diagnostics report",
    )
    parser.add_argument(
        "--target-mode",
        choices=("geomean", "average", "original"),
        default="geomean",
        help="Rebalance target mode",
    )
    parser.add_argument(
        "--margin-commodity",
        type=str,
        default="ser",
        help="Commodity row used to absorb MARG->I reallocation",
    )
    parser.add_argument("--epsilon", type=float, default=1e-9, help="IPFP epsilon seed")
    parser.add_argument("--tol", type=float, default=1e-8, help="IPFP convergence tolerance")
    parser.add_argument("--max-iter", type=int, default=20000, help="Max IPFP iterations")
    parser.add_argument(
        "--run-pep-check",
        action="store_true",
        help="Run a quick excel-init solver check on output SAM",
    )
    parser.add_argument(
        "--pep-max-iterations",
        type=int,
        default=200,
        help="Iterations for optional quick PEP check",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = transform_sam_to_pep_compatible(
        input_sam=args.input_sam,
        output_sam=args.output_sam,
        report_json=args.report_json,
        target_mode=args.target_mode,
        margin_commodity=args.margin_commodity,
        epsilon=args.epsilon,
        tol=args.tol,
        max_iter=args.max_iter,
    )

    if args.run_pep_check:
        report["pep_check"] = run_optional_pep_check(args.output_sam, args.pep_max_iterations)
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Transformation complete")
    print(f"  input:  {args.input_sam}")
    print(f"  output: {args.output_sam}")
    print(f"  report: {args.report_json}")
    print(f"  ignored inflows before: {report['before']['pep_compatibility']['totals']['ignored_inflows']:.6f}")
    print(f"  ignored inflows after : {report['after']['pep_compatibility']['totals']['ignored_inflows']:.6f}")
    print(f"  max row-col diff after: {report['after']['balance']['max_row_col_abs_diff']:.6e}")
    if args.run_pep_check:
        pep_check = report.get("pep_check", {})
        if pep_check.get("ok"):
            print(
                "  pep_check: "
                f"LEON={pep_check['LEON']:.6f}, "
                f"GDP_FD-GDP_MP={pep_check['GDP_gap_FD_minus_MP']:.6f}, "
                f"res={pep_check['final_residual']:.6e}"
            )
        else:
            print(f"  pep_check: failed ({pep_check.get('error', 'unknown error')})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
