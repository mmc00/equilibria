#!/usr/bin/env python3
"""
Run PEP model solver using SAM loaded directly from Excel.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from equilibria.qa.reporting import format_report_summary
from equilibria.qa.sam_checks import run_sam_qa_from_file
from equilibria.templates.pep_calibration_unified import PEPModelState
from equilibria.templates.pep_calibration_unified_excel import PEPModelCalibratorExcel
from equilibria.templates.pep_model_solver import PEPModelSolver
from equilibria.templates.pep_sam_compat import (
    should_apply_cri_pep_fix,
    transform_sam_to_pep_compatible,
)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve calibrated PEP model (Excel SAM)")
    parser.add_argument(
        "--sam-file",
        type=Path,
        default=Path("src/equilibria/templates/reference/pep2/data/SAM-V2_0.xls"),
        help="Path to SAM Excel file (.xls/.xlsx)",
    )
    parser.add_argument(
        "--val-par-file",
        type=Path,
        default=Path("src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx"),
        help="Path to VAL_PAR file (.xlsx/.gdx)",
    )
    parser.add_argument("--calibrated-state", type=Path, default=None)
    parser.add_argument("--save-solution", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "ipopt", "simple_iteration"],
    )
    parser.add_argument(
        "--gams-results-gdx",
        type=Path,
        default=Path("src/equilibria/templates/reference/pep2/scripts/Results.gdx"),
        help="GAMS Results.gdx used for gams initialization",
    )
    parser.add_argument(
        "--gams-results-slice",
        choices=["base", "sim1"],
        default="base",
        help="Scenario slice in Results.gdx to use for gams levels",
    )
    parser.add_argument(
        "--init-mode",
        type=str,
        default="excel",
        choices=["gams", "excel"],
    )
    parser.add_argument(
        "--dynamic-sets",
        action="store_true",
        help="Derive model sets dynamically from SAM instead of using template defaults",
    )
    parser.add_argument(
        "--sam-qa-mode",
        choices=["hard_fail", "warn", "off"],
        default="hard_fail",
        help="Pre-calibration SAM QA behavior",
    )
    parser.add_argument("--sam-qa-report", type=Path, default=None)
    parser.add_argument("--sam-qa-balance-rel-tol", type=float, default=1e-6)
    parser.add_argument("--sam-qa-gdp-rel-tol", type=float, default=0.08)
    parser.add_argument("--sam-qa-max-samples", type=int, default=8)
    parser.add_argument("--baseline-manifest", type=Path, default=None)
    parser.add_argument("--require-baseline-manifest", action="store_true")
    parser.add_argument("--disable-strict-gams-baseline-check", action="store_true")
    parser.add_argument("--strict-gams-baseline-rel-tol", type=float, default=1e-4)
    parser.add_argument(
        "--gdxdump-bin",
        type=str,
        default="/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump",
    )
    parser.add_argument(
        "--cri-fix-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="Apply CRI->PEP SAM compatibility transform before calibration",
    )
    parser.add_argument(
        "--cri-fix-output",
        type=Path,
        default=None,
        help="Optional output SAM path for transformed CRI SAM",
    )
    parser.add_argument(
        "--cri-fix-report",
        type=Path,
        default=None,
        help="Optional JSON report path for CRI SAM transform",
    )
    parser.add_argument(
        "--cri-fix-target-mode",
        choices=["geomean", "average", "original"],
        default="geomean",
        help="RAS target mode for CRI SAM transform",
    )
    parser.add_argument(
        "--cri-fix-margin-commodity",
        type=str,
        default="ser",
        help="Commodity that absorbs margin reallocation in CRI transform",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    print("=" * 70)
    print("PEP-1-1_v2_1 MODEL SOLVER (EXCEL SAM)")
    print("=" * 70)
    print()

    sam_file_for_run = args.sam_file
    cri_fix_report = None

    if args.calibrated_state and args.calibrated_state.exists():
        import json

        with open(args.calibrated_state) as f:
            state_data = json.load(f)
        state = PEPModelState(**state_data)
        print(f"✓ Loaded calibrated state: {args.calibrated_state}")
    else:
        if not args.sam_file.exists():
            print(f"Error: SAM file not found: {args.sam_file}")
            return 1

        if should_apply_cri_pep_fix(args.sam_file, mode=args.cri_fix_mode):
            output_sam = (
                args.cri_fix_output
                if args.cri_fix_output
                else Path("output") / f"{args.sam_file.stem}-pep-compatible{args.sam_file.suffix}"
            )
            report_json = (
                args.cri_fix_report
                if args.cri_fix_report
                else Path("output") / f"{args.sam_file.stem}-pep-compatible-report.json"
            )
            cri_fix_report = transform_sam_to_pep_compatible(
                input_sam=args.sam_file,
                output_sam=output_sam,
                report_json=report_json,
                target_mode=args.cri_fix_target_mode,
                margin_commodity=args.cri_fix_margin_commodity,
            )
            sam_file_for_run = output_sam
            print("Applied CRI->PEP SAM compatibility transform")
            print(f"  input : {args.sam_file}")
            print(f"  output: {sam_file_for_run}")
            print(
                "  ignored inflows: "
                f"{cri_fix_report['before']['pep_compatibility']['totals']['ignored_inflows']:.6f}"
                " -> "
                f"{cri_fix_report['after']['pep_compatibility']['totals']['ignored_inflows']:.6f}"
            )

        if args.sam_qa_mode != "off":
            qa_report = run_sam_qa_from_file(
                sam_file=sam_file_for_run,
                dynamic_sam=False,
                balance_rel_tol=args.sam_qa_balance_rel_tol,
                gdp_rel_tol=args.sam_qa_gdp_rel_tol,
                max_samples=args.sam_qa_max_samples,
            )
            print(format_report_summary(qa_report))
            if args.sam_qa_report:
                qa_report.save_json(args.sam_qa_report)
                print(f"Saved SAM QA report: {args.sam_qa_report}")
            if not qa_report.passed:
                if args.sam_qa_mode == "hard_fail":
                    print("✗ SAM QA failed in hard_fail mode. Aborting before calibration.")
                    return 2
                print("⚠ SAM QA failed, continuing because mode=warn.")
        print(f"Running calibration from Excel SAM: {sam_file_for_run}")
        calibrator = PEPModelCalibratorExcel(
            sam_file=sam_file_for_run,
            val_par_file=args.val_par_file if args.val_par_file else None,
            dynamic_sets=args.dynamic_sets,
        )
        state = calibrator.calibrate()
        calibrator.print_report()

    print("\n" + "=" * 70)
    print("SOLVING MODEL")
    print("=" * 70)

    solver = PEPModelSolver(
        calibrated_state=state,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        init_mode=args.init_mode,
        gams_results_gdx=args.gams_results_gdx,
        gams_results_slice=args.gams_results_slice,
        baseline_manifest=args.baseline_manifest,
        require_baseline_manifest=args.require_baseline_manifest,
        baseline_compatibility_rel_tol=args.strict_gams_baseline_rel_tol,
        enforce_strict_gams_baseline=(not args.disable_strict_gams_baseline_check),
        sam_file=sam_file_for_run,
        val_par_file=args.val_par_file,
        gdxdump_bin=args.gdxdump_bin,
    )
    try:
        solution = solver.solve(method=args.method)
    except RuntimeError as exc:
        print(f"✗ Solver initialization failed: {exc}")
        return 2

    print()
    print(solution.summary())
    validation = solver.validate_solution(solution)
    print(f"\nRMS Residual: {validation['rms_residual']:.2e}")
    print(f"Max Residual: {validation['max_residual']:.2e}")

    if args.save_solution:
        import json

        args.save_solution.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_solution, "w") as f:
            json.dump(
                {
                    "converged": solution.converged,
                    "iterations": solution.iterations,
                    "final_residual": solution.final_residual,
                    "message": solution.message,
                    "validation": validation,
                },
                f,
                indent=2,
            )
        print(f"✓ Solution saved to: {args.save_solution}")

    return 0 if solution.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
