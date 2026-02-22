#!/usr/bin/env python3
"""Run PEP dynamic-SAM solver for CRI dataset (GDX or Excel)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from equilibria.qa.reporting import format_report_summary  # noqa: E402
from equilibria.qa.sam_checks import run_sam_qa_from_file  # noqa: E402
from equilibria.templates.pep_calibration_unified_dynamic import (  # noqa: E402
    PEPModelCalibratorDynamicSAM,
    PEPModelCalibratorExcelDynamicSAM,
)
from equilibria.templates.pep_model_solver import PEPModelSolver  # noqa: E402
from equilibria.templates.pep_sam_compat import (  # noqa: E402
    should_apply_cri_pep_fix,
    transform_sam_to_pep_compatible,
)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve CRI dynamic-SAM PEP model in Python")
    parser.add_argument(
        "--sam-file",
        type=Path,
        default=Path("src/equilibria/templates/reference/pep2/data/SAM-CRI-gams-fixed.xlsx"),
    )
    parser.add_argument(
        "--val-par-file",
        type=Path,
        default=Path("src/equilibria/templates/reference/pep2/data/VAL_PAR-CRI-gams.xlsx"),
    )
    parser.add_argument("--acc-gvt", type=str, default="gvt")
    parser.add_argument("--acc-row", type=str, default="row")
    parser.add_argument("--acc-td", type=str, default="td")
    parser.add_argument("--acc-ti", type=str, default="ti")
    parser.add_argument("--acc-tm", type=str, default="tm")
    parser.add_argument("--acc-tx", type=str, default="tx")
    parser.add_argument("--acc-inv", type=str, default="inv")
    parser.add_argument("--acc-vstk", type=str, default="vstk")
    parser.add_argument("--method", choices=["auto", "ipopt", "simple_iteration"], default="auto")
    parser.add_argument(
        "--init-mode",
        choices=["gams", "excel"],
        default="excel",
    )
    parser.add_argument(
        "--gams-results-gdx",
        type=Path,
        default=Path("src/equilibria/templates/reference/pep2/scripts/Results.gdx"),
        help="GAMS Results.gdx used for gams initial levels",
    )
    parser.add_argument(
        "--gams-results-slice",
        choices=["base", "sim1"],
        default="sim1",
        help="Scenario slice in Results.gdx to use for gams levels",
    )
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--save-solution", type=Path, default=None)
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
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    sam_file_for_run = args.sam_file

    if not args.sam_file.exists():
        print(f"SAM file not found: {args.sam_file}")
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

    accounts = {
        "gvt": args.acc_gvt,
        "row": args.acc_row,
        "td": args.acc_td,
        "ti": args.acc_ti,
        "tm": args.acc_tm,
        "tx": args.acc_tx,
        "inv": args.acc_inv,
        "vstk": args.acc_vstk,
    }

    if args.sam_qa_mode != "off":
        qa_report = run_sam_qa_from_file(
            sam_file=sam_file_for_run,
            dynamic_sam=True,
            accounts=accounts,
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

    if sam_file_for_run.suffix.lower() in {".xlsx", ".xls"}:
        calibrator = PEPModelCalibratorExcelDynamicSAM(
            sam_file=sam_file_for_run,
            val_par_file=args.val_par_file,
            accounts=accounts,
        )
    else:
        calibrator = PEPModelCalibratorDynamicSAM(
            sam_file=sam_file_for_run,
            val_par_file=args.val_par_file,
            accounts=accounts,
        )

    state = calibrator.calibrate()
    calibrator.print_report()

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

    print(solution.summary())
    validation = solver.validate_solution(solution)
    print(f"RMS Residual: {validation['rms_residual']:.2e}")
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
        print(f"Saved solution to: {args.save_solution}")

    return 0 if solution.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
