#!/usr/bin/env python3
"""
Run PEP model solver.

This script solves the calibrated PEP model using the equation system.

Usage:
    python run_solver.py
    python run_solver.py --calibrated-state output/calibrated_state.json
    python run_solver.py --save-solution output/solution.json
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from equilibria.qa.reporting import format_report_summary
from equilibria.qa.sam_checks import run_sam_qa_from_file
from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_model_solver import PEPModelSolver


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Solve calibrated PEP model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calibrate and solve in one step
  python run_solver.py

  # Use pre-calibrated state
  python run_solver.py --calibrated-state output/state.json

  # Save solution
  python run_solver.py --save-solution output/solution.json

  # Custom tolerance
  python run_solver.py --tolerance 1e-8
        """,
    )

    parser.add_argument(
        "--sam-file",
        type=Path,
        default=Path("/Users/marmol/proyectos/cge_babel/pep_static_clean/data/original/SAM-V2_0.gdx"),
        help="Path to SAM GDX file",
    )
    parser.add_argument(
        "--val-par-file",
        type=Path,
        default=None,
        help="Optional path to VAL_PAR file",
    )
    parser.add_argument(
        "--calibrated-state",
        type=Path,
        default=None,
        help="Path to pre-calibrated state JSON file (optional)",
    )
    parser.add_argument(
        "--save-solution",
        type=Path,
        default=None,
        help="Save solution to JSON file",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Convergence tolerance",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum iterations",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "ipopt", "simple_iteration"],
        help="Solver method (auto: try IPOPT first, fall back to simple iteration)",
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
        help="Initialization mode for solver state (gams or excel)",
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
    parser.add_argument(
        "--sam-qa-report",
        type=Path,
        default=None,
        help="Optional path to save SAM QA report JSON",
    )
    parser.add_argument(
        "--sam-qa-balance-rel-tol",
        type=float,
        default=1e-6,
        help="Relative tolerance for SAM balance contracts",
    )
    parser.add_argument(
        "--sam-qa-gdp-rel-tol",
        type=float,
        default=0.08,
        help="Relative tolerance for SAM GDP proxy closure contract",
    )
    parser.add_argument(
        "--sam-qa-max-samples",
        type=int,
        default=8,
        help="Max failure samples stored per SAM QA check",
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=None,
        help="Optional gams baseline manifest JSON",
    )
    parser.add_argument(
        "--require-baseline-manifest",
        action="store_true",
        help="Require baseline manifest when running gams",
    )
    parser.add_argument(
        "--disable-strict-gams-baseline-check",
        action="store_true",
        help="Disable gams baseline compatibility gate",
    )
    parser.add_argument(
        "--strict-gams-baseline-rel-tol",
        type=float,
        default=1e-4,
        help="Relative tolerance for gams baseline anchor checks",
    )
    parser.add_argument(
        "--gdxdump-bin",
        type=str,
        default="/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump",
        help="Path to gdxdump binary for gams baseline checks",
    )
    parser.add_argument(
        "--blockwise-commodity-alpha",
        type=float,
        default=0.75,
        help="Damping alpha for blockwise commodity balance updates",
    )
    parser.add_argument(
        "--blockwise-trade-market-alpha",
        type=float,
        default=0.5,
        help="Damping alpha for blockwise trade market-clearing updates",
    )
    parser.add_argument(
        "--blockwise-macro-alpha",
        type=float,
        default=1.0,
        help="Damping alpha for blockwise macro closure updates",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    print("=" * 70)
    print("PEP-1-1_v2_1 MODEL SOLVER")
    print("=" * 70)
    print()

    # Get calibrated state
    if args.calibrated_state and args.calibrated_state.exists():
        print(f"Loading calibrated state from: {args.calibrated_state}")
        import json
        with open(args.calibrated_state) as f:
            state_data = json.load(f)

        # Recreate state object (simplified)
        from equilibria.templates.pep_calibration_unified import PEPModelState
        state = PEPModelState(**state_data)
        print("✓ Loaded calibrated state")
    else:
        print("Running calibration first...")
        print()

        if not args.sam_file.exists():
            print(f"Error: SAM file not found: {args.sam_file}")
            return 1

        if args.sam_qa_mode != "off":
            qa_report = run_sam_qa_from_file(
                sam_file=args.sam_file,
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

        calibrator = PEPModelCalibrator(
            sam_file=args.sam_file,
            val_par_file=args.val_par_file,
            dynamic_sets=args.dynamic_sets,
        )
        state = calibrator.calibrate()
        calibrator.print_report()
        print()

    # Create solver and solve
    print("=" * 70)
    print("SOLVING MODEL")
    print("=" * 70)
    print()

    solver = PEPModelSolver(
        calibrated_state=state,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        init_mode=args.init_mode,
        blockwise_commodity_alpha=args.blockwise_commodity_alpha,
        blockwise_trade_market_alpha=args.blockwise_trade_market_alpha,
        blockwise_macro_alpha=args.blockwise_macro_alpha,
        gams_results_gdx=args.gams_results_gdx,
        gams_results_slice=args.gams_results_slice,
        baseline_manifest=args.baseline_manifest,
        require_baseline_manifest=args.require_baseline_manifest,
        baseline_compatibility_rel_tol=args.strict_gams_baseline_rel_tol,
        enforce_strict_gams_baseline=(not args.disable_strict_gams_baseline_check),
        sam_file=args.sam_file,
        val_par_file=args.val_par_file,
        gdxdump_bin=args.gdxdump_bin,
    )

    try:
        solution = solver.solve(method=args.method)
    except RuntimeError as exc:
        print(f"✗ Solver initialization failed: {exc}")
        return 2

    # Print results
    print()
    print(solution.summary())
    print()

    # Validate
    validation = solver.validate_solution(solution)

    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print(f"\nRMS Residual: {validation['rms_residual']:.2e}")
    print(f"Max Residual: {validation['max_residual']:.2e}")

    print("\nChecks:")
    for check_name, check_result in validation['checks'].items():
        status = "✓" if check_result.get('passed', False) else "✗"
        print(f"  {status} {check_name}")
        if 'error' in check_result:
            print(f"      Error: {check_result['error']:.2e}")

    # Save solution if requested
    if args.save_solution:
        args.save_solution.parent.mkdir(parents=True, exist_ok=True)

        solution_data = {
            "converged": solution.converged,
            "iterations": solution.iterations,
            "final_residual": solution.final_residual,
            "message": solution.message,
            "validation": validation,
            "key_variables": {
                "GDP_BP": solution.variables.GDP_BP,
                "GDP_MP": solution.variables.GDP_MP,
                "GDP_IB": solution.variables.GDP_IB,
                "GDP_FD": solution.variables.GDP_FD,
                "total_consumption": sum(solution.variables.CTH.values()),
                "total_investment": solution.variables.IT,
                "trade_balance": sum(solution.variables.EXD.values()) - sum(solution.variables.IM.values()),
                "PIXCON": solution.variables.PIXCON,
                "exchange_rate": solution.variables.e,
            },
        }

        import json
        with open(args.save_solution, 'w') as f:
            json.dump(solution_data, f, indent=2)

        print(f"\n✓ Solution saved to: {args.save_solution}")

    print()
    print("=" * 70)
    if solution.converged:
        print("✓ MODEL SOLVED SUCCESSFULLY")
    else:
        print("✗ MODEL SOLUTION INCOMPLETE")
    print("=" * 70)

    return 0 if solution.converged else 1


if __name__ == "__main__":
    sys.exit(main())
