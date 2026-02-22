#!/usr/bin/env python3
"""
Run complete PEP model calibration.

This script runs all five calibration phases and generates a comprehensive report.

Usage:
    python run_all_calibration.py
    python run_all_calibration.py --save-state output/calibrated_state.json
    python run_all_calibration.py --save-report output/calibration_report.json
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from equilibria.qa.reporting import format_report_summary
from equilibria.qa.sam_checks import run_sam_qa_from_file
from equilibria.templates.pep_calibration_unified import PEPModelCalibrator


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run complete PEP model calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run calibration with default paths
  python run_all_calibration.py

  # Save calibrated state to file
  python run_all_calibration.py --save-state output/state.json

  # Save detailed report
  python run_all_calibration.py --save-report output/report.json

  # Run with verbose output
  python run_all_calibration.py --verbose
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
        help="Path to VAL_PAR Excel file (optional)",
    )
    parser.add_argument(
        "--save-state",
        type=Path,
        default=None,
        help="Save calibrated model state to JSON file",
    )
    parser.add_argument(
        "--save-report",
        type=Path,
        default=None,
        help="Save calibration report to JSON file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
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
    parser.add_argument("--sam-qa-balance-rel-tol", type=float, default=1e-6)
    parser.add_argument("--sam-qa-gdp-rel-tol", type=float, default=0.08)
    parser.add_argument("--sam-qa-max-samples", type=int, default=8)

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    print("=" * 70)
    print("PEP-1-1_v2_1 MODEL CALIBRATION")
    print("Complete Calibration Pipeline")
    print("=" * 70)
    print()

    # Check if SAM file exists
    if not args.sam_file.exists():
        print(f"Error: SAM file not found: {args.sam_file}")
        return 1

    print(f"SAM File: {args.sam_file}")
    if args.val_par_file:
        print(f"VAL_PAR File: {args.val_par_file}")
    print()

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
        print()

    try:
        # Create calibrator and run calibration
        calibrator = PEPModelCalibrator(
            sam_file=args.sam_file,
            val_par_file=args.val_par_file,
            dynamic_sets=args.dynamic_sets,
        )

        state = calibrator.calibrate()

        # Print report
        calibrator.print_report()

        # Save outputs if requested
        if args.save_state:
            args.save_state.parent.mkdir(parents=True, exist_ok=True)
            state.save_json(args.save_state)
            print(f"\n✓ Model state saved to: {args.save_state}")

        if args.save_report:
            args.save_report.parent.mkdir(parents=True, exist_ok=True)
            calibrator.save_report(args.save_report)
            print(f"✓ Calibration report saved to: {args.save_report}")

        # Final status
        print("\n" + "=" * 70)
        if calibrator.report.validation_passed:
            print("✓ CALIBRATION COMPLETED SUCCESSFULLY")
        else:
            print("⚠ CALIBRATION COMPLETED WITH WARNINGS")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\n✗ Calibration failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
