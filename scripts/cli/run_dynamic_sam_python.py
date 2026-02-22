#!/usr/bin/env python3
"""
Run Python PEP calibration equivalent to dynamic SAM GAMS template.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from equilibria.templates.pep_calibration_unified_dynamic import (  # noqa: E402
    PEPModelCalibratorDynamicSAM,
    PEPModelCalibratorExcelDynamicSAM,
)
from equilibria.templates.pep_sam_compat import (  # noqa: E402
    should_apply_cri_pep_fix,
    transform_sam_to_pep_compatible,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dynamic-SAM PEP calibration in Python")
    parser.add_argument("--sam-file", type=Path, required=True, help="Path to SAM (.gdx or .xlsx)")
    parser.add_argument("--val-par-file", type=Path, default=None, help="Optional VAL_PAR file")
    parser.add_argument("--acc-gvt", type=str, default="gvt")
    parser.add_argument("--acc-row", type=str, default="row")
    parser.add_argument("--acc-td", type=str, default="td")
    parser.add_argument("--acc-ti", type=str, default="ti")
    parser.add_argument("--acc-tm", type=str, default="tm")
    parser.add_argument("--acc-tx", type=str, default="tx")
    parser.add_argument("--acc-inv", type=str, default="inv")
    parser.add_argument("--acc-vstk", type=str, default="vstk")
    parser.add_argument(
        "--cri-fix-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="Apply CRI->PEP SAM compatibility transform before calibration",
    )
    parser.add_argument("--cri-fix-output", type=Path, default=None)
    parser.add_argument("--cri-fix-report", type=Path, default=None)
    parser.add_argument(
        "--cri-fix-target-mode",
        choices=["geomean", "average", "original"],
        default="geomean",
    )
    parser.add_argument("--cri-fix-margin-commodity", type=str, default="ser")
    parser.add_argument("--save-state", type=Path, default=None)
    parser.add_argument("--save-report", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

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

    if not args.sam_file.exists():
        print(f"SAM file not found: {args.sam_file}")
        return 1

    sam_file_for_run = args.sam_file
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

    ext = sam_file_for_run.suffix.lower()
    if ext == ".xlsx":
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

    if args.save_state:
        args.save_state.parent.mkdir(parents=True, exist_ok=True)
        state.save_json(args.save_state)
        print(f"Saved state to: {args.save_state}")

    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        calibrator.save_report(args.save_report)
        print(f"Saved report to: {args.save_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
