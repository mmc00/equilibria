#!/usr/bin/env python3
"""Generate strict-gams baseline manifest for a calibrated run."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from equilibria.baseline.compatibility import (
    evaluate_strict_gams_baseline_compatibility,
)
from equilibria.baseline.manifest import build_baseline_manifest
from equilibria.templates.pep_calibration_unified import (
    PEPModelCalibrator,
    PEPModelState,
)
from equilibria.templates.pep_calibration_unified_dynamic import (
    PEPModelCalibratorDynamicSAM,
    PEPModelCalibratorExcelDynamicSAM,
)
from equilibria.templates.pep_calibration_unified_excel import PEPModelCalibratorExcel
from equilibria.templates.pep_sam_compat import (
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


def _build_state(args: argparse.Namespace) -> PEPModelState:
    if args.calibrated_state:
        data = json.loads(Path(args.calibrated_state).read_text())
        return PEPModelState(**data)

    if args.sam_file is None:
        raise ValueError("--sam-file is required when --calibrated-state is not provided")

    sam_path = Path(args.sam_file)
    val_path = Path(args.val_par_file) if args.val_par_file else None

    if should_apply_cri_pep_fix(sam_path, mode=args.cri_fix_mode):
        output_sam = (
            args.cri_fix_output
            if args.cri_fix_output
            else Path("output") / f"{sam_path.stem}-pep-compatible{sam_path.suffix}"
        )
        report_json = (
            args.cri_fix_report
            if args.cri_fix_report
            else Path("output") / f"{sam_path.stem}-pep-compatible-report.json"
        )
        cri_fix_report = transform_sam_to_pep_compatible(
            input_sam=sam_path,
            output_sam=output_sam,
            report_json=report_json,
            target_mode=args.cri_fix_target_mode,
            margin_commodity=args.cri_fix_margin_commodity,
        )
        print("Applied CRI->PEP SAM compatibility transform")
        print(f"  input : {sam_path}")
        print(f"  output: {output_sam}")
        print(
            "  ignored inflows: "
            f"{cri_fix_report['before']['pep_compatibility']['totals']['ignored_inflows']:.6f}"
            " -> "
            f"{cri_fix_report['after']['pep_compatibility']['totals']['ignored_inflows']:.6f}"
        )
        sam_path = output_sam
        args.sam_file = sam_path

    is_excel = sam_path.suffix.lower() in {".xls", ".xlsx"}

    if args.dynamic_sam:
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
        if is_excel:
            calibrator = PEPModelCalibratorExcelDynamicSAM(
                sam_file=sam_path,
                val_par_file=val_path,
                accounts=accounts,
            )
        else:
            calibrator = PEPModelCalibratorDynamicSAM(
                sam_file=sam_path,
                val_par_file=val_path,
                accounts=accounts,
            )
        return calibrator.calibrate()

    if is_excel:
        calibrator = PEPModelCalibratorExcel(
            sam_file=sam_path,
            val_par_file=val_path,
            dynamic_sets=args.dynamic_sets,
        )
        return calibrator.calibrate()

    calibrator = PEPModelCalibrator(
        sam_file=sam_path,
        val_par_file=val_path,
        dynamic_sets=args.dynamic_sets,
    )
    return calibrator.calibrate()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate strict-gams baseline manifest")
    parser.add_argument("--sam-file", type=Path, default=None)
    parser.add_argument("--val-par-file", type=Path, default=None)
    parser.add_argument("--calibrated-state", type=Path, default=None)
    parser.add_argument(
        "--results-gdx",
        type=Path,
        default=Path("src/equilibria/templates/reference/pep2/scripts/Results.gdx"),
    )
    parser.add_argument("--gams-slice", choices=["base", "sim1"], default="sim1")
    parser.add_argument("--dynamic-sets", action="store_true")
    parser.add_argument("--dynamic-sam", action="store_true")
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/strict_gams_baseline_manifest.json"),
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=1e-4,
        help="Compatibility relative tolerance used for immediate self-check",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    if not args.results_gdx.exists():
        print(f"Results.gdx not found: {args.results_gdx}")
        return 1

    state = _build_state(args)
    manifest = build_baseline_manifest(
        state=state,
        results_gdx=args.results_gdx,
        gams_slice=args.gams_slice,
        sam_file=args.sam_file,
        val_par_file=args.val_par_file,
        metadata={
            "dynamic_sets": bool(args.dynamic_sets),
            "dynamic_sam": bool(args.dynamic_sam),
        },
    )
    manifest.save_json(args.output)
    print(f"Saved manifest: {args.output}")

    report = evaluate_strict_gams_baseline_compatibility(
        state=state,
        results_gdx=args.results_gdx,
        gams_slice=args.gams_slice,
        manifest_path=args.output,
        sam_file=args.sam_file,
        val_par_file=args.val_par_file,
        rel_tol=args.rel_tol,
        require_manifest=True,
    )
    print(report.summary())
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
