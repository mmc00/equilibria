#!/usr/bin/env python3
"""
Run systemic parity gates for PEP model.

This script enforces a fail-fast workflow:
1) Build initialized levels
2) Check equation-contract gates by block
3) Optionally run solver and check gates again
4) Emit JSON trace report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from equilibria.qa.reporting import format_report_summary
from equilibria.qa.sam_checks import run_sam_qa_from_file
from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_calibration_unified_dynamic import (
    PEPModelCalibratorDynamicSAM,
    PEPModelCalibratorExcelDynamicSAM,
)
from equilibria.templates.pep_model_solver import PEPModelSolver
from equilibria.templates.pep_sam_compat import (
    should_apply_cri_pep_fix,
    transform_sam_to_pep_compatible,
)
from equilibria.templates.pep_parity_pipeline import (
    classify_pipeline_outcome,
    default_equation_contracts,
    evaluate_block_gates,
    evaluate_eq29_eq39_against_gams,
    evaluate_eq79_eq84_against_gams,
    evaluate_levels_against_gams,
    evaluate_residual_parity_against_gams,
    evaluate_results_baseline_compatibility,
    summarize_residuals,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _build_state(args: argparse.Namespace):
    sam_path = Path(args.sam_file)
    val_path = Path(args.val_par_file) if args.val_par_file else None

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
        if sam_path.suffix.lower() in {".xlsx", ".xls"}:
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

    calibrator = PEPModelCalibrator(
        sam_file=sam_path,
        val_par_file=val_path,
    )
    return calibrator.calibrate()


def _print_stage(name: str, summary: dict, gates: dict) -> None:
    print("=" * 78)
    print(name)
    print("=" * 78)
    print(f"Residual count: {summary['count']}")
    print(f"RMS: {summary['rms']:.6e}")
    print(f"MAX: {summary['max_abs']:.6e}")
    print(f"Gates passed: {gates['overall_passed']}")
    if gates.get("first_failed_block"):
        print(f"First failed block: {gates['first_failed_block']}")
    print("Top residuals:")
    for eq, val in summary["top_abs"][:10]:
        print(f"  {eq:<24} {val:>14.6e}")


def _json_safe(obj):
    """Convert non-standard scalar types (e.g., numpy) to JSON-safe Python types."""
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def main() -> int:
    parser = argparse.ArgumentParser(description="Systemic parity pipeline for PEP model")
    parser.add_argument("--sam-file", type=Path, required=True)
    parser.add_argument("--val-par-file", type=Path, default=None)
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
        "--init-mode",
        choices=["gams", "excel"],
        default="excel",
    )
    parser.add_argument(
        "--method",
        choices=["none", "auto", "ipopt", "simple_iteration"],
        default="none",
        help="Use 'none' to check initialization only.",
    )
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=120)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--init-gates-mode",
        choices=["absolute", "gams_anchor"],
        default="absolute",
        help="Use absolute residual gates or residual-delta parity gates vs GAMS baseline.",
    )
    parser.add_argument(
        "--gams-anchor-max-abs-tol",
        type=float,
        default=1e-6,
        help="Max absolute residual-delta tolerance for gams_anchor init gates.",
    )
    parser.add_argument(
        "--gams-anchor-rms-tol",
        type=float,
        default=1e-7,
        help="RMS residual-delta tolerance for gams_anchor init gates.",
    )
    parser.add_argument(
        "--gams-results-gdx",
        type=Path,
        default=Path("src/equilibria/templates/reference/pep2/scripts/Results.gdx"),
    )
    parser.add_argument("--gams-results-slice", choices=["base", "sim1"], default="base")
    parser.add_argument("--blockwise-commodity-alpha", type=float, default=0.75)
    parser.add_argument("--blockwise-trade-market-alpha", type=float, default=0.5)
    parser.add_argument("--blockwise-macro-alpha", type=float, default=1.0)
    parser.add_argument(
        "--eq29-eq39-parity",
        action="store_true",
        help="Evaluate EQ29/EQ39/EQ40 residual parity against GAMS Results.gdx.",
    )
    parser.add_argument(
        "--eq29-eq39-tol",
        type=float,
        default=1e-6,
        help="Tolerance for EQ29/EQ39 residual parity delta vs GAMS.",
    )
    parser.add_argument(
        "--gdxdump-bin",
        type=str,
        default="/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump",
        help="Path to gdxdump binary for GAMS-anchored parity checks.",
    )
    parser.add_argument(
        "--eq79-eq84-parity",
        action="store_true",
        help="Evaluate EQ79/EQ84 residual parity against GAMS Results.gdx.",
    )
    parser.add_argument(
        "--eq79-eq84-tol",
        type=float,
        default=1e-6,
        help="Tolerance for EQ79/EQ84 residual parity delta vs GAMS.",
    )
    parser.add_argument(
        "--levels-parity",
        action="store_true",
        help="Compare initialized Python levels against GAMS val* levels.",
    )
    parser.add_argument(
        "--levels-parity-tol",
        type=float,
        default=1e-9,
        help="Tolerance for variable-level parity delta vs GAMS.",
    )
    parser.add_argument(
        "--check-baseline-compatibility",
        action="store_true",
        help="Check if Results.gdx baseline matches calibrated SAM baseline.",
    )
    parser.add_argument(
        "--baseline-compatibility-rel-tol",
        type=float,
        default=1e-4,
        help="Relative tolerance for baseline compatibility check (GDP_BP anchor).",
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
        help="Require baseline manifest for gams initialization",
    )
    parser.add_argument(
        "--disable-strict-gams-baseline-check",
        action="store_true",
        help="Disable gams baseline compatibility gate",
    )
    parser.add_argument(
        "--save-report",
        type=Path,
        default=Path("output/pep_systemic_parity_report.json"),
    )
    parser.add_argument(
        "--sam-qa-mode",
        choices=["hard_fail", "warn", "off"],
        default="hard_fail",
        help="Pre-calibration SAM QA behavior",
    )
    parser.add_argument("--sam-qa-balance-rel-tol", type=float, default=1e-6)
    parser.add_argument("--sam-qa-gdp-rel-tol", type=float, default=0.08)
    parser.add_argument("--sam-qa-max-samples", type=int, default=8)
    parser.add_argument(
        "--cri-fix-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="Apply CRI->PEP SAM compatibility transform before QA/calibration",
    )
    parser.add_argument("--cri-fix-output", type=Path, default=None)
    parser.add_argument("--cri-fix-report", type=Path, default=None)
    parser.add_argument(
        "--cri-fix-target-mode",
        choices=["geomean", "average", "original"],
        default="geomean",
    )
    parser.add_argument("--cri-fix-margin-commodity", type=str, default="ser")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    sam_file_original = Path(args.sam_file)
    cri_fix_report = None
    cri_fix_report_path: Path | None = None
    if should_apply_cri_pep_fix(sam_file_original, mode=args.cri_fix_mode):
        output_sam = (
            args.cri_fix_output
            if args.cri_fix_output
            else Path("output") / f"{sam_file_original.stem}-pep-compatible{sam_file_original.suffix}"
        )
        report_json = (
            args.cri_fix_report
            if args.cri_fix_report
            else Path("output") / f"{sam_file_original.stem}-pep-compatible-report.json"
        )
        cri_fix_report_path = report_json
        cri_fix_report = transform_sam_to_pep_compatible(
            input_sam=sam_file_original,
            output_sam=output_sam,
            report_json=report_json,
            target_mode=args.cri_fix_target_mode,
            margin_commodity=args.cri_fix_margin_commodity,
        )
        args.sam_file = output_sam
        print("Applied CRI->PEP SAM compatibility transform")
        print(f"  input : {sam_file_original}")
        print(f"  output: {args.sam_file}")
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
    sam_qa_report = None
    if args.sam_qa_mode != "off":
        sam_qa_report = run_sam_qa_from_file(
            sam_file=args.sam_file,
            dynamic_sam=args.dynamic_sam,
            accounts=accounts,
            balance_rel_tol=args.sam_qa_balance_rel_tol,
            gdp_rel_tol=args.sam_qa_gdp_rel_tol,
            max_samples=args.sam_qa_max_samples,
        )
        print(format_report_summary(sam_qa_report))
        if not sam_qa_report.passed:
            if args.sam_qa_mode == "hard_fail":
                classification = classify_pipeline_outcome(
                    sam_qa_report=sam_qa_report,
                    init_gates=None,
                    solve_report=None,
                    method=args.method,
                )
                report = {
                    "config": {
                        "sam_file": str(args.sam_file),
                        "sam_file_original": str(sam_file_original),
                        "cri_fix_mode": args.cri_fix_mode,
                        "cri_fix_applied": cri_fix_report is not None,
                        "cri_fix_report": str(cri_fix_report_path) if cri_fix_report_path else None,
                        "val_par_file": str(args.val_par_file) if args.val_par_file else None,
                        "dynamic_sam": args.dynamic_sam,
                        "init_mode": args.init_mode,
                        "method": args.method,
                        "tolerance": args.tolerance,
                        "max_iterations": args.max_iterations,
                        "init_gates_mode": args.init_gates_mode,
                        "gams_anchor_max_abs_tol": args.gams_anchor_max_abs_tol,
                        "gams_anchor_rms_tol": args.gams_anchor_rms_tol,
                        "fail_fast": args.fail_fast,
                        "gams_results_gdx": str(args.gams_results_gdx),
                        "gams_results_slice": args.gams_results_slice,
                        "blockwise_commodity_alpha": args.blockwise_commodity_alpha,
                        "blockwise_trade_market_alpha": args.blockwise_trade_market_alpha,
                        "blockwise_macro_alpha": args.blockwise_macro_alpha,
                        "baseline_manifest": str(args.baseline_manifest) if args.baseline_manifest else None,
                    },
                    "sam_qa": sam_qa_report.to_dict(),
                    "contracts": [],
                    "init": None,
                    "solve": None,
                    "classification": classification,
                }
                args.save_report.parent.mkdir(parents=True, exist_ok=True)
                args.save_report.write_text(json.dumps(report, indent=2, default=_json_safe))
                print("SAM QA failed in hard_fail mode. Aborting before calibration.")
                print(
                    f"Outcome classification: {classification['kind']} ({classification['reason']})"
                )
                print(f"Saved report: {args.save_report}")
                return 2
            print("⚠ SAM QA failed, continuing because mode=warn.")

    state = _build_state(args)

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
        baseline_compatibility_rel_tol=args.baseline_compatibility_rel_tol,
        enforce_strict_gams_baseline=(not args.disable_strict_gams_baseline_check),
        sam_file=args.sam_file,
        val_par_file=args.val_par_file,
        gdxdump_bin=args.gdxdump_bin,
    )

    try:
        vars_init = solver._create_initial_guess()
    except RuntimeError as exc:
        classification = classify_pipeline_outcome(
            sam_qa_report=sam_qa_report,
            init_gates=None,
            solve_report=None,
            method=args.method,
            error=str(exc),
        )
        report = {
            "config": {
                "sam_file": str(args.sam_file),
                "sam_file_original": str(sam_file_original),
                "cri_fix_mode": args.cri_fix_mode,
                "cri_fix_applied": cri_fix_report is not None,
                "cri_fix_report": str(cri_fix_report_path) if cri_fix_report_path else None,
                "val_par_file": str(args.val_par_file) if args.val_par_file else None,
                "dynamic_sam": args.dynamic_sam,
                "init_mode": args.init_mode,
                "method": args.method,
                "tolerance": args.tolerance,
                "max_iterations": args.max_iterations,
                "init_gates_mode": args.init_gates_mode,
                "gams_anchor_max_abs_tol": args.gams_anchor_max_abs_tol,
                "gams_anchor_rms_tol": args.gams_anchor_rms_tol,
                "fail_fast": args.fail_fast,
                "gams_results_gdx": str(args.gams_results_gdx),
                "gams_results_slice": args.gams_results_slice,
                "blockwise_commodity_alpha": args.blockwise_commodity_alpha,
                "blockwise_trade_market_alpha": args.blockwise_trade_market_alpha,
                "blockwise_macro_alpha": args.blockwise_macro_alpha,
                "baseline_manifest": str(args.baseline_manifest) if args.baseline_manifest else None,
            },
            "sam_qa": sam_qa_report.to_dict() if sam_qa_report else None,
            "contracts": [],
            "init": None,
            "solve": None,
            "error": str(exc),
            "classification": classification,
        }
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(json.dumps(report, indent=2, default=_json_safe))
        print(f"Initialization failed: {exc}")
        print(f"Outcome classification: {classification['kind']} ({classification['reason']})")
        print(f"Saved report: {args.save_report}")
        return 2
    init_residuals = solver.equations.calculate_all_residuals(vars_init)
    init_summary = asdict(summarize_residuals(init_residuals))
    contracts = [asdict(c) for c in default_equation_contracts()]
    init_gates_absolute = evaluate_block_gates(
        residuals=init_residuals,
        fail_fast=args.fail_fast,
    )
    gams_residual_parity = None
    init_gates = init_gates_absolute
    if args.init_gates_mode == "gams_anchor":
        gams_residual_parity = evaluate_residual_parity_against_gams(
            vars_obj=vars_init,
            equations=solver.equations,
            results_gdx=args.gams_results_gdx,
            gdxdump_bin=args.gdxdump_bin,
            gams_slice=args.gams_results_slice,
            max_abs_tol=args.gams_anchor_max_abs_tol,
            rms_tol=args.gams_anchor_rms_tol,
        )
        init_gates = {
            "overall_passed": bool(gams_residual_parity["passed"]),
            "fail_fast": args.fail_fast,
            "mode": "gams_anchor",
            "first_failed_block": gams_residual_parity.get("first_failed_block"),
            "blocks": [
                {
                    "block": str(b["block"]),
                    "passed": bool(b["passed"]),
                    "max_abs": float(b["max_abs_delta"]),
                    "rms": float(b["rms_delta"]),
                    "max_abs_gate": float(b["max_abs_tol"]),
                    "rms_gate": float(b["rms_tol"]),
                    "top_abs": list(b["top_abs_delta"]),
                }
                for b in gams_residual_parity.get("blocks", [])
            ],
        }

    baseline_compatibility = None
    if args.check_baseline_compatibility:
        baseline_compatibility = evaluate_results_baseline_compatibility(
            state=state,
            results_gdx=args.gams_results_gdx,
            gdxdump_bin=args.gdxdump_bin,
            gams_slice=args.gams_results_slice,
            rel_tol=args.baseline_compatibility_rel_tol,
        )
        print("Baseline compatibility (calibration vs Results.gdx):")
        print(f"  passed: {baseline_compatibility['passed']}")
        print(f"  Python GDP_BP: {baseline_compatibility['python_gdp_bp']:.6f}")
        print(f"  GAMS   GDP_BP: {baseline_compatibility['gams_gdp_bp']:.6f}")
        print(f"  rel delta: {baseline_compatibility['rel_delta']:.6e}")
        if not baseline_compatibility["passed"]:
            init_gates["overall_passed"] = False
            if not init_gates.get("first_failed_block"):
                init_gates["first_failed_block"] = "baseline_reference_consistency"

    eq29_eq39_parity = None
    if args.eq29_eq39_parity:
        eq29_eq39_parity = evaluate_eq29_eq39_against_gams(
            vars_obj=vars_init,
            results_gdx=args.gams_results_gdx,
            gdxdump_bin=args.gdxdump_bin,
            gams_slice=args.gams_results_slice,
            tol=args.eq29_eq39_tol,
        )
        if not eq29_eq39_parity["passed"]:
            # Anchor production-tax block to GAMS equation residual parity.
            init_gates["overall_passed"] = False
            if not init_gates.get("first_failed_block"):
                init_gates["first_failed_block"] = "production_tax_consistency"
            init_gates["blocks"].append(
                {
                    "block": "production_tax_consistency_gams_anchor",
                    "passed": False,
                    "max_abs": float(eq29_eq39_parity["max_abs_delta_overall"]),
                    "rms": 0.0,
                    "max_abs_gate": float(args.eq29_eq39_tol),
                    "rms_gate": 0.0,
                    "top_abs": [
                        ("EQ29_delta", float(eq29_eq39_parity["eq29"]["delta"])),
                        (
                            "EQ39_max_delta",
                            float(eq29_eq39_parity["eq39"]["max_abs_delta"]),
                        ),
                        (
                            "EQ40_max_delta",
                            float(eq29_eq39_parity["eq40"]["max_abs_delta"]),
                        ),
                    ],
                }
            )

    _print_stage("INIT STAGE", init_summary, init_gates)
    if gams_residual_parity is not None:
        print("GAMS residual-delta parity (anchor mode):")
        print(f"  passed: {gams_residual_parity['passed']}")
        print(f"  max delta: {gams_residual_parity['max_abs_delta']:.6e}")
        print(f"  rms delta: {gams_residual_parity['rms_delta']:.6e}")
    if eq29_eq39_parity is not None:
        print("EQ29/EQ39/EQ40 GAMS parity:")
        print(f"  passed: {eq29_eq39_parity['passed']}")
        print(f"  EQ29 delta: {eq29_eq39_parity['eq29']['delta']:.6e}")
        print(f"  EQ39 max delta: {eq29_eq39_parity['eq39']['max_abs_delta']:.6e}")
        print(f"  EQ40 max delta: {eq29_eq39_parity['eq40']['max_abs_delta']:.6e}")

    eq79_eq84_parity = None
    if args.eq79_eq84_parity:
        eq79_eq84_parity = evaluate_eq79_eq84_against_gams(
            vars_obj=vars_init,
            results_gdx=args.gams_results_gdx,
            gdxdump_bin=args.gdxdump_bin,
            gams_slice=args.gams_results_slice,
            tol=args.eq79_eq84_tol,
        )
        if not eq79_eq84_parity["passed"]:
            init_gates["overall_passed"] = False
            if not init_gates.get("first_failed_block"):
                init_gates["first_failed_block"] = "trade_price_index_consistency"
        print("EQ79/EQ84 GAMS parity:")
        print(f"  passed: {eq79_eq84_parity['passed']}")
        print(f"  EQ79 max delta: {eq79_eq84_parity['eq79']['max_abs_delta']:.6e}")
        print(f"  EQ84 max delta: {eq79_eq84_parity['eq84']['max_abs_delta']:.6e}")

    levels_parity = None
    if args.levels_parity:
        levels_parity = evaluate_levels_against_gams(
            vars_obj=vars_init,
            results_gdx=args.gams_results_gdx,
            gdxdump_bin=args.gdxdump_bin,
            gams_slice=args.gams_results_slice,
            tol=args.levels_parity_tol,
        )
        print("Levels parity (Python init vs GAMS val*):")
        print(f"  passed: {levels_parity['passed']}")
        print(f"  compared: {levels_parity['count_compared']}")
        print(f"  missing in python: {levels_parity['count_missing_in_python']}")
        print(f"  max delta: {levels_parity['max_abs_delta']:.6e}")
        print(f"  rms delta: {levels_parity['rms_delta']:.6e}")

    report = {
        "config": {
            "sam_file": str(args.sam_file),
            "sam_file_original": str(sam_file_original),
            "cri_fix_mode": args.cri_fix_mode,
            "cri_fix_applied": cri_fix_report is not None,
            "cri_fix_report": str(cri_fix_report_path) if cri_fix_report_path else None,
            "val_par_file": str(args.val_par_file) if args.val_par_file else None,
            "dynamic_sam": args.dynamic_sam,
            "init_mode": args.init_mode,
            "method": args.method,
            "tolerance": args.tolerance,
            "max_iterations": args.max_iterations,
            "init_gates_mode": args.init_gates_mode,
            "gams_anchor_max_abs_tol": args.gams_anchor_max_abs_tol,
            "gams_anchor_rms_tol": args.gams_anchor_rms_tol,
            "fail_fast": args.fail_fast,
            "gams_results_gdx": str(args.gams_results_gdx),
            "gams_results_slice": args.gams_results_slice,
            "blockwise_commodity_alpha": args.blockwise_commodity_alpha,
            "blockwise_trade_market_alpha": args.blockwise_trade_market_alpha,
            "blockwise_macro_alpha": args.blockwise_macro_alpha,
            "baseline_manifest": str(args.baseline_manifest) if args.baseline_manifest else None,
        },
        "sam_qa": sam_qa_report.to_dict() if sam_qa_report else None,
        "contracts": contracts,
        "init": {
            "summary": init_summary,
            "gates": init_gates,
            "gates_absolute": init_gates_absolute,
            "gates_mode": args.init_gates_mode,
            "gams_residual_parity": gams_residual_parity,
            "baseline_compatibility": baseline_compatibility,
            "eq29_eq39_gams_parity": eq29_eq39_parity,
            "eq79_eq84_gams_parity": eq79_eq84_parity,
            "levels_gams_parity": levels_parity,
        },
        "solve": None,
    }

    if args.method != "none":
        if args.fail_fast and not init_gates["overall_passed"]:
            print("Fail-fast enabled: skipping solve stage because init gates failed.")
        else:
            solution = solver.solve(method=args.method)
            try:
                solve_residuals = solver.equations.calculate_all_residuals(solution.variables)
                solve_summary = asdict(summarize_residuals(solve_residuals))
                solve_gates = evaluate_block_gates(
                    residuals=solve_residuals,
                    fail_fast=args.fail_fast,
                )
                _print_stage("SOLVE STAGE", solve_summary, solve_gates)
                report["solve"] = {
                    "converged": solution.converged,
                    "iterations": solution.iterations,
                    "final_residual": solution.final_residual,
                    "message": solution.message,
                    "summary": solve_summary,
                    "gates": solve_gates,
                }
            except Exception as exc:
                report["solve"] = {
                    "converged": False,
                    "iterations": solution.iterations,
                    "final_residual": solution.final_residual,
                    "message": f"{solution.message} | residual_evaluation_error: {exc}",
                    "summary": None,
                    "gates": None,
                }
                print(f"SOLVE STAGE evaluation error: {exc}")

    report["classification"] = classify_pipeline_outcome(
        sam_qa_report=sam_qa_report,
        init_gates=init_gates,
        solve_report=report.get("solve"),
        method=args.method,
    )

    args.save_report.parent.mkdir(parents=True, exist_ok=True)
    args.save_report.write_text(json.dumps(report, indent=2, default=_json_safe))
    classification = report["classification"]
    print(f"Outcome classification: {classification['kind']} ({classification['reason']})")
    print(f"Saved report: {args.save_report}")

    # Exit non-zero if init gates fail, or if solve stage fails when requested.
    exit_ok = bool(init_gates["overall_passed"])
    if args.method != "none":
        solve_report = report.get("solve")
        if solve_report is None:
            # solve was skipped (fail_fast). Treat as failure when a solve was requested.
            exit_ok = False
        else:
            solve_gates = solve_report.get("gates")
            solve_gates_ok = bool(solve_gates and solve_gates.get("overall_passed", False))
            solve_converged = bool(solve_report.get("converged", False))
            exit_ok = exit_ok and solve_converged and solve_gates_ok

    return 0 if exit_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
