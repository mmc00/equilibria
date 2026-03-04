#!/usr/bin/env python3
"""Run PEP BASE and EXPORT_TAX scenarios and compare each one to GAMS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from equilibria.templates.pep_scenario_parity import (  # noqa: E402
    DEFAULT_RESULTS_GDX,
    DEFAULT_SAM_FILE,
    DEFAULT_VAL_PAR_FILE,
    PEPScenarioParityRunner,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pep2 BASE + EXPORT_TAX and compare against GAMS (base/sim1)."
    )
    parser.add_argument("--sam-file", type=Path, default=DEFAULT_SAM_FILE)
    parser.add_argument("--val-par-file", type=Path, default=DEFAULT_VAL_PAR_FILE)
    parser.add_argument("--results-gdx", type=Path, default=DEFAULT_RESULTS_GDX)
    parser.add_argument("--gdxdump-bin", type=str, default="gdxdump")
    parser.add_argument("--method", choices=["auto", "ipopt"], default="ipopt")
    parser.add_argument("--init-mode", choices=["excel", "gams"], default="excel")
    parser.add_argument("--solve-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--export-tax-multiplier", type=float, default=0.75)
    parser.add_argument(
        "--disable-export-tax-homotopy",
        action="store_true",
        help="Disable staged export-tax continuation (ipopt + excel init).",
    )
    parser.add_argument(
        "--export-tax-homotopy-steps",
        type=int,
        default=5,
        help="Number of continuation steps from 1.0 to export-tax multiplier.",
    )
    parser.add_argument("--compare-abs-tol", type=float, default=1e-6)
    parser.add_argument("--compare-rel-tol", type=float, default=1e-6)
    parser.add_argument("--no-dynamic-sets", action="store_true")
    parser.add_argument("--save-report", type=Path, default=None)
    return parser


def _print_scenario_report(name: str, payload: dict[str, object]) -> None:
    solve = payload["solve"]
    cmp_ = payload["gams_comparison"]

    print("-" * 84)
    print(f"Scenario: {name}")
    print(
        "  solve: converged={converged} iterations={iterations} residual={residual:.3e}".format(
            converged=solve["converged"],
            iterations=solve["iterations"],
            residual=float(solve["final_residual"]),
        )
    )
    print(
        "  compare[{slice_}]: passed={passed} compared={compared} mismatches={mismatches} missing={missing}".format(
            slice_=cmp_["gams_slice"],
            passed=cmp_["passed"],
            compared=cmp_["compared"],
            mismatches=cmp_["mismatches"],
            missing=cmp_["missing"],
        )
    )
    print(
        "  max_abs_diff={abs_:.6e} max_rel_diff={rel_:.6e}".format(
            abs_=float(cmp_["max_abs_diff"]),
            rel_=float(cmp_["max_rel_diff"]),
        )
    )


def main() -> int:
    args = _build_parser().parse_args()

    runner = PEPScenarioParityRunner(
        sam_file=args.sam_file,
        val_par_file=args.val_par_file,
        gams_results_gdx=args.results_gdx,
        gdxdump_bin=args.gdxdump_bin,
        dynamic_sets=(not args.no_dynamic_sets),
        init_mode=args.init_mode,
        method=args.method,
        solve_tolerance=args.solve_tolerance,
        max_iterations=args.max_iterations,
        export_tax_multiplier=args.export_tax_multiplier,
        export_tax_homotopy=(not args.disable_export_tax_homotopy),
        export_tax_homotopy_steps=args.export_tax_homotopy_steps,
        compare_abs_tol=args.compare_abs_tol,
        compare_rel_tol=args.compare_rel_tol,
    )

    report = runner.run()
    print("=" * 84)
    print("PEP BASE + EXPORT_TAX PARITY")
    print("=" * 84)
    _print_scenario_report("base", report["scenarios"]["base"])
    _print_scenario_report("export_tax", report["scenarios"]["export_tax"])
    print("-" * 84)

    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(json.dumps(report, indent=2))
        print(f"Saved report: {args.save_report}")

    base_ok = bool(report["scenarios"]["base"]["gams_comparison"]["passed"])
    export_ok = bool(report["scenarios"]["export_tax"]["gams_comparison"]["passed"])
    return 0 if (base_ok and export_ok) else 2


if __name__ == "__main__":
    raise SystemExit(main())
