#!/usr/bin/env python3
"""Benchmark analytic vs numeric Jacobian modes on the public PEP scenario pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from equilibria.baseline import GAMSNLPReferenceManifest  # noqa: E402
from equilibria.simulations import PepSimulator  # noqa: E402
from equilibria.simulations import export_tax, government_spending, import_price, import_shock  # noqa: E402

DEFAULT_SAM_FILE = REPO_ROOT / "src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx"
DEFAULT_VAL_PAR_FILE = REPO_ROOT / "src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure analytic vs numeric Jacobian impact on the public PEP core scenario pack "
            "(base + export_tax + import_price_agr + import_shock + government_spending)."
        )
    )
    parser.add_argument("--sam-file", type=Path, default=DEFAULT_SAM_FILE)
    parser.add_argument("--val-par-file", type=Path, default=DEFAULT_VAL_PAR_FILE)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--gdxdump-bin", type=str, default="gdxdump")
    parser.add_argument("--method", choices=["auto", "ipopt"], default="ipopt")
    parser.add_argument("--init-mode", choices=["excel", "gams"], default="excel")
    parser.add_argument("--solve-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--compare-abs-tol", type=float, default=1e-6)
    parser.add_argument("--compare-rel-tol", type=float, default=1e-6)
    parser.add_argument("--no-dynamic-sets", action="store_true")
    parser.add_argument("--sam-qa-mode", choices=["off", "warn", "hard_fail"], default="off")
    parser.add_argument("--cri-fix-mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--import-price-commodity", type=str, default="agr")
    parser.add_argument("--import-price-multiplier", type=float, default=1.25)
    parser.add_argument("--import-shock-multiplier", type=float, default=1.25)
    parser.add_argument("--export-tax-multiplier", type=float, default=0.75)
    parser.add_argument("--government-spending-multiplier", type=float, default=1.2)
    parser.add_argument(
        "--gate",
        action="store_true",
        help=(
            "Fail if analytic mode uses finite differences, if either mode stops converging, "
            "or if analytic parity is worse than numeric parity."
        ),
    )
    parser.add_argument(
        "--max-analytic-fd-evals",
        type=int,
        default=0,
        help="Maximum allowed finite-difference evaluations in analytic mode when --gate is enabled.",
    )
    parser.add_argument("--save-report", type=Path, default=None)
    return parser


def _load_reference_manifest(path: Path) -> dict[str, dict[str, str]]:
    raw = json.loads(path.read_text())
    manifest = GAMSNLPReferenceManifest.model_validate(raw)
    if manifest.scenario_references is None:
        raise ValueError("reference manifest is missing scenario_references")
    out: dict[str, dict[str, str]] = {}
    for name, reference in manifest.scenario_references.items():
        out[str(name).strip().lower()] = {
            "results_gdx": reference.results_gdx.path,
            "slice": reference.slice,
        }
    return out


def _scenario_pack(args: argparse.Namespace) -> list[tuple[str, Any | None]]:
    return [
        ("base", None),
        ("export_tax", export_tax(multiplier=args.export_tax_multiplier)),
        (
            f"import_price_{args.import_price_commodity.strip().lower()}",
            import_price(
                commodity=args.import_price_commodity,
                multiplier=args.import_price_multiplier,
            ),
        ),
        ("import_shock", import_shock(multiplier=args.import_shock_multiplier)),
        (
            "government_spending",
            government_spending(multiplier=args.government_spending_multiplier),
        ),
    ]


def _run_one(
    simulator: PepSimulator,
    *,
    scenario_name: str,
    scenario: Any | None,
    reference_manifest: dict[str, dict[str, str]],
    compare_abs_tol: float,
    compare_rel_tol: float,
) -> dict[str, Any]:
    ref = reference_manifest[scenario_name]
    if scenario is None:
        raw = simulator.run_scenarios(
            scenarios=[],
            include_base=True,
            warm_start=False,
            reference_results_gdx=ref["results_gdx"],
            base_reference_slice=ref["slice"],
            compare_abs_tol=compare_abs_tol,
            compare_rel_tol=compare_rel_tol,
        )
        entry = raw["base"]
    else:
        raw = simulator.run_scenarios(
            scenarios=[scenario],
            include_base=False,
            warm_start=False,
            reference_results_gdx=ref["results_gdx"],
            compare_abs_tol=compare_abs_tol,
            compare_rel_tol=compare_rel_tol,
        )
        entry = raw["scenarios"][0]
    entry["reference_slice"] = ref["slice"]
    if isinstance(entry.get("comparison"), dict):
        entry["comparison"]["gams_slice"] = ref["slice"]
    return entry


def _mode_summary(entry: dict[str, Any]) -> dict[str, Any]:
    solve = dict(entry["solve"])
    stats = dict(solve.get("solver_stats") or {})
    comparison = entry.get("comparison")
    return {
        "converged": bool(solve["converged"]),
        "iterations": int(solve["iterations"]),
        "final_residual": float(solve["final_residual"]),
        "message": str(solve["message"]),
        "solver_stats": stats,
        "comparison": comparison,
    }


def _safe_ratio(numerator: float | int, denominator: float | int) -> float | None:
    denom = float(denominator)
    if denom == 0.0:
        return None
    return float(numerator) / denom


def _compare_modes(
    analytic: dict[str, dict[str, Any]],
    numeric: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for scenario_name in analytic:
        analytic_entry = analytic[scenario_name]
        numeric_entry = numeric[scenario_name]
        analytic_stats = dict(analytic_entry.get("solver_stats") or {})
        numeric_stats = dict(numeric_entry.get("solver_stats") or {})
        out[scenario_name] = {
            "analytic": analytic_entry,
            "numeric": numeric_entry,
            "deltas": {
                "iteration_delta": int(analytic_entry["iterations"]) - int(numeric_entry["iterations"]),
                "final_residual_delta": float(analytic_entry["final_residual"])
                - float(numeric_entry["final_residual"]),
                "wall_time_delta_seconds": float(analytic_stats.get("wall_time_seconds", 0.0))
                - float(numeric_stats.get("wall_time_seconds", 0.0)),
                "analytic_speedup_vs_numeric": _safe_ratio(
                    float(numeric_stats.get("wall_time_seconds", 0.0)),
                    float(analytic_stats.get("wall_time_seconds", 0.0)),
                ),
                "constraint_eval_delta": int(analytic_stats.get("constraint_eval_count", 0))
                - int(numeric_stats.get("constraint_eval_count", 0)),
                "jacobian_eval_delta": int(analytic_stats.get("jacobian_eval_count", 0))
                - int(numeric_stats.get("jacobian_eval_count", 0)),
                "finite_difference_eval_delta": int(analytic_stats.get("finite_difference_eval_count", 0))
                - int(numeric_stats.get("finite_difference_eval_count", 0)),
            },
        }
    return out


def _evaluate_gate(
    report: dict[str, Any],
    *,
    max_analytic_fd_evals: int,
) -> dict[str, Any]:
    failures: list[str] = []
    for scenario_name, payload in report["mode_comparison"].items():
        analytic = payload["analytic"]
        numeric = payload["numeric"]
        analytic_stats = dict(analytic.get("solver_stats") or {})
        numeric_stats = dict(numeric.get("solver_stats") or {})

        if not bool(analytic["converged"]):
            failures.append(f"{scenario_name}: analytic solve did not converge")
        if not bool(numeric["converged"]):
            failures.append(f"{scenario_name}: numeric solve did not converge")

        analytic_fd = int(analytic_stats.get("finite_difference_eval_count", 0))
        if analytic_fd > int(max_analytic_fd_evals):
            failures.append(
                f"{scenario_name}: analytic finite_difference_eval_count={analytic_fd} exceeds {max_analytic_fd_evals}"
            )

        analytic_cmp = analytic.get("comparison")
        numeric_cmp = numeric.get("comparison")
        if isinstance(analytic_cmp, dict) and isinstance(numeric_cmp, dict):
            if bool(analytic_cmp["passed"]) is False and bool(numeric_cmp["passed"]) is True:
                failures.append(
                    f"{scenario_name}: analytic parity failed while numeric parity passed"
                )
            if int(analytic_cmp["mismatches"]) > int(numeric_cmp["mismatches"]):
                failures.append(
                    f"{scenario_name}: analytic mismatches={analytic_cmp['mismatches']} worse than numeric={numeric_cmp['mismatches']}"
                )
            if int(analytic_cmp["missing"]) > int(numeric_cmp["missing"]):
                failures.append(
                    f"{scenario_name}: analytic missing={analytic_cmp['missing']} worse than numeric={numeric_cmp['missing']}"
                )

    return {
        "passed": not failures,
        "max_analytic_fd_evals": int(max_analytic_fd_evals),
        "failures": failures,
    }


def _print_summary(report: dict[str, Any]) -> None:
    print("=" * 92)
    print("PEP JACOBIAN MODE BENCHMARK")
    print("=" * 92)
    for scenario_name, payload in report["mode_comparison"].items():
        analytic = payload["analytic"]
        numeric = payload["numeric"]
        deltas = payload["deltas"]
        a_stats = analytic.get("solver_stats") or {}
        n_stats = numeric.get("solver_stats") or {}
        print(f"Scenario: {scenario_name}")
        print(
            "  analytic: converged={conv} iter={it} residual={res:.3e} wall={wall:.3f}s fd={fd}".format(
                conv=analytic["converged"],
                it=analytic["iterations"],
                res=float(analytic["final_residual"]),
                wall=float(a_stats.get("wall_time_seconds", 0.0)),
                fd=int(a_stats.get("finite_difference_eval_count", 0)),
            )
        )
        print(
            "  numeric : converged={conv} iter={it} residual={res:.3e} wall={wall:.3f}s fd={fd}".format(
                conv=numeric["converged"],
                it=numeric["iterations"],
                res=float(numeric["final_residual"]),
                wall=float(n_stats.get("wall_time_seconds", 0.0)),
                fd=int(n_stats.get("finite_difference_eval_count", 0)),
            )
        )
        speedup = deltas["analytic_speedup_vs_numeric"]
        speedup_txt = f"{speedup:.2f}x" if speedup is not None else "n/a"
        print(
            "  delta   : iter={it_delta:+d} wall={wall_delta:+.3f}s analytic_speedup_vs_numeric={speedup}".format(
                it_delta=int(deltas["iteration_delta"]),
                wall_delta=float(deltas["wall_time_delta_seconds"]),
                speedup=speedup_txt,
            )
        )
        analytic_cmp = analytic.get("comparison")
        numeric_cmp = numeric.get("comparison")
        if isinstance(analytic_cmp, dict) and isinstance(numeric_cmp, dict):
            print(
                "  parity  : analytic passed={a_pass} mismatches={a_mm} | numeric passed={n_pass} mismatches={n_mm}".format(
                    a_pass=analytic_cmp["passed"],
                    a_mm=analytic_cmp["mismatches"],
                    n_pass=numeric_cmp["passed"],
                    n_mm=numeric_cmp["mismatches"],
                )
            )
        print("-" * 92)


def main() -> int:
    args = _build_parser().parse_args()
    reference_manifest = _load_reference_manifest(args.reference_manifest)
    scenario_pack = _scenario_pack(args)

    per_mode: dict[str, dict[str, Any]] = {}
    for jacobian_mode in ("analytic", "numeric"):
        simulator = PepSimulator(
            sam_file=args.sam_file,
            val_par_file=args.val_par_file,
            gdxdump_bin=args.gdxdump_bin,
            init_mode=args.init_mode,
            method=args.method,
            solve_tolerance=args.solve_tolerance,
            max_iterations=args.max_iterations,
            dynamic_sets=(not args.no_dynamic_sets),
            sam_qa_mode=args.sam_qa_mode,
            cri_fix_mode=args.cri_fix_mode,
            config={"jacobian_mode": jacobian_mode},
        ).fit()

        mode_entries: dict[str, Any] = {}
        for scenario_name, scenario in scenario_pack:
            mode_entries[scenario_name] = _mode_summary(
                _run_one(
                    simulator,
                    scenario_name=scenario_name,
                    scenario=scenario,
                    reference_manifest=reference_manifest,
                    compare_abs_tol=args.compare_abs_tol,
                    compare_rel_tol=args.compare_rel_tol,
                )
            )
        per_mode[jacobian_mode] = mode_entries

    report = {
        "config": {
            "sam_file": str(args.sam_file),
            "val_par_file": str(args.val_par_file) if args.val_par_file is not None else None,
            "reference_manifest": str(args.reference_manifest),
            "gdxdump_bin": args.gdxdump_bin,
            "method": args.method,
            "init_mode": args.init_mode,
            "solve_tolerance": float(args.solve_tolerance),
            "max_iterations": int(args.max_iterations),
            "compare_abs_tol": float(args.compare_abs_tol),
            "compare_rel_tol": float(args.compare_rel_tol),
            "dynamic_sets": bool(not args.no_dynamic_sets),
            "sam_qa_mode": args.sam_qa_mode,
            "cri_fix_mode": args.cri_fix_mode,
            "gate": bool(args.gate),
            "max_analytic_fd_evals": int(args.max_analytic_fd_evals),
        },
        "analytic": per_mode["analytic"],
        "numeric": per_mode["numeric"],
        "mode_comparison": _compare_modes(per_mode["analytic"], per_mode["numeric"]),
    }
    report["gate"] = _evaluate_gate(
        report,
        max_analytic_fd_evals=args.max_analytic_fd_evals,
    )

    _print_summary(report)
    if args.gate and not bool(report["gate"]["passed"]):
        print("GATE FAILURES:")
        for failure in report["gate"]["failures"]:
            print(f"  - {failure}")
    if args.save_report is not None:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(json.dumps(report, indent=2, sort_keys=True))
    if args.gate and not bool(report["gate"]["passed"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
