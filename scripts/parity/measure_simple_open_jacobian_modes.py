#!/usr/bin/env python
"""Benchmark analytic vs numeric Jacobian modes for the simple-open template."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from equilibria.solver import (  # noqa: E402
    compare_jacobian_modes,
    evaluate_jacobian_mode_gate,
    solver_stats_payload,
    summarize_jacobian_mode_entry,
)
from equilibria.templates import (  # noqa: E402
    SimpleOpenEconomy,
    build_simple_open_contract,
    build_simple_open_runtime_config,
)
from equilibria.templates.simple_open_constraint_jacobian import (  # noqa: E402
    SimpleOpenConstraintJacobianHarness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--closures",
        nargs="+",
        default=["simple_open_default", "flexible_external_balance"],
        help="SimpleOpen closure names to benchmark.",
    )
    parser.add_argument(
        "--comparison-tol",
        type=float,
        default=1e-6,
        help="Absolute tolerance used to compare Jacobian values against the analytic reference.",
    )
    parser.add_argument(
        "--max-analytic-fd-evals",
        type=int,
        default=0,
        help="Maximum finite-difference evaluations allowed for analytic mode when --gate is enabled.",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Fail non-zero if the structural analytic-vs-numeric gate does not pass.",
    )
    parser.add_argument(
        "--save-report",
        type=Path,
        default=None,
        help="Optional path to save the JSON report.",
    )
    return parser.parse_args()


def _build_entry(
    *,
    closure_name: str,
    jacobian_mode: str,
    comparison_tol: float,
) -> dict[str, Any]:
    contract = build_simple_open_contract({"closure": {"name": closure_name}})
    runtime_config = build_simple_open_runtime_config({"jacobian_mode": jacobian_mode})
    template = SimpleOpenEconomy(contract=contract, runtime_config=runtime_config)
    harness = SimpleOpenConstraintJacobianHarness(
        contract=template.contract,
        jacobian_mode=template.runtime_config.jacobian_mode,
    )
    reference_harness = SimpleOpenConstraintJacobianHarness(
        contract=template.contract,
        jacobian_mode="analytic",
    )

    x0 = harness.benchmark_point
    started_at = time.perf_counter()
    residuals = harness.evaluate_constraints(x0)
    rows, cols = harness.jacobian_structure()
    values = harness.evaluate_jacobian_values(x0)
    elapsed = float(time.perf_counter() - started_at)

    ref_residuals = reference_harness.evaluate_constraints(reference_harness.benchmark_point)
    ref_rows, ref_cols = reference_harness.jacobian_structure()
    ref_values = reference_harness.evaluate_jacobian_values(reference_harness.benchmark_point)

    compared = int(min(len(values), len(ref_values)))
    structure_match = bool(
        np.array_equal(rows, ref_rows) and np.array_equal(cols, ref_cols)
    )
    value_diffs = np.abs(values[:compared] - ref_values[:compared]) if compared else np.array([], dtype=float)
    residual_diffs = np.abs(residuals - ref_residuals)
    mismatches = int(np.count_nonzero(value_diffs > float(comparison_tol)))
    missing = int(abs(len(values) - len(ref_values)))
    passed = bool(
        structure_match
        and missing == 0
        and mismatches == 0
        and float(np.max(residual_diffs, initial=0.0)) <= float(comparison_tol)
    )

    solve = {
        "converged": bool(float(np.max(np.abs(residuals), initial=0.0)) <= 1e-12),
        "iterations": 0,
        "final_residual": float(np.max(np.abs(residuals), initial=0.0)),
        "message": f"{closure_name}-{jacobian_mode}-benchmark",
        "solver_stats": solver_stats_payload(
            jacobian_stats=harness.stats(),
            wall_time_seconds=elapsed,
            objective_eval_count=0,
        ),
    }
    comparison = {
        "reference_mode": "analytic",
        "passed": passed,
        "compared": compared,
        "mismatches": mismatches,
        "missing": missing,
        "max_abs_diff": float(np.max(value_diffs, initial=0.0)),
        "max_rel_diff": float(
            np.max(
                np.divide(
                    value_diffs,
                    np.maximum(np.abs(ref_values[:compared]), 1.0),
                    out=np.zeros_like(value_diffs),
                    where=np.maximum(np.abs(ref_values[:compared]), 1.0) > 0.0,
                ),
                initial=0.0,
            )
        )
        if compared
        else 0.0,
        "structure_match": structure_match,
        "constraint_max_abs_diff": float(np.max(residual_diffs, initial=0.0)),
    }
    return {
        "name": closure_name,
        "template": template.get_info(),
        "solve": solve,
        "comparison": comparison,
    }


def _mode_summary(entry: dict[str, Any]) -> dict[str, Any]:
    return summarize_jacobian_mode_entry(entry).to_dict()


def _print_summary(report: dict[str, Any]) -> None:
    print("=" * 92)
    print("SIMPLE OPEN JACOBIAN MODE BENCHMARK")
    print("=" * 92)
    for closure_name, payload in report["mode_comparison"].items():
        analytic = payload["analytic"]
        numeric = payload["numeric"]
        deltas = payload["deltas"]
        analytic_stats = analytic["solver_stats"]
        numeric_stats = numeric["solver_stats"]
        print(f"Closure: {closure_name}")
        print(
            "  analytic:"
            f" converged={analytic['converged']}"
            f" residual={analytic['final_residual']:.3e}"
            f" wall={analytic_stats.get('wall_time_seconds', 0.0):.3f}s"
            f" fd={analytic_stats.get('finite_difference_eval_count', 0)}"
        )
        print(
            "  numeric :"
            f" converged={numeric['converged']}"
            f" residual={numeric['final_residual']:.3e}"
            f" wall={numeric_stats.get('wall_time_seconds', 0.0):.3f}s"
            f" fd={numeric_stats.get('finite_difference_eval_count', 0)}"
        )
        speedup = deltas["analytic_speedup_vs_numeric"]
        speedup_text = "n/a" if speedup is None else f"{speedup:.2f}x"
        print(
            "  delta   :"
            f" wall={deltas['wall_time_delta_seconds']:+.3f}s"
            f" analytic_speedup_vs_numeric={speedup_text}"
        )
        print(
            "  compare :"
            f" analytic passed={analytic['comparison']['passed']} mismatches={analytic['comparison']['mismatches']}"
            f" | numeric passed={numeric['comparison']['passed']} mismatches={numeric['comparison']['mismatches']}"
        )
        print("-" * 92)


def main() -> int:
    args = parse_args()
    closures = tuple(dict.fromkeys(str(item).strip() for item in args.closures if str(item).strip()))
    per_mode: dict[str, dict[str, dict[str, Any]]] = {"analytic": {}, "numeric": {}}

    for jacobian_mode in ("analytic", "numeric"):
        for closure_name in closures:
            entry = _build_entry(
                closure_name=closure_name,
                jacobian_mode=jacobian_mode,
                comparison_tol=float(args.comparison_tol),
            )
            per_mode[jacobian_mode][closure_name] = _mode_summary(entry)

    report = {
        "metadata": {
            "model": "simple_open",
            "gate": bool(args.gate),
            "closures": list(closures),
            "comparison_tol": float(args.comparison_tol),
        },
        "analytic": per_mode["analytic"],
        "numeric": per_mode["numeric"],
        "mode_comparison": compare_jacobian_modes(per_mode["analytic"], per_mode["numeric"]),
    }
    report["gate"] = evaluate_jacobian_mode_gate(
        report["mode_comparison"],
        max_analytic_fd_evals=int(args.max_analytic_fd_evals),
    )

    _print_summary(report)

    if args.save_report is not None:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(json.dumps(report, indent=2, sort_keys=True))

    if args.gate and not bool(report["gate"]["passed"]):
        print("Gate failed:")
        for failure in report["gate"]["failures"]:
            print(f"  - {failure}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
