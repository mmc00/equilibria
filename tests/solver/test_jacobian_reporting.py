from __future__ import annotations

from equilibria.solver import (
    compare_jacobian_modes,
    evaluate_jacobian_mode_gate,
    solver_stats_payload,
)


def test_solver_stats_payload_builds_common_schema() -> None:
    payload = solver_stats_payload(
        jacobian_stats={
            "jacobian_mode": "analytic",
            "constraint_eval_count": 10,
            "jacobian_eval_count": 5,
            "structure_eval_count": 1,
            "finite_difference_eval_count": 0,
            "jacobian_nonzero_count": 25,
            "hard_constraint_count": 8,
            "variable_count": 12,
        },
        wall_time_seconds=0.5,
        objective_eval_count=3,
    )

    assert payload["jacobian_mode"] == "analytic"
    assert payload["wall_time_seconds"] == 0.5
    assert payload["objective_eval_count"] == 3
    assert payload["jacobian_nonzero_count"] == 25


def test_compare_jacobian_modes_builds_generic_deltas() -> None:
    analytic = {
        "base": {
            "converged": True,
            "iterations": 4,
            "final_residual": 1e-9,
            "message": "ok",
            "solver_stats": {
                "wall_time_seconds": 0.2,
                "constraint_eval_count": 10,
                "jacobian_eval_count": 5,
                "finite_difference_eval_count": 0,
            },
            "comparison": {"passed": True, "mismatches": 0, "missing": 0},
        }
    }
    numeric = {
        "base": {
            "converged": True,
            "iterations": 4,
            "final_residual": 2e-9,
            "message": "ok",
            "solver_stats": {
                "wall_time_seconds": 2.0,
                "constraint_eval_count": 20,
                "jacobian_eval_count": 5,
                "finite_difference_eval_count": 100,
            },
            "comparison": {"passed": True, "mismatches": 0, "missing": 0},
        }
    }

    report = compare_jacobian_modes(analytic, numeric)

    assert report["base"]["deltas"]["iteration_delta"] == 0
    assert report["base"]["deltas"]["analytic_speedup_vs_numeric"] == 10.0
    assert report["base"]["deltas"]["finite_difference_eval_delta"] == -100


def test_evaluate_jacobian_mode_gate_flags_worse_analytic_result() -> None:
    mode_comparison = {
        "base": {
            "analytic": {
                "converged": True,
                "iterations": 4,
                "final_residual": 1e-9,
                "message": "ok",
                "solver_stats": {"finite_difference_eval_count": 1},
                "comparison": {"passed": False, "mismatches": 2, "missing": 0},
            },
            "numeric": {
                "converged": True,
                "iterations": 4,
                "final_residual": 1e-9,
                "message": "ok",
                "solver_stats": {"finite_difference_eval_count": 100},
                "comparison": {"passed": True, "mismatches": 0, "missing": 0},
            },
            "deltas": {},
        }
    }

    gate = evaluate_jacobian_mode_gate(mode_comparison, max_analytic_fd_evals=0)

    assert gate["passed"] is False
    assert any("finite_difference_eval_count" in msg for msg in gate["failures"])
    assert any("analytic parity failed while numeric parity passed" in msg for msg in gate["failures"])
