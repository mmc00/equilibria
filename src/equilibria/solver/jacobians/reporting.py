"""Generic reporting helpers for analytic vs numeric Jacobian benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from equilibria.solver.jacobians.base import ConstraintJacobianSolverStats


@dataclass
class JacobianModeSummary:
    """Normalized one-mode solve summary for one scenario."""

    converged: bool
    iterations: int
    final_residual: float
    message: str
    solver_stats: dict[str, Any]
    comparison: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JacobianModeDelta:
    """Derived comparison metrics between analytic and numeric runs."""

    iteration_delta: int
    final_residual_delta: float
    wall_time_delta_seconds: float
    analytic_speedup_vs_numeric: float | None
    constraint_eval_delta: int
    jacobian_eval_delta: int
    finite_difference_eval_delta: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JacobianModeComparison:
    """Scenario-level analytic vs numeric comparison payload."""

    analytic: JacobianModeSummary
    numeric: JacobianModeSummary
    deltas: JacobianModeDelta

    def to_dict(self) -> dict[str, Any]:
        return {
            "analytic": self.analytic.to_dict(),
            "numeric": self.numeric.to_dict(),
            "deltas": self.deltas.to_dict(),
        }


@dataclass
class JacobianModeGateResult:
    """Pass/fail result for a structural Jacobian-mode gate."""

    passed: bool
    max_analytic_fd_evals: int
    failures: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def solver_stats_payload(
    *,
    jacobian_stats: dict[str, Any] | None = None,
    wall_time_seconds: float = 0.0,
    objective_eval_count: int = 0,
) -> dict[str, Any]:
    """Build the common public `solver_stats` payload."""

    base = dict(jacobian_stats or {})
    stats = ConstraintJacobianSolverStats(
        jacobian_mode=str(base.get("jacobian_mode", "analytic")),
        constraint_eval_count=int(base.get("constraint_eval_count", 0)),
        jacobian_eval_count=int(base.get("jacobian_eval_count", 0)),
        structure_eval_count=int(base.get("structure_eval_count", 0)),
        finite_difference_eval_count=int(base.get("finite_difference_eval_count", 0)),
        jacobian_nonzero_count=int(base.get("jacobian_nonzero_count", 0)),
        hard_constraint_count=int(base.get("hard_constraint_count", 0)),
        variable_count=int(base.get("variable_count", 0)),
        wall_time_seconds=float(wall_time_seconds),
        objective_eval_count=int(objective_eval_count),
    )
    return stats.to_dict()


def summarize_jacobian_mode_entry(entry: dict[str, Any]) -> JacobianModeSummary:
    """Normalize one scenario/base entry from a simulation report."""

    solve = dict(entry["solve"])
    return JacobianModeSummary(
        converged=bool(solve["converged"]),
        iterations=int(solve["iterations"]),
        final_residual=float(solve["final_residual"]),
        message=str(solve["message"]),
        solver_stats=dict(solve.get("solver_stats") or {}),
        comparison=entry.get("comparison"),
    )


def _safe_ratio(numerator: float | int, denominator: float | int) -> float | None:
    denom = float(denominator)
    if denom == 0.0:
        return None
    return float(numerator) / denom


def compare_jacobian_modes(
    analytic: dict[str, JacobianModeSummary | dict[str, Any]],
    numeric: dict[str, JacobianModeSummary | dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build generic analytic-vs-numeric comparison payloads by scenario."""

    out: dict[str, dict[str, Any]] = {}
    for scenario_name in analytic:
        analytic_entry = (
            analytic[scenario_name]
            if isinstance(analytic[scenario_name], JacobianModeSummary)
            else JacobianModeSummary(**analytic[scenario_name])
        )
        numeric_entry = (
            numeric[scenario_name]
            if isinstance(numeric[scenario_name], JacobianModeSummary)
            else JacobianModeSummary(**numeric[scenario_name])
        )
        analytic_stats = dict(analytic_entry.solver_stats or {})
        numeric_stats = dict(numeric_entry.solver_stats or {})
        comparison = JacobianModeComparison(
            analytic=analytic_entry,
            numeric=numeric_entry,
            deltas=JacobianModeDelta(
                iteration_delta=int(analytic_entry.iterations) - int(numeric_entry.iterations),
                final_residual_delta=float(analytic_entry.final_residual)
                - float(numeric_entry.final_residual),
                wall_time_delta_seconds=float(analytic_stats.get("wall_time_seconds", 0.0))
                - float(numeric_stats.get("wall_time_seconds", 0.0)),
                analytic_speedup_vs_numeric=_safe_ratio(
                    float(numeric_stats.get("wall_time_seconds", 0.0)),
                    float(analytic_stats.get("wall_time_seconds", 0.0)),
                ),
                constraint_eval_delta=int(analytic_stats.get("constraint_eval_count", 0))
                - int(numeric_stats.get("constraint_eval_count", 0)),
                jacobian_eval_delta=int(analytic_stats.get("jacobian_eval_count", 0))
                - int(numeric_stats.get("jacobian_eval_count", 0)),
                finite_difference_eval_delta=int(analytic_stats.get("finite_difference_eval_count", 0))
                - int(numeric_stats.get("finite_difference_eval_count", 0)),
            ),
        )
        out[scenario_name] = comparison.to_dict()
    return out


def evaluate_jacobian_mode_gate(
    mode_comparison: dict[str, dict[str, Any]],
    *,
    max_analytic_fd_evals: int,
) -> dict[str, Any]:
    """Evaluate a structural pass/fail gate over analytic-vs-numeric results."""

    failures: list[str] = []
    for scenario_name, payload in mode_comparison.items():
        analytic = dict(payload["analytic"])
        numeric = dict(payload["numeric"])
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

    return JacobianModeGateResult(
        passed=not failures,
        max_analytic_fd_evals=int(max_analytic_fd_evals),
        failures=failures,
    ).to_dict()
