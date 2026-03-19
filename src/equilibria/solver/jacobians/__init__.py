"""Reusable Jacobian harness primitives for nonlinear model solvers."""

from equilibria.solver.jacobians.base import (
    ConstraintJacobianHarness,
    ConstraintJacobianSolverStats,
    ConstraintJacobianStats,
    ConstraintJacobianStructure,
)
from equilibria.solver.jacobians.reporting import (
    JacobianModeComparison,
    JacobianModeDelta,
    JacobianModeGateResult,
    JacobianModeSummary,
    compare_jacobian_modes,
    evaluate_jacobian_mode_gate,
    solver_stats_payload,
    summarize_jacobian_mode_entry,
)

__all__ = [
    "ConstraintJacobianHarness",
    "ConstraintJacobianSolverStats",
    "ConstraintJacobianStats",
    "ConstraintJacobianStructure",
    "JacobianModeSummary",
    "JacobianModeDelta",
    "JacobianModeComparison",
    "JacobianModeGateResult",
    "solver_stats_payload",
    "summarize_jacobian_mode_entry",
    "compare_jacobian_modes",
    "evaluate_jacobian_mode_gate",
]
