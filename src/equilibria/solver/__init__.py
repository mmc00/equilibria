"""Canonical solver transform and guard utilities."""

from equilibria.solver.guards import rebuild_tax_detail_from_rates
from equilibria.solver.jacobians import (
    ConstraintJacobianHarness,
    ConstraintJacobianSolverStats,
    ConstraintJacobianStats,
    ConstraintJacobianStructure,
    JacobianModeComparison,
    JacobianModeDelta,
    JacobianModeGateResult,
    JacobianModeSummary,
    compare_jacobian_modes,
    evaluate_jacobian_mode_gate,
    solver_stats_payload,
    summarize_jacobian_mode_entry,
)
from equilibria.solver.transforms import pep_array_to_variables, pep_variables_to_array

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
    "pep_array_to_variables",
    "pep_variables_to_array",
    "rebuild_tax_detail_from_rates",
]
