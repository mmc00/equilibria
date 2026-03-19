"""Canonical solver transform and guard utilities."""

from equilibria.solver.guards import rebuild_tax_detail_from_rates
from equilibria.solver.jacobians import (
    ConstraintJacobianHarness,
    ConstraintJacobianStats,
    ConstraintJacobianStructure,
)
from equilibria.solver.transforms import pep_array_to_variables, pep_variables_to_array

__all__ = [
    "ConstraintJacobianHarness",
    "ConstraintJacobianStats",
    "ConstraintJacobianStructure",
    "pep_array_to_variables",
    "pep_variables_to_array",
    "rebuild_tax_detail_from_rates",
]
