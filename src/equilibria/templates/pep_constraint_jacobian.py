"""Constraint and Jacobian harness for PEP IPOPT solves."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from equilibria.solver.transforms import pep_array_to_variables
from equilibria.templates.pep_model_equations import PEPModelEquations


class PEPConstraintJacobianHarness:
    """Encapsulate hard-constraint values, Jacobian values, and structure.

    This keeps IPOPT-facing constraint logic separate from the feasibility NLP
    wrapper so later epics can swap finite differences for analytic derivatives
    without reopening the whole `CGEProblem` class.
    """

    def __init__(
        self,
        *,
        equations: PEPModelEquations,
        sets: dict[str, list[str]],
        n_variables: int,
        hard_constraints: Sequence[str] | None = None,
    ) -> None:
        self.equations = equations
        self.sets = sets
        self.n_variables = int(n_variables)
        self.constraint_names = tuple(hard_constraints or ())
        self._constraint_scale: np.ndarray | None = None

    def evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        """Return scaled hard-constraint residuals in declared order."""
        if not self.constraint_names:
            return np.array([], dtype=float)

        residual_dict = self.equations.calculate_all_residuals(pep_array_to_variables(x, self.sets))
        residuals = np.array(
            [residual_dict.get(eq, 0.0) for eq in self.constraint_names],
            dtype=float,
        )
        if self._constraint_scale is None:
            self._constraint_scale = np.maximum(np.abs(residuals), 1.0)
        return residuals / self._constraint_scale

    def evaluate_jacobian_values(self, x: np.ndarray) -> np.ndarray:
        """Return dense Jacobian values in row-major order via finite differences."""
        m = len(self.constraint_names)
        n = len(x)
        if m == 0:
            return np.array([], dtype=float)

        base = self.evaluate_constraints(x)
        jac = np.zeros((m, n), dtype=float)
        fd_eps = np.sqrt(np.finfo(float).eps)
        for i in range(n):
            step = max(1e-8, fd_eps * max(abs(float(x[i])), 1.0))
            x_plus = x.copy()
            x_plus[i] += step
            c_plus = self.evaluate_constraints(x_plus)
            jac[:, i] = (c_plus - base) / step
        return jac.ravel()

    def jacobian_structure(self) -> tuple[np.ndarray, np.ndarray]:
        """Return dense row/column indices for the current constraint set."""
        m = len(self.constraint_names)
        n = self.n_variables
        if m == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        rows = np.repeat(np.arange(m, dtype=int), n)
        cols = np.tile(np.arange(n, dtype=int), m)
        return rows, cols
