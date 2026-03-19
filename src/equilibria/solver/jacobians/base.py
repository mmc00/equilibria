"""Generic constraint/Jacobian harness base for nonlinear model solvers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class ConstraintJacobianStats:
    """Evaluation counters and lightweight Jacobian metadata."""

    jacobian_mode: str
    constraint_eval_count: int = 0
    jacobian_eval_count: int = 0
    structure_eval_count: int = 0
    finite_difference_eval_count: int = 0
    jacobian_nonzero_count: int = 0
    hard_constraint_count: int = 0
    variable_count: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass
class ConstraintJacobianSolverStats(ConstraintJacobianStats):
    """Jacobian stats extended with solve-level timing counters."""

    wall_time_seconds: float = 0.0
    objective_eval_count: int = 0


@dataclass
class ConstraintJacobianStructure:
    """Cached sparse row/column structure for one harness instance."""

    rows: np.ndarray
    cols: np.ndarray


class ConstraintJacobianHarness:
    """Generic sparse Jacobian harness with analytic/numeric fallback.

    Subclasses provide three model-specific hooks:

    - how to build the per-point evaluation context
    - how to evaluate constraint residuals in declared name order
    - which analytic derivatives are available for one row
    """

    def __init__(
        self,
        *,
        n_variables: int,
        constraint_names: Sequence[str] | None = None,
        variable_names: Sequence[str] | None = None,
        sparsity_reference_x: np.ndarray | None = None,
        sparsity_tol: float = 1e-12,
        jacobian_mode: str = "analytic",
    ) -> None:
        self.n_variables = int(n_variables)
        self.constraint_names = tuple(constraint_names or ())
        self.variable_names = tuple(variable_names or ())
        self.variable_index = {name: idx for idx, name in enumerate(self.variable_names)}
        self.sparsity_reference_x = (
            None if sparsity_reference_x is None else np.array(sparsity_reference_x, dtype=float)
        )
        self.sparsity_tol = float(sparsity_tol)
        self.jacobian_mode = str(jacobian_mode).strip().lower()
        if self.jacobian_mode not in {"analytic", "numeric"}:
            raise ValueError(f"Unsupported Jacobian mode: {jacobian_mode!r}")
        self._constraint_scale: np.ndarray | None = None
        self._structure: ConstraintJacobianStructure | None = None
        self._stats = ConstraintJacobianStats(
            jacobian_mode=self.jacobian_mode,
            hard_constraint_count=len(self.constraint_names),
            variable_count=self.n_variables,
        )

    def evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        """Return scaled hard-constraint residuals in declared order."""
        self._stats.constraint_eval_count += 1
        if not self.constraint_names:
            return np.array([], dtype=float)

        context = self._build_context(x)
        residual_dict = self._calculate_constraint_residual_dict(context)
        residuals = np.array(
            [residual_dict.get(name, 0.0) for name in self.constraint_names],
            dtype=float,
        )
        if self._constraint_scale is None:
            self._constraint_scale = np.maximum(np.abs(residuals), 1.0)
        return residuals / self._constraint_scale

    def evaluate_jacobian_values(self, x: np.ndarray) -> np.ndarray:
        """Return Jacobian values aligned with the cached sparse structure."""
        self._stats.jacobian_eval_count += 1
        rows, cols = self.jacobian_structure()
        m = len(self.constraint_names)
        n = len(x)
        if m == 0 or len(rows) == 0:
            return np.array([], dtype=float)

        base = self.evaluate_constraints(x)
        context = self._build_context(x)
        jac = np.zeros((m, n), dtype=float)
        analytic_rows: set[int] = set()
        if self.jacobian_mode == "analytic":
            for row, name in enumerate(self.constraint_names):
                analytic = self._analytic_constraint_derivatives(name, context)
                if analytic is None:
                    continue
                analytic_rows.add(row)
                scale = float(self._constraint_scale[row]) if self._constraint_scale is not None else 1.0
                for col, value in analytic.items():
                    jac[row, col] = float(value) / scale

        numeric_rows = tuple(row for row in range(m) if row not in analytic_rows)
        if numeric_rows:
            fd_eps = np.sqrt(np.finfo(float).eps)
            for col in range(n):
                step = max(1e-8, fd_eps * max(abs(float(x[col])), 1.0))
                x_plus = x.copy()
                x_plus[col] += step
                self._stats.finite_difference_eval_count += 1
                c_plus = self.evaluate_constraints(x_plus)
                delta = (c_plus - base) / step
                for row in numeric_rows:
                    jac[row, col] = delta[row]
        return jac[rows, cols]

    def jacobian_structure(self) -> tuple[np.ndarray, np.ndarray]:
        """Return sparse row/column structure for the current constraint set."""
        if self._structure is not None:
            return self._structure.rows, self._structure.cols
        self._stats.structure_eval_count += 1

        m = len(self.constraint_names)
        n = self.n_variables
        if m == 0:
            empty = ConstraintJacobianStructure(
                rows=np.array([], dtype=int),
                cols=np.array([], dtype=int),
            )
            self._structure = empty
            return empty.rows, empty.cols

        if self.sparsity_reference_x is None:
            dense = ConstraintJacobianStructure(
                rows=np.repeat(np.arange(m, dtype=int), n),
                cols=np.tile(np.arange(n, dtype=int), m),
            )
            self._structure = dense
            self._stats.jacobian_nonzero_count = int(len(dense.rows))
            return dense.rows, dense.cols

        base = self.evaluate_constraints(self.sparsity_reference_x)
        context = self._build_context(self.sparsity_reference_x)
        fd_eps = np.sqrt(np.finfo(float).eps)
        pairs: set[tuple[int, int]] = set()
        analytic_rows: set[int] = set()
        if self.jacobian_mode == "analytic":
            for row, name in enumerate(self.constraint_names):
                analytic = self._analytic_constraint_derivatives(name, context)
                if analytic is None:
                    continue
                analytic_rows.add(row)
                for col in analytic:
                    pairs.add((int(row), int(col)))

        numeric_rows = tuple(row for row in range(m) if row not in analytic_rows)
        if numeric_rows:
            for col in range(n):
                step = max(1e-8, fd_eps * max(abs(float(self.sparsity_reference_x[col])), 1.0))
                x_plus = self.sparsity_reference_x.copy()
                x_plus[col] += step
                self._stats.finite_difference_eval_count += 1
                c_plus = self.evaluate_constraints(x_plus)
                delta = (c_plus - base) / step
                hit_rows = np.flatnonzero(np.abs(delta) > self.sparsity_tol)
                for row in hit_rows.tolist():
                    if row not in numeric_rows:
                        continue
                    pairs.add((int(row), int(col)))

        if not pairs:
            structure = ConstraintJacobianStructure(
                rows=np.repeat(np.arange(m, dtype=int), n),
                cols=np.tile(np.arange(n, dtype=int), m),
            )
            self._structure = structure
            self._stats.jacobian_nonzero_count = int(len(structure.rows))
            return structure.rows, structure.cols

        ordered_pairs = sorted(pairs)
        structure = ConstraintJacobianStructure(
            rows=np.array([row for row, _ in ordered_pairs], dtype=int),
            cols=np.array([col for _, col in ordered_pairs], dtype=int),
        )
        self._structure = structure
        self._stats.jacobian_nonzero_count = int(len(structure.rows))
        return structure.rows, structure.cols

    def stats(self) -> dict[str, int | str]:
        """Return lightweight evaluation counters and Jacobian metadata."""
        self.jacobian_structure()
        return self._stats.to_dict()

    @property
    def constraint_eval_count(self) -> int:
        return self._stats.constraint_eval_count

    @property
    def jacobian_eval_count(self) -> int:
        return self._stats.jacobian_eval_count

    @property
    def structure_eval_count(self) -> int:
        return self._stats.structure_eval_count

    @property
    def finite_difference_eval_count(self) -> int:
        return self._stats.finite_difference_eval_count

    def _var_idx(self, name: str) -> int | None:
        return self.variable_index.get(name)

    def _build_context(self, x: np.ndarray) -> Any:
        raise NotImplementedError

    def _calculate_constraint_residual_dict(self, context: Any) -> Mapping[str, float]:
        raise NotImplementedError

    def _analytic_constraint_derivatives(
        self,
        constraint_name: str,
        context: Any,
    ) -> dict[int, float] | None:
        return None
