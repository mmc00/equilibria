from __future__ import annotations

import numpy as np
import pytest

from equilibria.solver import ConstraintJacobianHarness


class _ToyConstraintHarness(ConstraintJacobianHarness):
    def _build_context(self, x: np.ndarray) -> np.ndarray:
        return np.array(x, dtype=float)

    def _calculate_constraint_residual_dict(self, context: np.ndarray) -> dict[str, float]:
        return {
            "EQ_A": float(context[0] + 2.0 * context[1]),
            "EQ_B": float((context[0] ** 2) + context[1]),
        }

    def _analytic_constraint_derivatives(
        self,
        constraint_name: str,
        context: np.ndarray,
    ) -> dict[int, float] | None:
        if constraint_name == "EQ_A":
            return {0: 1.0, 1: 2.0}
        if constraint_name == "EQ_B":
            return {0: 2.0 * float(context[0]), 1: 1.0}
        return None


def test_generic_constraint_jacobian_harness_analytic_mode_avoids_fd() -> None:
    x0 = np.array([3.0, 4.0], dtype=float)
    harness = _ToyConstraintHarness(
        n_variables=2,
        constraint_names=["EQ_A", "EQ_B"],
        variable_names=["x0", "x1"],
        sparsity_reference_x=x0,
        jacobian_mode="analytic",
    )

    residuals = harness.evaluate_constraints(x0)
    rows, cols = harness.jacobian_structure()
    values = harness.evaluate_jacobian_values(x0)
    stats = harness.stats()

    np.testing.assert_allclose(residuals, np.array([1.0, 1.0]))
    assert rows.tolist() == [0, 0, 1, 1]
    assert cols.tolist() == [0, 1, 0, 1]
    np.testing.assert_allclose(values, np.array([1.0 / 11.0, 2.0 / 11.0, 6.0 / 13.0, 1.0 / 13.0]))
    assert stats["jacobian_mode"] == "analytic"
    assert stats["finite_difference_eval_count"] == 0
    assert stats["jacobian_nonzero_count"] == 4


def test_generic_constraint_jacobian_harness_numeric_mode_uses_fd() -> None:
    x0 = np.array([3.0, 4.0], dtype=float)
    harness = _ToyConstraintHarness(
        n_variables=2,
        constraint_names=["EQ_A", "EQ_B"],
        variable_names=["x0", "x1"],
        sparsity_reference_x=x0,
        jacobian_mode="numeric",
    )

    rows, cols = harness.jacobian_structure()
    values = harness.evaluate_jacobian_values(x0)
    stats = harness.stats()

    assert rows.tolist() == [0, 0, 1, 1]
    assert cols.tolist() == [0, 1, 0, 1]
    np.testing.assert_allclose(values, np.array([1.0 / 11.0, 2.0 / 11.0, 6.0 / 13.0, 1.0 / 13.0]), rtol=1e-5, atol=1e-8)
    assert stats["jacobian_mode"] == "numeric"
    assert stats["finite_difference_eval_count"] > 0


def test_generic_constraint_jacobian_harness_rejects_bad_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported Jacobian mode"):
        _ToyConstraintHarness(
            n_variables=1,
            constraint_names=["EQ_A"],
            variable_names=["x0"],
            jacobian_mode="bad",
        )
