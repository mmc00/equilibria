from __future__ import annotations

import numpy as np

from equilibria.templates import build_simple_open_contract
from equilibria.templates.simple_open_constraint_jacobian import (
    SimpleOpenConstraintJacobianHarness,
)


def test_simple_open_harness_analytic_mode_avoids_fd() -> None:
    contract = build_simple_open_contract("simple_open_v1")
    harness = SimpleOpenConstraintJacobianHarness(contract=contract, jacobian_mode="analytic")

    x0 = harness.benchmark_point
    residuals = harness.evaluate_constraints(x0)
    rows, cols = harness.jacobian_structure()
    values = harness.evaluate_jacobian_values(x0)
    stats = harness.stats()

    np.testing.assert_allclose(residuals, np.zeros_like(residuals), atol=1e-12)
    assert len(rows) == len(cols) == len(values)
    assert stats["jacobian_mode"] == "analytic"
    assert stats["finite_difference_eval_count"] == 0
    assert stats["jacobian_nonzero_count"] < len(contract.equations.include) * len(x0)


def test_simple_open_harness_numeric_mode_matches_analytic_reference() -> None:
    contract = build_simple_open_contract({"closure": {"name": "flexible_external_balance"}})
    analytic = SimpleOpenConstraintJacobianHarness(contract=contract, jacobian_mode="analytic")
    numeric = SimpleOpenConstraintJacobianHarness(contract=contract, jacobian_mode="numeric")

    x0 = analytic.benchmark_point
    a_rows, a_cols = analytic.jacobian_structure()
    n_rows, n_cols = numeric.jacobian_structure()
    a_vals = analytic.evaluate_jacobian_values(x0)
    n_vals = numeric.evaluate_jacobian_values(x0)

    assert np.array_equal(a_rows, n_rows)
    assert np.array_equal(a_cols, n_cols)
    np.testing.assert_allclose(a_vals, n_vals, rtol=1e-6, atol=1e-8)
    assert numeric.stats()["finite_difference_eval_count"] > 0
