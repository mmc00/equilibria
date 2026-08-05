"""Task 8: trade equations are satisfied at Julia's solution."""

import pyomo.environ as pyo
import pytest

from equilibria.templates.gtap_julia.equations import build_group
from equilibria.templates.gtap_julia.solution import (
    dump_solution,
    load_solution,
    seed_model,
)


@pytest.mark.slow
def test_trade_residuals_zero_at_julia_solution(tmp_path):
    sol = load_solution(dump_solution(dataset="sample", out_dir=tmp_path))
    m = pyo.ConcreteModel()
    seed_model(m, sol)
    cons = build_group(m, sol, "trade")
    assert len(cons) > 0

    worst, worst_name = 0.0, None
    for name, c in cons:
        for idx in c:
            r = abs(pyo.value(c[idx].body))
            if r > worst:
                worst, worst_name = r, f"{name}{idx}"
    assert worst < 1e-6, f"worst trade residual {worst:.2e} at {worst_name}"
