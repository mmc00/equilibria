"""Task 6: production equations are satisfied at Julia's solution.

Seed a Pyomo model with Julia's solved base point + calibrated params, build the
production equation group, and assert every constraint residual is ~0 (the point
satisfies the ported equations). This validates the port group-by-group,
independent of convergence.
"""

import pyomo.environ as pyo
import pytest

from equilibria.templates.gtap_julia.equations import build_group
from equilibria.templates.gtap_julia.solution import (
    dump_solution,
    load_solution,
    seed_model,
)


@pytest.mark.slow
def test_production_residuals_zero_at_julia_solution(tmp_path):
    sol_csv = dump_solution(dataset="sample", out_dir=tmp_path)
    sol = load_solution(sol_csv)

    m = pyo.ConcreteModel()
    seed_model(m, sol)  # builds sets/vars/params, seeds vars to Julia's solution
    cons = build_group(m, sol, "production")
    assert len(cons) > 0

    worst = 0.0
    worst_name = None
    for name, c in cons:
        for idx in c:
            r = abs(pyo.value(c[idx].body))
            if r > worst:
                worst, worst_name = r, f"{name}{idx}"
    assert worst < 1e-6, f"worst production residual {worst:.2e} at {worst_name}"
