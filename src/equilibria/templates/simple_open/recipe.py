"""simple_open's recipe: comparative statics, base -> check -> shock (stdcge)."""

from __future__ import annotations

from equilibria.core.recipe import KindPlan, Period, Recipe, Solver
from equilibria.core.steps import freeze_other_periods, seed_from_prior
from equilibria.solver.ipopt import ipopt
from equilibria.templates.simple_open.steps import apply_shock


def stdcge_static(solver: Solver | None = None) -> Recipe:
    """``base`` is the calibrated point and is not solved (as in GAMS); ``check``
    solves with no shock; ``shock`` solves with the exercise's shock. Each
    period starts from the previous one."""
    solve = solver or ipopt()
    return Recipe(
        periods=(
            Period("base", "calibrate"),
            Period("check", "replicate"),
            Period("shock", "shock"),
        ),
        plans={
            "calibrate": KindPlan(before=(freeze_other_periods,)),
            "replicate": KindPlan(
                before=(freeze_other_periods, seed_from_prior), solver=solve
            ),
            "shock": KindPlan(
                before=(freeze_other_periods, seed_from_prior, apply_shock),
                solver=solve,
            ),
        },
    )
