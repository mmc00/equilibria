"""The generic steps: freeze the other periods, seed from the prior one."""

from __future__ import annotations

from typing import Any

from tests.backends.test_pyomo_periods import single_period

from equilibria.backends.pyomo_periods import PyomoPeriodModel, reflect
from equilibria.core.recipe import KindPlan, Period, Recipe, StepContext
from equilibria.core.steps import freeze_other_periods, seed_from_prior

PERIODS = (Period("base", "calibrate"), Period("check", "replicate"))
RECIPE = Recipe(
    periods=PERIODS, plans={"calibrate": KindPlan(), "replicate": KindPlan()}
)


def ctx_for(m: Any, i: int) -> StepContext:
    return StepContext(
        model=PyomoPeriodModel(m),
        template=None,
        recipe=RECIPE,
        period=PERIODS[i],
        prior=RECIPE.prior(PERIODS[i]),
    )


def test_freeze_leaves_only_the_period_free_and_its_rows_active() -> None:
    m = reflect(single_period(), RECIPE.names)
    freeze_other_periods(ctx_for(m, 1))
    for i in ("a", "b"):
        assert m.x[i, "base"].fixed and not m.x[i, "check"].fixed
        assert not m.c[i, "base"].active and m.c[i, "check"].active
    assert m.y["base"].fixed and not m.y["check"].fixed
    assert not m.d["base"].active and m.d["check"].active


def test_freeze_then_back_to_an_earlier_period_frees_it_again() -> None:
    m = reflect(single_period(), RECIPE.names)
    freeze_other_periods(ctx_for(m, 1))
    freeze_other_periods(ctx_for(m, 0))
    assert not m.x["a", "base"].fixed and m.x["a", "check"].fixed
    assert m.c["a", "base"].active and not m.c["a", "check"].active


def test_freeze_never_frees_an_exogenous_cell() -> None:
    m = reflect(single_period(), RECIPE.names)
    freeze_other_periods(ctx_for(m, 1))
    assert all(m.tau[i, t].fixed for i in ("a", "b") for t in RECIPE.names)


def test_seed_copies_the_prior_period_except_exogenous_cells() -> None:
    m = reflect(single_period(), RECIPE.names)
    for i in ("a", "b"):
        m.x[i, "base"].set_value(5.0)
        m.tau[i, "check"].fix(7.0)
    seed_from_prior(ctx_for(m, 1))
    assert all(m.x[i, "check"].value == 5.0 for i in ("a", "b"))
    assert all(m.tau[i, "check"].value == 7.0 for i in ("a", "b"))


def test_seed_without_prior_does_nothing() -> None:
    m = reflect(single_period(), RECIPE.names)
    m.x["a", "base"].set_value(5.0)
    seed_from_prior(ctx_for(m, 0))
    assert m.x["a", "base"].value == 5.0
