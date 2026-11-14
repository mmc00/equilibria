"""Recipe: the periods, their kinds and what to do in each kind."""

from __future__ import annotations

import pytest

from equilibria.core.recipe import PERIOD_KINDS, KindPlan, Period, Recipe

STATIC = (
    Period("base", "calibrate"),
    Period("check", "replicate"),
    Period("shock", "shock"),
)
PLANS = {"calibrate": KindPlan(), "replicate": KindPlan(), "shock": KindPlan()}


def test_the_core_ships_four_kinds() -> None:
    assert PERIOD_KINDS == ("calibrate", "replicate", "shock", "step")


@pytest.mark.parametrize(
    "periods,plans",
    [
        ((), PLANS),
        ((Period("base", "calibrate"), Period("base", "replicate")), PLANS),
        ((Period("base", "calibrate"),), {}),
    ],
    ids=["no periods", "duplicate names", "kind without plan"],
)
def test_recipe_rejects_bad_periods(periods: tuple[Period, ...], plans: dict) -> None:
    with pytest.raises(ValueError):
        Recipe(periods=periods, plans=plans)


def test_prior_is_the_previous_period_and_none_for_the_first() -> None:
    recipe = Recipe(periods=STATIC, plans=PLANS)
    assert recipe.names == ("base", "check", "shock")
    assert recipe.prior(STATIC[0]) is None
    assert recipe.prior(STATIC[2]) == STATIC[1]
    assert recipe.plan(STATIC[1]) is PLANS["replicate"]
    assert recipe.has_kind("shock") and not recipe.has_kind("step")


def test_a_recipe_may_add_its_own_kind() -> None:
    plan = KindPlan()
    recipe = Recipe(periods=(Period("y1", "mykind"),), plans={"mykind": plan})
    assert recipe.plan(recipe.periods[0]) is plan
