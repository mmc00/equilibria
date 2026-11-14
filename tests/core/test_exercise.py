"""Exercise: one template, one recipe, one shock, its own @overwrite blocks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from tests.backends.test_pyomo_periods import single_period

from equilibria.backends.pyomo_periods import PyomoPeriodModel, reflect
from equilibria.core import Exercise
from equilibria.core.recipe import KindPlan, Period, Recipe

STATIC = Recipe(
    (
        Period("base", "calibrate"),
        Period("check", "replicate"),
        Period("shock", "shock"),
    ),
    {"calibrate": KindPlan(), "replicate": KindPlan(), "shock": KindPlan()},
)
NO_SHOCK = Recipe(
    (Period("base", "calibrate"), Period("check", "replicate")),
    {"calibrate": KindPlan(), "replicate": KindPlan()},
)


class FakeTemplate:
    def __init__(self) -> None:
        self.builds: list[tuple[tuple[str, ...], tuple[Any, ...]]] = []

    def build(self, periods: Sequence[str], overwrite: Sequence[Any] = ()) -> Any:
        self.builds.append((tuple(periods), tuple(overwrite)))
        return PyomoPeriodModel(reflect(single_period(), periods))

    def default_recipe(self) -> Recipe:
        return STATIC


def test_run_builds_and_flies_the_default_recipe() -> None:
    template = FakeTemplate()
    ex = Exercise(template)
    assert ex.model is None
    results = ex.run()
    assert set(results) == {"base", "check", "shock"}
    assert template.builds == [(("base", "check", "shock"), ())]
    assert ex.model is not None


def test_a_shock_without_a_shock_period_fails_at_creation() -> None:
    with pytest.raises(ValueError, match="shock"):
        Exercise(FakeTemplate(), recipe=NO_SHOCK, shock={"tau": 0.0})


def test_an_empty_shock_needs_no_shock_period() -> None:
    assert set(Exercise(FakeTemplate(), recipe=NO_SHOCK).run()) == {"base", "check"}


def test_each_run_gets_its_own_model() -> None:
    ex = Exercise(FakeTemplate())
    ex.run()
    first = ex.model
    ex.run()
    assert ex.model is not first


def test_overwrite_reaches_the_template_build() -> None:
    template = FakeTemplate()
    Exercise(template, overwrite=["my_block"]).run()
    assert template.builds == [(("base", "check", "shock"), ("my_block",))]
