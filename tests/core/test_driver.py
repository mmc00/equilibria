"""fly: the driver runs each period's steps, solves and measures, knowing no model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tests.core.test_reflection import single_period

from equilibria.core import driver
from equilibria.core.driver import fly
from equilibria.core.recipe import KindPlan, Period, Recipe, StepContext
from equilibria.core.reflection import reflect

PERIODS = (
    Period("base", "calibrate"),
    Period("check", "replicate"),
    Period("shock", "shock"),
)


def logging_plan(log: list[str], code: str | None) -> KindPlan:
    def before(ctx: StepContext) -> None:
        log.append(f"{ctx.period.name}:before")

    def after(ctx: StepContext) -> None:
        log.append(f"{ctx.period.name}:after")

    def solver(model: Any) -> str:
        log.append("solve")
        assert code is not None
        return code

    return KindPlan(before=(before,), after=(after,), solver=solver if code else None)


def model() -> Any:
    return reflect(single_period(), [p.name for p in PERIODS])


def test_runs_before_solve_after_in_order_for_each_period() -> None:
    log: list[str] = []
    plan = logging_plan(log, "optimal")
    recipe = Recipe(PERIODS, dict.fromkeys(("calibrate", "replicate", "shock"), plan))
    fly(model(), recipe)
    assert log == [
        "base:before", "solve", "base:after",
        "check:before", "solve", "check:after",
        "shock:before", "solve", "shock:after",
    ]  # fmt: skip


def test_a_period_without_solver_is_not_solved() -> None:
    log: list[str] = []
    solved = logging_plan(log, "optimal")
    recipe = Recipe(
        PERIODS,
        {"calibrate": logging_plan(log, None), "replicate": solved, "shock": solved},
    )
    results = fly(model(), recipe)
    assert results["base"]["code"] == "not solved"
    assert log[:3] == ["base:before", "base:after", "check:before"]


def test_a_failed_period_is_reported_and_the_next_still_runs() -> None:
    log: list[str] = []
    recipe = Recipe(
        PERIODS,
        {
            "calibrate": logging_plan(log, "optimal"),
            "replicate": logging_plan(log, "infeasible"),
            "shock": logging_plan(log, "optimal"),
        },
    )
    results = fly(model(), recipe)
    assert [results[p]["code"] for p in ("base", "check", "shock")] == [
        "optimal",
        "infeasible",
        "optimal",
    ]
    assert log[-1] == "shock:after"


def test_residual_is_measured_on_the_period_rows() -> None:
    m = model()
    m.x["a", "shock"].set_value(7.0)
    plan = KindPlan()
    results = fly(
        m, Recipe(PERIODS, dict.fromkeys(("calibrate", "replicate", "shock"), plan))
    )
    assert results["check"]["residual"] == pytest.approx(0.0, abs=1e-12)
    assert results["shock"]["residual"] == pytest.approx(0.9)


def test_steps_see_the_template_shock_and_prior() -> None:
    seen: list[tuple] = []

    def look(ctx: StepContext) -> None:
        seen.append((ctx.template, dict(ctx.shock), ctx.prior and ctx.prior.name))

    plan = KindPlan(before=(look,))
    recipe = Recipe(PERIODS, dict.fromkeys(("calibrate", "replicate", "shock"), plan))
    fly(model(), recipe, template="T", shock={"tau": 0.0})
    assert seen == [
        ("T", {"tau": 0.0}, None),
        ("T", {"tau": 0.0}, "base"),
        ("T", {"tau": 0.0}, "check"),
    ]


def test_driver_stays_small() -> None:
    assert len(Path(driver.__file__).read_text().splitlines()) <= 300
