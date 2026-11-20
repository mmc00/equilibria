"""The driver (ADR-0011): flies a recipe over a model with periods.

For each period of the recipe: the steps before, the solve with the period's
strategy and solver, the residual, the steps after. It knows no model, no shock
and no closure, and keeps no state between runs.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping
from typing import Any

from equilibria.core.period_model import PeriodModel
from equilibria.core.recipe import KindPlan, Recipe, StepContext

NOT_SOLVED = "not solved"

_log = logging.getLogger(__name__)


def _solve(model: PeriodModel, plan: KindPlan, period: str) -> str:
    """The period's status; an exception from the solver becomes an
    ``"error: <Type>: <message>"`` status, logged, so later periods still run."""
    if not plan.solver:
        return NOT_SOLVED
    try:
        return plan.strategy(model, plan.solver)
    except Exception as exc:  # reported in the results, never swallowed
        _log.warning("period %s: the solve raised %r", period, exc)
        return f"error: {type(exc).__name__}: {exc}"


def _residual(model: PeriodModel, period: str) -> float:
    """The period's residual; ``nan`` when the rows cannot be evaluated
    (e.g. a division by zero at the point the solve left)."""
    t0 = time.perf_counter()
    try:
        r = model.residual(period)
    except (ArithmeticError, ValueError) as exc:
        _log.warning("period %s: residual not evaluable: %r", period, exc)
        return math.nan
    _log.info("period %s: residual %.3g (%.1f s)", period, r, time.perf_counter() - t0)
    return r


def fly(
    model: PeriodModel,
    recipe: Recipe,
    *,
    template: Any = None,
    shock: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run every period of ``recipe`` on ``model`` (a ``PeriodModel``).

    Returns ``{period: {"code": status, "residual": largest row residual}}``.
    A period whose solve fails, or raises, is reported and the next one still
    runs. Exceptions from the steps are not caught: they are bugs, not results.
    """
    return {
        period.name: fly_period(
            model, recipe, period.name, template=template, shock=shock
        )
        for period in recipe.periods
    }


def fly_period(
    model: PeriodModel,
    recipe: Recipe,
    period: str,
    *,
    template: Any = None,
    shock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one period of ``recipe`` on ``model``, as :func:`fly` runs it (its steps
    before, the solve, the residual, its steps after); returns its
    ``{"code", "residual"}``. Flying the periods one by one in the recipe's order
    is :func:`fly`; a caller may look at the model between them."""
    by_name = {p.name: p for p in recipe.periods}
    if period not in by_name:
        raise ValueError(
            f"no period {period!r} in the recipe (periods: {', '.join(by_name)})"
        )
    p = by_name[period]
    plan = recipe.plan(p)
    ctx = StepContext(
        model=model,
        template=template,
        recipe=recipe,
        period=p,
        prior=recipe.prior(p),
        shock=dict(shock or {}),
    )
    for step in plan.before:
        step(ctx)
    code = _solve(model, plan, p.name)
    residual = _residual(model, p.name)
    ctx.status = code
    for step in plan.after:
        step(ctx)
    return {"code": code, "residual": residual}
