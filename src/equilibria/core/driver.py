"""The driver (ADR-0011): flies a recipe over a model with periods.

For each period of the recipe: the steps before, the solve with the period's
strategy and solver, the residual, the steps after. It knows no model, no shock
and no closure, and keeps no state between runs.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

from equilibria.core.recipe import Recipe, StepContext
from equilibria.core.reflection import period_residual

NOT_SOLVED = "not solved"

_log = logging.getLogger(__name__)


def _solve(model: Any, plan: Any, period: str) -> str:
    """The period's status; an exception from the solver becomes an
    ``"error: <Type>: <message>"`` status, logged, so later periods still run."""
    if not plan.solver:
        return NOT_SOLVED
    try:
        return plan.strategy(model, plan.solver)
    except Exception as exc:  # reported in the results, never swallowed
        _log.warning("period %s: the solve raised %r", period, exc)
        return f"error: {type(exc).__name__}: {exc}"


def _residual(model: Any, period: str) -> float:
    """The period's residual; ``nan`` when the rows cannot be evaluated
    (e.g. a division by zero at the point the solve left)."""
    try:
        return period_residual(model, period)
    except (ArithmeticError, ValueError) as exc:
        _log.warning("period %s: residual not evaluable: %r", period, exc)
        return math.nan


def fly(
    model: Any,
    recipe: Recipe,
    *,
    template: Any = None,
    shock: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run every period of ``recipe`` on ``model``.

    Returns ``{period: {"code": status, "residual": largest row residual}}``.
    A period whose solve fails, or raises, is reported and the next one still
    runs. Exceptions from the steps are not caught: they are bugs, not results.
    """
    results: dict[str, dict[str, Any]] = {}
    for period in recipe.periods:
        plan = recipe.plan(period)
        ctx = StepContext(
            model=model,
            template=template,
            recipe=recipe,
            period=period,
            prior=recipe.prior(period),
            shock=dict(shock or {}),
        )
        for step in plan.before:
            step(ctx)
        code = _solve(model, plan, period.name)
        residual = _residual(model, period.name)
        for step in plan.after:
            step(ctx)
        results[period.name] = {"code": code, "residual": residual}
    return results
