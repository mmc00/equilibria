"""The driver (ADR-0011): flies a recipe over a model with periods.

For each period of the recipe: the steps before, the solve with the period's
strategy and solver, the residual, the steps after. It knows no model, no shock
and no closure, and keeps no state between runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from equilibria.core.recipe import Recipe, StepContext
from equilibria.core.reflection import period_residual

NOT_SOLVED = "not solved"


def fly(
    model: Any,
    recipe: Recipe,
    *,
    template: Any = None,
    shock: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run every period of ``recipe`` on ``model``.

    Returns ``{period: {"code": status, "residual": largest row residual}}``.
    A period whose solve fails is reported and the next one still runs.
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
        code = plan.strategy(model, plan.solver) if plan.solver else NOT_SOLVED
        residual = period_residual(model, period.name)
        for step in plan.after:
            step(ctx)
        results[period.name] = {"code": code, "residual": residual}
    return results
