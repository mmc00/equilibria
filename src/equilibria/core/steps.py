"""Generic steps any recipe can use (ADR-0011).

A step is ``step(ctx) -> None``: it mutates ``ctx.model`` in place. Steps that
serve one model live with that model (``templates/<model>/steps.py``).
"""

from __future__ import annotations

from typing import Any, cast

import pyomo.environ as pyo

from equilibria.core.recipe import Step, StepContext
from equilibria.core.reflection import exogenous, period_of

__all__ = ["Step", "StepContext", "freeze_other_periods", "seed_from_prior"]


def _cells(m: Any):
    """``(name, index without t, period, cell)`` of every variable cell."""
    for v in m.component_objects(pyo.Var, active=True):
        for k in v:
            key = k if isinstance(k, tuple) else (k,)
            yield v.name, key[:-1], key[-1], cast("Any", v[k])


def freeze_other_periods(ctx: StepContext) -> None:
    """Leave only ``ctx.period`` free: its cells unfixed and its rows active;
    every other period's cells fixed at their value and its rows inactive. An
    exogenous cell (fixed in the single-period model) is never touched."""
    m, period = ctx.model, ctx.period.name
    exo = exogenous(m)
    for name, key, t, cell in _cells(m):
        if (name, key) in exo:
            continue
        if t == period:
            cell.unfix()
        elif not cell.fixed:
            cell.fix()
    for con in m.component_objects(pyo.Constraint):
        for k in con:
            row = cast("Any", con[k])
            if period_of(k) == period:
                row.activate()
            else:
                row.deactivate()


def seed_from_prior(ctx: StepContext) -> None:
    """Start ``ctx.period`` from the prior period's values (exogenous cells
    keep theirs). No-op for the first period."""
    if ctx.prior is None:
        return
    m, period, prior = ctx.model, ctx.period.name, ctx.prior.name
    exo = exogenous(m)
    for name, key, t, cell in _cells(m):
        if t != period or (name, key) in exo:
            continue
        cell.set_value(m.component(name)[(*key, prior)].value, skip_validation=True)
