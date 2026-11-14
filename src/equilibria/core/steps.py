"""Generic steps any recipe can use (ADR-0011).

A step is ``step(ctx) -> None``: it changes ``ctx.model`` (a ``PeriodModel``)
in place, through its operations only, so it runs on any backend. Steps that
serve one model live with that model (``templates/<model>/steps.py``).
"""

from __future__ import annotations

from equilibria.core.recipe import Step, StepContext

__all__ = ["Step", "StepContext", "freeze_other_periods", "seed_from_prior"]


def freeze_other_periods(ctx: StepContext) -> None:
    """Leave only ``ctx.period`` free: its cells unfixed and its equations
    active; every other period's cells fixed at their level and its equations
    inactive. An exogenous cell (fixed in the single-period model) is never
    touched. Run it before any step that fixes a cell of the period."""
    m, period = ctx.model, ctx.period.name
    exo = m.exogenous()
    for t in m.periods:
        for name, index in m.cells(t):
            if (name, index) in exo:
                continue
            if t == period:
                m.unfix(name, index, t)
            elif not m.is_fixed(name, index, t):
                m.fix(name, index, t)
    m.activate_rows(period)


def seed_from_prior(ctx: StepContext) -> None:
    """Start ``ctx.period`` from the prior period's levels (exogenous cells
    keep theirs). No-op for the first period."""
    if ctx.prior is None:
        return
    m, period, prior = ctx.model, ctx.period.name, ctx.prior.name
    exo = m.exogenous()
    for name, index in m.cells(period):
        if (name, index) not in exo:
            m.set_value(name, index, period, m.value(name, index, prior))
