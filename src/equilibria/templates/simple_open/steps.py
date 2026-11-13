"""simple_open's own steps."""

from __future__ import annotations

from collections.abc import Mapping

from equilibria.core.recipe import StepContext
from equilibria.core.reflection import exogenous


def apply_shock(ctx: StepContext) -> None:
    """Set the exercise's shock in ``ctx.period``: new levels of instruments.

    ``ctx.shock`` maps an instrument (a fixed variable, e.g. ``tau_m``) to a
    float, for every cell, or to ``{index: float}`` (an index is a ``str`` for a
    one-set variable, a tuple otherwise).
    """
    m, period = ctx.model, ctx.period.name
    exo = exogenous(m)
    instruments = {name for name, _ in exo}
    for name, levels in ctx.shock.items():
        if name not in instruments:
            raise ValueError(
                f"shock on {name!r}: not an instrument of this model "
                f"(instruments: {', '.join(sorted(instruments))})"
            )
        cells = {key for n, key in exo if n == name}
        if isinstance(levels, Mapping):
            targets = {
                (k if isinstance(k, tuple) else (k,)): float(v)
                for k, v in levels.items()
            }
            missing = sorted(str(k) for k in targets if k not in cells)
            if missing:
                raise ValueError(f"shock on {name!r}: no index {', '.join(missing)}")
        else:
            targets = dict.fromkeys(cells, float(levels))
        var = m.component(name)
        for key, level in targets.items():
            var[(*key, period)].fix(level)
