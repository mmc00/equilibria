"""Squaring helpers for the GTAP blocks: GAMS $-condition emulation.

GAMS generates demand equations only where the benchmark share is nonzero
(`xdeq(r,i,aa,t)$(alphad ne 0)`) and leaves the degenerate variable inert
(its aggregations are also $-conditioned). Our newton_tr solves a SQUARE NLP,
so a free inert variable is a real empty column; the faithful adaptation is to
fix it to 0 while omitting its defining equation. This helper does both.
"""

from __future__ import annotations

from typing import Any


def skip_degenerate_cell(model: Any, var: Any, idx: Any) -> None:
    """Omit a degenerate cell's equation and fix its variable to 0.

    Returns ``None`` so the blocks bridge (``pyomo_backend.py:323-325``) drops
    this tuple — the Constraint is born with a restricted index, square like
    GAMS. Fixing the variable keeps it from being a free empty column in the
    square NLP. Idempotent and safe if the var index is missing.
    """
    try:
        vd = var[idx]
        if not vd.fixed:
            vd.fix(0.0)
    except (KeyError, AttributeError, TypeError):
        pass
    return None
