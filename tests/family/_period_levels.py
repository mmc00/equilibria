"""Solve one comparative-static exercise on either backend through the core driver and
read every Var cell back through the ``PeriodModel`` protocol both backends implement,
so the family route can be checked against a live Pyomo run (MAR-251)."""

from __future__ import annotations

import math
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

Cell = tuple[str, tuple, str]


@contextmanager
def solver_env(extra: Mapping[str, str] | None = None) -> Iterator[None]:
    """The Newton-TR + MUMPS settings the family route requires, NLP-square on, for
    both backends (so both solve the same system)."""
    from tests.family._prep import _newton_mumps

    env = {**_newton_mumps(), "EQUILIBRIA_GTAP_SOLVE_NLP": "1", **(extra or {})}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def levels(model: Any) -> dict[Cell, float]:
    """``{(var, index, period): level}`` of every Var cell of every period."""
    out: dict[Cell, float] = {}
    for period in model.periods:
        for name in model.var_names():
            for idx, v in model.values(name, period).items():
                out[name, idx, period] = float(v)
    return out


def mismatches(
    got: Mapping[Cell, float], want: Mapping[Cell, float], rel: float
) -> list[str]:
    """Cells of ``want`` (the reference) that ``got`` lacks or puts beyond ``rel``
    (relative, with an absolute floor of ``rel`` for cells near 0), and cells of
    ``got`` that ``want`` lacks."""
    bad = [f"{k}: extra" for k in got if k not in want]
    for k, w in want.items():
        if k not in got:
            bad.append(f"{k}: missing")
            continue
        g = got[k]
        if math.isnan(w) and math.isnan(g):
            continue
        if not abs(g - w) <= rel * max(1.0, abs(w)):
            bad.append(f"{k}: {g!r} != {w!r}")
    return bad
