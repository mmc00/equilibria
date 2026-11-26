"""Ways to solve the shock period (ADR-0011 solve strategies), over PeriodModel.

* ``in_steps``: the shocked instruments go from check to shock in fractions, each
  solve starting from the previous one, as GEMPACK's Gragg path.
* ``johansen``: one linear step from check, the first-order response, as
  GEMPACK's Johansen method.

Both move only the exogenous cells whose shock level differs from the check.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from typing import Any

from equilibria.core.period_model import Index, PeriodModel
from equilibria.core.recipe import Solver, Strategy
from equilibria.templates.gtap.period_rules import JOHANSEN_STEP

_log = logging.getLogger(__name__)

Shocked = tuple[str, Index, float, float]


def shocked_cells(
    model: PeriodModel, *, shock: str = "shock", check: str = "check"
) -> list[Shocked]:
    """``(name, index, check level, shock level)`` of each exogenous cell whose
    shock level differs from its check level, names in order."""
    exo = model.exogenous()
    out: list[Shocked] = []
    for name in sorted({n for n, _ in exo}):
        at_shock = model.values(name, shock)
        at_check = model.values(name, check)
        for idx in model.var_cells(name, shock):
            if (name, idx) not in exo or idx not in at_check:
                continue
            if at_shock[idx] != at_check[idx]:
                out.append((name, idx, at_check[idx], at_shock[idx]))
    return out


def _require(cells: list[Shocked]) -> list[Shocked]:
    if not cells:
        raise ValueError("no shocked cell: nothing to solve in steps")
    return cells


def _fix_at(model: PeriodModel, cells: list[Shocked], level: Callable) -> None:
    for name, idx, chk, full in cells:
        model.fix_cells(name, "shock", {idx: level(chk, full)})


def _full(chk: float, full: float) -> float:
    return full


def in_steps(lambdas: Sequence[float]) -> Strategy:
    """Solve the shock in steps ``lambdas`` (fractions; 1.0 is appended when
    missing). At step ``λ`` a shocked cell is ``check·(shock/check)**λ`` (the
    compounded % change), or ``check + λ·(shock − check)`` when either is not
    positive. Stops at the first step that does not converge, with the cells
    back at the full shock, and returns its status."""
    steps = [float(x) for x in lambdas]
    if not steps or steps[-1] != 1.0:
        steps.append(1.0)

    def strategy(model: PeriodModel, solver: Solver) -> str:
        cells = _require(shocked_cells(model))
        status = ""
        for lam in steps:

            def at(chk: float, full: float, lam: float = lam) -> float:
                if lam == 1.0:
                    return full
                if chk > 0.0 and full > 0.0:
                    return chk * (full / chk) ** lam
                return chk + lam * (full - chk)

            _fix_at(model, cells, at)
            status = solver(model)
            _log.info("[shock steps] λ=%.3f status=%s", lam, status)
            if status != "optimal":
                _fix_at(model, cells, _full)
                break
        return status

    return strategy


def _richardson(f: Sequence[float], h: float) -> float:
    """d/dλ at 0 from f(h), f(−h), f(2h), f(−2h): central differences, O(h⁴)."""
    return (8.0 * (f[0] - f[1]) - (f[2] - f[3])) / (12.0 * h)


def johansen(after: Callable[[Any], None] | None = None) -> Strategy:
    """The shock as one linear step from check (GEMPACK's Johansen method).

    The slope ``d x / d λ`` at λ = 0 (shocked cells at ``check + λ·(shock −
    check)``) comes from four solves at λ = ±h, ±2h (h = ``JOHANSEN_STEP``),
    each from the same start, combined by Richardson. ``after(model)`` runs
    after each solve (the period's recomputations), so those cells get their
    slope too. A positive cell ends at ``check × (1 + slope of its log)``, any
    other at ``check + slope``; the shocked cells end at the shock. The point
    does not satisfy the nonlinear equations: that is the method.
    """
    h = JOHANSEN_STEP

    def strategy(model: PeriodModel, solver: Solver) -> str:
        cells = _require(shocked_cells(model))
        shocked = {(name, idx) for name, idx, _, _ in cells}
        free = {
            name: [
                k for k in model.var_cells(name, "shock") if (name, k) not in shocked
            ]
            for name in model.var_names()
        }
        start = {name: model.values(name, "shock") for name in free}
        at: dict[float, dict[str, dict[Index, float]]] = {}
        try:
            for lam in (h, -h, 2 * h, -2 * h):
                for name, ks in free.items():
                    model.set_values(name, "shock", {k: start[name][k] for k in ks})
                _fix_at(model, cells, lambda c, f, lam=lam: c + lam * (f - c))
                status = solver(model)
                if status != "optimal":
                    return status
                if after is not None:
                    after(model)
                at[lam] = {name: model.values(name, "shock") for name in free}
        finally:
            _fix_at(model, cells, _full)
        n = 0
        for name, ks in free.items():
            check = model.values(name, "check")
            fixed = model.fixed_cells(name, "shock")
            out = {}
            for k in ks:
                if k not in check:
                    continue
                x0 = check[k]
                pts = [at[lam][name][k] for lam in (h, -h, 2 * h, -2 * h)]
                if any(math.isnan(y) for y in pts):
                    continue
                if k in fixed and all(y == start[name][k] for y in pts):
                    continue
                n += 1
                if x0 > 0.0 and min(pts) > 0.0:
                    out[k] = x0 * (1.0 + _richardson([math.log(y) for y in pts], h))
                else:
                    out[k] = x0 + _richardson(pts, h)
            model.set_values(name, "shock", out)
        _log.info(
            "[johansen] %d variables, one linear step (slope from λ = ±%g, ±%g)",
            n,
            h,
            2 * h,
        )
        return "optimal"

    strategy.linear = True  # type: ignore[attr-defined]  # read by equilibria.gtap.results
    return strategy
