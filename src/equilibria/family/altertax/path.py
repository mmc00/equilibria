"""Path-integral welfare decomposition of ALTERTAX for shocks that are not small.

:mod:`decomp` is GEMPACK's WELVIEW in one step: every contribution is ``0.01 * COEF(pre) *
q%`` with ``pre`` the check. GEMPACK integrates the same linear contributions along its
multistep path, with the coefficients updated at every step, so the one step is exact
only to first order in the shock: on the 10 % RunGTAP references the import-tax totals
come out ~4.8 % low (the tax base grows ~10 % along the path, the midpoint is half of it).

Controller ruling (2026-11-09): the shock is solved in K equal sub-steps; each step's
contributions use the previous step as ``pre`` (:func:`decomp.changes_between`), and the
decomposition is their sum. K doubles from 1 until the EV of every region moves by less
than ``PATH_TOL = TOL_CHANGE / 10`` between K/2 and K (:func:`pick_steps`).

Path. A shock is a percent change of the instrument's power (``imptx`` and ``prdtx_rai``
are GEMPACK kind ``power``); GEMPACK splits it into steps whose powers compound
(``(1 + p_t/100) = (1 + p/100)**t``), i.e. linear in the log of the power, which is what
its linearised percent-change variable integrates. :func:`shock_fraction` gives the shock
at fraction ``t`` of that path. (Measured: interpolating the percent arithmetically moves
the EV by ~1e-7 relative on nus333, so the choice is immaterial at TOL_CHANGE.)

Sub-steps. Point ``j/K`` is solved through the run's own :class:`FamilySession` (same
base, check and closure; under ``trade_balance="ratio"`` each point keeps ``savf / W`` at
the check's ratio), warm-started from point ``(j-1)/K``. Doubling K reuses every point of
K/2. The last point is the run's own shock (solved from the check), so the path ends on
exactly the solution the other outputs (updated, summary, volume, pct) are read from.

Sums. The CNT headers and the ``dvol`` columns are linear change variables of GEMPACK
(``Variable (change)``): summed over the steps. ``taxrateb`` is the rate at the first
``pre``, ``taxrateu`` at the last ``post``. E1's ``percent`` form (``c1_cr`` ...) and F1's
``pinv``/``psave`` are percent-change variables: compounded. E1's ``value`` form is
re-derived from the summed ``CNTtotr`` and the compounded percents (the ``postsim``
formula on cumulative results). F1's ``ISbal`` is the initial coefficient.

``EVSCALFACT`` (``[UTILELASEV / UTILELAS] * [INCOMEEV / INCOME]``) stays 1 at every step,
as in :mod:`decomp`: along a 10 % path it moves with the regional income; measured, the
converged EV agrees with GEMPACK's within 0.06 % (nus333 ROW) without it.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np

from equilibria.babel.har.symbols import HeaderArray
from equilibria.compare.measure import rel_diff
from equilibria.family.altertax.decomp import (
    DECOMP_HEADERS,
    _div,
    changes_between,
    decomp_from_changes,
)
from equilibria.family.altertax.tolerances import ABS_FLOOR, TOL_CHANGE
from equilibria.family.altertax.updated import _elements, _Period, headers_at

PATH_TOL = TOL_CHANGE / 10  # EV change between K/2 and K that stops the doubling
K_MAX = 64  # doubling stops with an error past this many sub-steps


def shock_fraction(
    shock: Mapping[str, Mapping[tuple, float]], t: float
) -> dict[str, dict[tuple, float]]:
    """The percent shock at fraction ``t`` (0 < t <= 1) of GEMPACK's path: the power
    compounds, ``1 + p_t/100 = (1 + p/100)**t``."""
    if not 0.0 < t <= 1.0:
        raise ValueError(f"shock fraction {t!r} outside (0, 1]")
    if t == 1.0:
        return {n: dict(cells) for n, cells in shock.items()}
    return {
        n: {c: 100.0 * ((1.0 + p / 100.0) ** t - 1.0) for c, p in cells.items()}
        for n, cells in shock.items()
    }


def ev_change(ev: np.ndarray, ev_prev: np.ndarray) -> float:
    """Largest relative change of a region's EV, ``rel_diff`` against the new value."""
    return max(
        rel_diff(float(a), float(b), float(a), ABS_FLOOR)
        for a, b in zip(ev, ev_prev, strict=True)
    )


def pick_steps(
    ev_at: Callable[[int], np.ndarray], tol: float = PATH_TOL, k_max: int = K_MAX
) -> tuple[int, dict[int, dict[str, Any]]]:
    """Double K from 1 until :func:`ev_change` of ``ev_at(K)`` against ``ev_at(K/2)``
    is below ``tol``. Returns that K (the finer of the pair) and ``{K: {"ev", "change"}}``
    for every K tried (``change`` is None at K = 1). ``RuntimeError`` past ``k_max``."""
    hist: dict[int, dict[str, Any]] = {}
    k, prev = 1, None
    while True:
        ev = np.asarray(ev_at(k), dtype=np.float64)
        ch = None if prev is None else ev_change(ev, prev)
        hist[k] = {"ev": [float(x) for x in ev], "change": ch}
        if ch is not None and ch < tol:
            return k, hist
        if k >= k_max:
            raise RuntimeError(
                f"path decomposition: EV did not settle within {k_max} sub-steps "
                f"(last change {ch:.3g} >= {tol:.3g})"
            )
        prev, k = ev, 2 * k


def sum_steps(steps: list[dict[str, HeaderArray]]) -> dict[str, HeaderArray]:
    """The decomposition of a path from the decompositions of its steps (module
    docstring, "Sums")."""
    out: dict[str, np.ndarray] = {}
    for name in DECOMP_HEADERS:
        arrs = [np.asarray(s[name].array, dtype=np.float64) for s in steps]
        tot = np.sum(arrs, axis=0)
        if steps[0][name].set_names[-1:] == ["COL"]:
            tot[..., 2] = arrs[0][..., 2]  # taxrateb: the rate at the first pre
            tot[..., 3] = arrs[-1][..., 3]  # taxrateu: the rate at the last post
        out[name] = tot

    def compound(arrs: list[np.ndarray]) -> np.ndarray:
        return 100.0 * (np.prod([1.0 + a / 100.0 for a in arrs], axis=0) - 1.0)

    f1 = [np.asarray(s["F1"].array, dtype=np.float64) for s in steps]
    out["F1"] = np.stack(
        [f1[0][:, 0], compound([a[:, 1] for a in f1]), compound([a[:, 2] for a in f1])],
        axis=1,
    )
    pct = compound([np.asarray(s["E1"].array, dtype=np.float64)[..., 0] for s in steps])
    cnt_tot = out["A"][:, 4]
    den = pct.sum(axis=(0, 2))
    val = _div(
        cnt_tot[None, :, None] * pct, np.broadcast_to(den[None, :, None], pct.shape)
    )
    out["E1"] = np.stack([pct, val], axis=-1)
    return {n: dataclasses.replace(steps[0][n], array=out[n]) for n in DECOMP_HEADERS}


@dataclass
class DecompPath:
    """The path decomposition of a run: ``headers`` (as :func:`decomp.decomp_headers`)
    summed over ``steps`` sub-steps; ``history`` the EV per K tried; ``points`` the
    shock arrays at fractions ``1/steps .. 1`` (the last is the run's own)."""

    steps: int
    headers: dict[str, HeaderArray]
    history: dict[int, dict[str, Any]]
    points: list[Any] = field(repr=False)
    solves: int = 0  # sub-step shock solves (the run's own shock not counted)
    minutes: float = 0.0

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "path_tol": PATH_TOL,
            "solves": self.solves,
            "minutes": self.minutes,
            "ev_by_steps": {str(k): v["ev"] for k, v in self.history.items()},
            "change_by_steps": {str(k): v["change"] for k, v in self.history.items()},
        }


def solve_decomp_path(
    run: Any,
    shock: Mapping[str, Mapping[tuple, float]],
    steps: int | str = "auto",
    *,
    tol: float = PATH_TOL,
    k_max: int = K_MAX,
) -> DecompPath:
    """The path decomposition of ``run`` (an ``AltertaxRun`` solved with
    ``keep_session``) for its ``shock``: ``steps`` sub-steps, or ``"auto"`` for
    :func:`pick_steps`."""
    session = run.result.session
    if session is None:
        raise ValueError("solve_decomp_path needs a run solved with keep_session=True")
    t0 = time.perf_counter()
    els = _elements(run)
    pts: dict[Fraction, Any] = {Fraction(1): run.result.mp}
    cache: dict[Fraction, tuple[_Period, dict[str, np.ndarray]]] = {}
    solves = 0

    def point(f: Fraction, k: int):
        nonlocal solves
        if f not in pts:
            prev = f - Fraction(1, k)
            start = pts[prev] if prev > 0 else None
            mp, res = session.solve_shock(shock_fraction(shock, float(f)), start=start)
            if int(res["code"]) != 1:
                raise RuntimeError(
                    f"path decomposition: sub-step {f} of the shock did not converge "
                    f"(code {res['code']}, residual {res.get('residual')})"
                )
            pts[f] = mp
            solves += 1
        return pts[f]

    def at(f: Fraction, k: int) -> tuple[_Period, dict[str, np.ndarray]]:
        if f not in cache:
            if f == 0:
                r, per = run, "check"
            else:
                r = dataclasses.replace(
                    run, result=dataclasses.replace(run.result, mp=point(f, k))
                )
                per = "shock"
            heads = {
                n: np.asarray(h.array, dtype=np.float64)
                for n, h in headers_at(r, per).items()
            }
            cache[f] = (_Period(r, per), heads)
        return cache[f]

    decomps: dict[int, dict[str, HeaderArray]] = {}

    def headers_for(k: int) -> dict[str, HeaderArray]:
        if k not in decomps:
            for j in range(1, k):  # in order: each point warm-starts the next
                point(Fraction(j, k), k)
            parts = []
            for j in range(k):
                (A, va), (B, vb) = at(Fraction(j, k), k), at(Fraction(j + 1, k), k)
                parts.append(
                    decomp_from_changes(va, vb, changes_between(A, B, els), els)
                )
            decomps[k] = sum_steps(parts)
        return decomps[k]

    def ev_at(k: int) -> np.ndarray:
        return np.asarray(headers_for(k)["A"].array, dtype=np.float64).sum(axis=1)

    if steps == "auto":
        k, hist = pick_steps(ev_at, tol, k_max)
    else:
        k = int(steps)
        if k < 1:
            raise ValueError(f"decomp_steps={steps!r}: expected 'auto' or K >= 1")
        hist = {k: {"ev": [float(x) for x in ev_at(k)], "change": None}}
    return DecompPath(
        steps=k,
        headers=decomps[k],
        history=hist,
        points=[pts[Fraction(j, k)] for j in range(1, k + 1)],
        solves=solves,
        minutes=(time.perf_counter() - t0) / 60.0,
    )
