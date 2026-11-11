"""Path-integral welfare decomposition of ALTERTAX for shocks that are not small.

:mod:`decomp` is GEMPACK's WELVIEW in one step: every contribution is ``0.01 * COEF(pre) *
q%`` with ``pre`` the check. GEMPACK integrates the same linear contributions along its
multistep path, with the coefficients updated at every step, so the one step is exact
only to first order in the shock: on the 10 % RunGTAP references the import-tax totals
come out ~4.8 % low (the tax base grows ~10 % along the path, the midpoint is half of it).

Controller ruling (2026-11-09): the shock is solved in K equal sub-steps; each step's
contributions use the previous step as ``pre`` (:func:`decomp.changes_between`), and the
decomposition is their sum ``D(K)``. The totals converge first-order in 1/K (nus333 /
10x7: 17 %, 8.7 %, 4.5 %, 2.3 %, 1.2 % at K = 1 .. 16), so the reported decomposition is
the Richardson extrapolation ``R(K) = 2 D(K) - D(K/2)`` of every header (exact for
``D(K) = a + b / K``). K doubles from 1 until every gated total (the EV and the regional
totals the decomp gate checks, same relative metric) moves by less than
``PATH_TOL = TOL_CHANGE / 10`` between ``R(K/2)`` and ``R(K)`` (:func:`pick_steps`, first
possible stop K = 4). A fixed ``steps=K`` reports the plain ``D(K)``.

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

Sums. The CNT headers are linear change variables of GEMPACK (``Variable (change)``):
summed over the steps. The ``dvol`` columns (A231's slot too) are not: GEMPACK computes
them once after the solve as ``BASE * cumulative %`` with BASE the initial coefficient
(``BVDPB * qpd / 100`` ...), explanatory only, so they are the one-step value between the
check and the last point (:func:`one_step_dvol`), the path integral ``V(t) dq`` being a
different quantity (3x3 A22P ROW: GEMPACK -365.93, one step -365.94, path -384.35). ``taxrateb`` is the rate at the first
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
from equilibria.family.altertax.gates import NOISE, _regional_totals
from equilibria.family.altertax.tolerances import ABS_FLOOR, TOL_CHANGE
from equilibria.family.altertax.updated import _elements, _Period, headers_at

PATH_TOL = (
    TOL_CHANGE / 10
)  # change of any gated total between R(K/2) and R(K) that stops the doubling
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


def gated_totals(headers: Mapping[str, HeaderArray]) -> dict[tuple, float]:
    """``{(header, part, region): total}`` of what the decomp gate checks: the regional
    totals of every header but the exogenous ``C*`` ones (the ``A`` totals are the EVs)."""
    out: dict[tuple, float] = {}
    for name in DECOMP_HEADERS:
        if not name.startswith("C"):
            out |= _regional_totals(name, headers[name])
    return out


def noise_floor(headers: Mapping[str, HeaderArray]) -> float:
    """The gate's noise floor, ``NOISE * max|EV|`` (largest |component| of a WELFARE row)."""
    return NOISE * float(np.abs(np.asarray(headers["A"].array)).max())


def totals_change(
    new: Mapping[Any, float], old: Mapping[Any, float], floor: float = 0.0
) -> float:
    """Largest ``rel_diff`` (against ``new``, the gate's metric) over the totals that
    are above ``floor`` on either side."""
    return max(
        (
            rel_diff(float(v), float(old[k]), float(v), ABS_FLOOR)
            for k, v in new.items()
            if max(abs(v), abs(old[k])) > floor
        ),
        default=0.0,
    )


def richardson(
    fine: Mapping[str, HeaderArray], coarse: Mapping[str, HeaderArray]
) -> dict[str, HeaderArray]:
    """``2 D(K) - D(K/2)`` of every header: cancels the 1/K term of the path error."""
    return {
        n: dataclasses.replace(
            h,
            array=2.0 * np.asarray(h.array, dtype=np.float64)
            - np.asarray(coarse[n].array, dtype=np.float64),
        )
        for n, h in fine.items()
    }


def pick_steps(
    totals_at: Callable[[int], Mapping[Any, float]],
    tol: float = PATH_TOL,
    k_max: int = K_MAX,
    floor_at: Callable[[int], float] | None = None,
) -> tuple[int, dict[int, dict[str, Any]]]:
    """Double K from 1 until :func:`totals_change` between ``R(K/2)`` and ``R(K)`` is
    below ``tol``, with ``R(K) = 2 D(K) - D(K/2)`` and ``D(K) = totals_at(K)`` (a mapping
    of every gated total; ``floor_at(K)`` is the noise floor, default 0). Returns that K
    (the finer of the pair) and ``{K: {"change"}}`` for every K tried (``change`` is None
    until two R exist, K < 4). ``RuntimeError`` past ``k_max``."""
    hist: dict[int, dict[str, Any]] = {}
    k, d_prev, r_prev = 1, None, None
    while True:
        d = dict(totals_at(k))
        r = None if d_prev is None else {x: 2.0 * v - d_prev[x] for x, v in d.items()}
        ch = None
        if r is not None and r_prev is not None:
            ch = totals_change(r, r_prev, 0.0 if floor_at is None else floor_at(k))
        hist[k] = {"change": ch}
        if ch is not None and ch < tol:
            return k, hist
        if k >= k_max:
            raise RuntimeError(
                f"path decomposition: totals did not settle within {k_max} sub-steps "
                f"(last change {ch if ch is None else format(ch, '.3g')} >= {tol:.3g})"
            )
        d_prev, r_prev, k = d, r, 2 * k


def sum_steps(steps: list[dict[str, HeaderArray]]) -> dict[str, HeaderArray]:
    """The decomposition of a path from the decompositions of its steps (module
    docstring, "Sums")."""
    out: dict[str, np.ndarray] = {}
    for name in DECOMP_HEADERS:
        arrs = [np.asarray(s[name].array, dtype=np.float64) for s in steps]
        tot = np.sum(arrs, axis=0)
        names = list(steps[0][name].set_names)
        if "COL" in names:  # the last axis, or the one before DIREC (A231)
            ax = names.index("COL")
            col, first, last = (np.moveaxis(a, ax, 0) for a in (tot, arrs[0], arrs[-1]))
            col[2] = first[2]  # taxrateb: the rate at the first pre
            col[3] = last[3]  # taxrateu: the rate at the last post
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


def one_step_dvol(
    path: Mapping[str, HeaderArray], one: Mapping[str, HeaderArray]
) -> dict[str, HeaderArray]:
    """``path`` with the ``dvol`` column of every COL header and of A231 taken from
    ``one`` (module docstring, "Sums")."""
    out = dict(path)
    for name, h in path.items():
        if h.set_names[-1:] == ["COL"]:
            ix = (..., 1)
        elif name == "A231":
            ix = (..., 1, slice(None))  # COL is the axis before DIREC
        else:
            continue
        arr = np.array(h.array, dtype=np.float64)
        arr[ix] = np.asarray(one[name].array, dtype=np.float64)[ix]
        out[name] = dataclasses.replace(h, array=arr)
    return out


@dataclass
class DecompPath:
    """The path decomposition of a run: ``headers`` (as :func:`decomp.decomp_headers`)
    summed over ``steps`` sub-steps, Richardson-extrapolated with the ``steps // 2``
    decomposition when ``richardson``; ``history`` the EV per K tried; ``points`` the
    shock arrays at fractions ``1/steps .. 1`` (the last is the run's own)."""

    steps: int
    headers: dict[str, HeaderArray]
    history: dict[int, dict[str, Any]]
    points: list[Any] = field(repr=False)
    solves: int = 0  # sub-step shock solves (the run's own shock not counted)
    minutes: float = 0.0
    richardson: bool = False

    @property
    def steps_half(self) -> int | None:
        """K of the coarser decomposition of the extrapolation, None without it."""
        return self.steps // 2 if self.richardson else None

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "steps_half": self.steps_half,
            "richardson": self.richardson,
            "path_tol": PATH_TOL,
            "solves": self.solves,
            "minutes": self.minutes,
            "ev_by_steps": {str(k): v["ev"] for k, v in self.history.items()},
            "ev_richardson_by_steps": {
                str(k): v.get("ev_richardson") for k, v in self.history.items()
            },
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
    :func:`pick_steps` and the Richardson extrapolation (a fixed K reports the plain
    ``D(K)``)."""
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
            (A, va), (B, vb) = at(Fraction(0), k), at(Fraction(1), k)
            one = decomp_from_changes(va, vb, changes_between(A, B, els), els)
            decomps[k] = one_step_dvol(sum_steps(parts), one)
        return decomps[k]

    def ev_of(h: Mapping[str, HeaderArray]) -> list[float]:
        return [
            float(x) for x in np.asarray(h["A"].array, dtype=np.float64).sum(axis=1)
        ]

    if steps == "auto":
        k, hist = pick_steps(
            lambda j: gated_totals(headers_for(j)),
            tol,
            k_max,
            lambda j: noise_floor(headers_for(j)),
        )
        for j, rec in hist.items():
            rec["ev"] = ev_of(decomps[j])
            if j >= 2:
                rec["ev_richardson"] = ev_of(richardson(decomps[j], decomps[j // 2]))
        reported, extrapolated = richardson(decomps[k], decomps[k // 2]), True
    else:
        k = int(steps)
        if k < 1:
            raise ValueError(f"decomp_steps={steps!r}: expected 'auto' or K >= 1")
        hist = {k: {"ev": ev_of(headers_for(k)), "change": None}}
        reported, extrapolated = decomps[k], False
    return DecompPath(
        steps=k,
        headers=reported,
        history=hist,
        points=[pts[Fraction(j, k)] for j in range(1, k + 1)],
        solves=solves,
        minutes=(time.perf_counter() - t0) / 60.0,
        richardson=extrapolated,
    )
