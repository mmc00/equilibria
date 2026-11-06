"""The period recipe of the multiperiod driver (gtap mode), on arrays.

``PeriodPreparer.prepare`` runs an ordered list of steps over the Pyomo
``ConcreteModel`` before each solve. Here the same steps run over ``MPArrays`` (the
Var cells' value / fixed / lb / ub and the Constraint cells' active flag) as numpy
masks, with no model and no expressions:

* ``("check", "gtap")``: freeze the other periods, seed (only with ``seed_from_prior``),
  unfix ``regy``, collapse the ``pft`` block, copy the single-period reference's fixing
  and bounds, free what GAMS does not bound, mute the welfare tail, holdfix the
  sector-specific ``pf``.
* ``("shock", "gtap")``: the same, always seeded from the check, with the numeraire
  anchor, and the default +10 % tariff the driver writes right after ``prepare`` when the
  model carries no shock.

``sp_reference_arrays`` is the single-period reference (what the driver reads from a
throwaway Pyomo model) as arrays, from the same composer steps on a ``StateModel``.

The Fisher aggregates ``mq_*`` are refreshed after the seed from the sums over the
periods when ``run_recipe`` gets the cross-period recording (``cross=``); without it
they keep the value they had (their fixed / bounds are right).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from equilibria.family.mp_arrays import MPArrays, RowArr, VarArr

PERIODS = ("base", "check", "shock")
RECIPE_PERIODS = ("check", "shock")
_UID = itertools.count()


# ── The single-period reference as arrays ───────────────────────────────────


@dataclass
class SPVar:
    """One reference variable: every cell (key without period), flat in set order."""

    keys: list[tuple]
    value: np.ndarray  # a fixed cell's value (NaN -> 0.0 as in SPSnapshot)
    fixed: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    uid: int = field(default_factory=lambda: next(_UID))


@dataclass
class SPRef:
    vars: dict[str, SPVar]
    exogenous: frozenset[str]  # registered ShockBlock instruments
    endogenous: dict[str, frozenset]  # instrument cells an @overwrite hook freed


def sp_reference_arrays(params, closure, residual_region, *, state=None) -> SPRef:
    """The reference ``PeriodPreparer`` reads, from ``compose_block_state``.

    Same cells as ``_sp_snapshot(_build_sp_reference(...))``: the Fisher auxiliaries
    the driver deletes from the reference are left out. ``state`` is the composed state
    when the caller already has it (``MPArrays.build`` reads the same one), so the
    composition runs once; it is only read."""
    from equilibria.blocks.gtap.closure import FISHER_AUX_VARS

    if state is None:
        from equilibria.templates.gtap.gtap_block_model import compose_block_state

        _, state = compose_block_state(params, params.sets, closure, residual_region)
    out = {}
    for name, doms in state.var_domains.items():
        if name in FISHER_AUX_VARS:
            continue
        keys = list(itertools.product(*(state.sets[d] for d in doms)))
        value = np.asarray(state.values[name], dtype=float).ravel().copy()
        out[name] = SPVar(
            keys,
            np.where(np.isnan(value), 0.0, value),
            np.asarray(state.fixed[name], dtype=bool).ravel().copy(),
            np.asarray(state.lower[name], dtype=float).ravel().copy(),
            np.asarray(state.upper[name], dtype=float).ravel().copy(),
        )
    endo = getattr(state, "_endogenous_instrument_cells", None) or {}
    return SPRef(
        out,
        frozenset(getattr(state, "_exogenous_instruments", ())),
        {k: frozenset(v) for k, v in endo.items()},
    )


# ── Lookups (cached on the arrays: keys never change) ───────────────────────


def _table(obj: VarArr | RowArr, period: str) -> dict[tuple, int]:
    """Position of each cell of ``period``, by its key without the period."""
    cache = obj.__dict__.setdefault("_table", {})
    if period not in cache:
        sel = np.flatnonzero(obj.period == period)
        cache[period] = {obj.keys[i][:-1]: int(i) for i in sel}
    return cache[period]


def _aligned(obj: VarArr, a: str, b: str) -> tuple[np.ndarray, np.ndarray]:
    """Positions of the cells of period ``a`` that have a counterpart in ``b``."""
    cache = obj.__dict__.setdefault("_aligned", {})
    if (a, b) not in cache:
        tb = _table(obj, b)
        ia, ib = [], []
        for k, i in _table(obj, a).items():
            j = tb.get(k)
            if j is not None:
                ia.append(i)
                ib.append(j)
        cache[a, b] = (np.array(ia, dtype=int), np.array(ib, dtype=int))
    return cache[a, b]


def _sp_positions(v: VarArr, sp: SPVar, period: str) -> np.ndarray:
    """Position in ``v`` (period ``period``) of each reference cell, -1 if absent."""
    cache = v.__dict__.setdefault("_sp_pos", {})
    if (sp.uid, period) not in cache:
        tab = _table(v, period)
        cache[sp.uid, period] = np.array([tab.get(k, -1) for k in sp.keys], dtype=int)
    return cache[sp.uid, period]


@dataclass
class _Ctx:
    mp: MPArrays
    period: str
    params: Any
    closure: Any
    sp: SPRef
    mode: str
    mute_welfare: bool
    seed_from_prior: bool
    seed_prior: str
    cross: Any = None  # cross-period recording: refresh the mq_* after a seed

    def exo(self, name: str, v: VarArr) -> np.ndarray:
        """Cells of ``name`` that are exogenous instruments (the driver never frees,
        re-seeds or re-bounds them)."""
        from equilibria.templates.gtap.instruments import is_free_cell

        if name not in self.sp.exogenous:
            return np.zeros(len(v.keys), dtype=bool)
        libres = self.sp.endogenous.get(name)
        if not libres:
            return np.ones(len(v.keys), dtype=bool)
        return np.array([not is_free_cell(k, libres) for k in v.keys])


# ── Steps ───────────────────────────────────────────────────────────────────


def _freeze_inactive(c: _Ctx) -> None:
    """freeze_inactive_periods: other periods' Vars fixed and rows off; this period's
    Vars free and rows on. Instruments are left as they are."""
    for name, v in c.mp.vars.items():
        live = ~c.exo(name, v)
        active = v.period == c.period
        v.fixed[active & live] = False
        other = ~active & live & ~v.fixed
        v.value[other & np.isnan(v.value)] = 1.0
        v.fixed[other] = True
    for r in c.mp.rows.values():
        timed = (r.period == "base") | (r.period == "check") | (r.period == "shock")
        r.active[timed] = r.period[timed] == c.period


def _seed_from_prior(c: _Ctx) -> None:
    """Warm start: this period's free Vars take the prior period's value; the Fisher
    aggregates (mq_*) are then recomputed from the sums they define (every period's,
    as ``refresh_fisher_aggregates`` does), when the cross-period recording is given."""
    for name, v in c.mp.vars.items():
        ia, ib = _aligned(v, c.period, c.seed_prior)
        take = ~c.exo(name, v)[ia] & ~v.fixed[ia] & ~np.isnan(v.value[ib])
        v.value[ia[take]] = v.value[ib[take]]
    if c.cross is not None:
        from equilibria.family.cross_period import refresh_aggregates

        refresh_aggregates(c.cross, c.mp)


def _seed_from_prior_if_asked(c: _Ctx) -> None:
    if c.seed_from_prior:
        _seed_from_prior(c)


def _unfix_regy(c: _Ctx) -> None:
    regy = c.mp.var("regy")
    tab = _table(regy, c.period)
    for r in c.params.sets.r:
        regy.fixed[tab[(str(r),)]] = False


def _fix_shock_numeraire(c: _Ctx) -> None:
    from equilibria.templates.gtap.period_prep import _SHOCK_PNUM

    pnum = c.mp.var("pnum")
    i = _table(pnum, c.period)[()]
    if not pnum.fixed[i]:
        pnum.value[i] = _SHOCK_PNUM
        pnum.fixed[i] = True


def _collapse_pft(c: _Ctx) -> None:
    """Factor-price block (gtap): real factors drop eq_pfteq; dangling ones (no
    eq_xfteq) fix xft and pft, holdfix xf and drop its eq_pfeq."""
    mp, t = c.mp, c.period
    pft, xft, xf = mp.var("pft"), mp.var("xft"), mp.var("xf")
    xfteq, pfteq, pfeq = mp.row("eq_xfteq"), mp.row("eq_pfteq"), mp.row("eq_pfeq")
    xfteq_t, pfteq_t, pfeq_t = (_table(r, t) for r in (xfteq, pfteq, pfeq))
    xft_t, xf_t = _table(xft, t), _table(xf, t)
    by_rf: dict[tuple, list[int]] = {}
    for k in xf_t:
        by_rf.setdefault(k[:2], []).append(xf_t[k])
    for rf, ip in _table(pft, t).items():
        j = xfteq_t.get(rf)
        if j is not None and xfteq.active[j]:
            k = pfteq_t.get(rf)
            if k is not None:
                pfteq.active[k] = False
            continue
        i = xft_t.get(rf)
        if i is not None and not xft.fixed[i]:
            if np.isnan(xft.value[i]):
                xft.value[i] = 0.0
            xft.fixed[i] = True
        if not pft.fixed[ip]:
            pft.value[ip] = 1.0
            pft.fixed[ip] = True
        for ix in by_rf.get(rf, ()):
            val = 0.0 if np.isnan(xf.value[ix]) else xf.value[ix]
            if abs(val) < 1e-12:
                continue
            xf.fixed[ix] = True
            a = xf.keys[ix][2]
            k = pfeq_t.get((*rf, a))
            if k is not None:
                pfeq.active[k] = False


def _replicate_sp_reference(c: _Ctx) -> None:
    """Fix what the single-period model fixes (not over cells already fixed) and copy
    its bounds."""
    for name, sp in c.sp.vars.items():
        has_bound = np.isfinite(sp.lb) | np.isfinite(sp.ub)
        if not (sp.fixed.any() or has_bound.any()):
            continue
        v = c.mp.var(name)
        pos = _sp_positions(v, sp, c.period)
        here = pos >= 0
        todo = sp.fixed & here
        todo[todo] = ~v.fixed[pos[todo]]
        v.value[pos[todo]] = sp.value[todo]
        v.fixed[pos[todo]] = True
        lo = np.isfinite(sp.lb) & here
        v.lb[pos[lo]] = sp.lb[lo]
        hi = np.isfinite(sp.ub) & here
        v.ub[pos[hi]] = sp.ub[hi]


def _gams_bounds(c: _Ctx) -> None:
    from equilibria.templates.gtap.gtap_multiperiod_driver import GAMS_BOUNDED_VARS

    for name, v in c.mp.vars.items():
        if name in GAMS_BOUNDED_VARS:
            continue
        cells = (v.period == c.period) & ~c.exo(name, v)
        v.lb[cells] = -np.inf
        v.ub[cells] = np.inf


def _mute_welfare_if_asked(c: _Ctx) -> None:
    """Fix the inert welfare leaves and drop their rows; walras stays free in gtap."""
    if not c.mute_welfare:
        return
    from equilibria.templates.gtap.gtap_multiperiod_driver import _WELFARE_LEAVES

    for eqn, vn in _WELFARE_LEAVES:
        eq, v = c.mp.row(eqn), c.mp.var(vn)
        vt, et = _table(v, c.period), _table(eq, c.period)
        keep_row = c.mode == "gtap" and vn == "walras"
        for r in c.params.sets.r:
            for cand in ((str(r),), ()):
                i = vt.get(cand)
                if i is None:
                    continue
                if keep_row:
                    v.fixed[i] = False
                    break
                if not v.fixed[i] and not np.isnan(v.value[i]):
                    v.fixed[i] = True
                j = et.get(cand)
                if j is not None:
                    eq.active[j] = False
                break


def _fnm_pf(c: _Ctx) -> None:
    """Holdfix pf of sector-specific factors with etaff=0 and no factor flow."""
    p = c.params
    fset = getattr(p, "sets", None)
    if fset is None:
        return
    pf = c.mp.var("pf")
    mf = {str(f) for f in getattr(fset, "mf", [])}
    sf = {str(f) for f in getattr(fset, "sf", [])}
    etaff = getattr(getattr(p, "elasticities", None), "etaff", {}) or {}
    bm = getattr(p, "benchmark", None)

    def has_flow(r, f, a) -> bool:
        if bm is None:
            return True
        v = bm.evfb.get((r, f, a), bm.vfm.get((r, f, a), 0.0))
        return abs(float(v or 0.0)) > 1e-12

    for key, i in _table(pf, c.period).items():
        r, f, a = key
        if f in mf or f in sf:
            continue
        if float(etaff.get(key, 0.0) or 0.0) != 0.0 or has_flow(r, f, a):
            continue
        if not np.isnan(pf.value[i]) and not pf.fixed[i]:
            pf.fixed[i] = True


def _default_tariff(c: _Ctx) -> None:
    """The driver writes the +10 % default tariff into imptx[.., shock] when no
    instrument differs from its check value (``shocks.apply_default_tariff``)."""
    from equilibria.templates.gtap.shocks import DEFAULT_TARIFF_PCT, shocked

    mp, sp = c.mp, c.sp
    for name in sorted(sp.exogenous):  # shock_of: any shocked cell -> nothing to do
        v = mp.var(name)
        ia, ib = _aligned(v, "shock", "check")
        if (c.exo(name, v)[ia] & (v.value[ia] != v.value[ib])).any():
            return
    imptx = mp.var("imptx")
    if "imptx" not in sp.exogenous:
        raise ValueError("no imptx cell can take the default tariff")
    ia, ib = _aligned(imptx, "shock", "check")
    exo = c.exo("imptx", imptx)[ia]
    # a live row must read the cell: eq_pmeq, or under ifSUB the inline macro in
    # eq_xweq (assumed to read imptx wherever it is active)
    live = np.zeros(len(ia), dtype=bool)
    rows = ["eq_pmeq"] + (["eq_xweq"] if getattr(c.closure, "if_sub", False) else [])
    for name in rows:
        if name not in mp.rows:
            continue
        r = mp.rows[name]
        tab = _table(r, "shock")
        pos = np.array([tab.get(imptx.keys[i][:-1], -1) for i in ia], dtype=int)
        live |= (pos >= 0) & r.active[np.maximum(pos, 0)]
    take = exo & live
    if not take.any():
        raise ValueError("no imptx cell can take the default tariff")
    chk = imptx.value[ib[take]]
    new = shocked("power", chk, 1.0 + DEFAULT_TARIFF_PCT / 100.0)
    if (new == chk).any() or not (1.0 + new > 0.0).all():
        raise ValueError("the default tariff leaves the domain of imptx")
    imptx.value[ia[take]] = new
    imptx.fixed[ia[take]] = True


_RECIPES = {
    ("check", "gtap"): (
        _freeze_inactive,
        _seed_from_prior_if_asked,
        _unfix_regy,
        _collapse_pft,
        _replicate_sp_reference,
        _gams_bounds,
        _mute_welfare_if_asked,
        _fnm_pf,
    ),
    ("shock", "gtap"): (
        _freeze_inactive,
        _seed_from_prior,
        _unfix_regy,
        _fix_shock_numeraire,
        _collapse_pft,
        _replicate_sp_reference,
        _gams_bounds,
        _mute_welfare_if_asked,
        _fnm_pf,
        _default_tariff,
    ),
}


def seed_prior(period: str, *, base_calibrated: bool, solve_check: bool) -> str:
    """The period ``period`` is seeded from, as ``PeriodPreparer.seed_prior``: the
    shock from the check, unless F3.5 copied the check from the base."""
    if period == "shock":
        return "base" if base_calibrated and not solve_check else "check"
    return "base"


def run_recipe(
    mp: MPArrays,
    period: str,
    *,
    params,
    closure,
    sp_ref: SPRef,
    mode: str = "gtap",
    mute_welfare: bool,
    seed_from_prior: bool,
    base_calibrated: bool,
    solve_check: bool,
    cross: Any = None,
) -> MPArrays:
    """``mp`` after the recipe of ``period`` (check or shock; gtap), as new arrays.

    With ``cross`` (the cross-period recording) the ``mq_*`` aggregates are refreshed
    from the sums they define where the Pyomo recipe does (the seed from the prior
    period); without it they keep their value (see the module docstring).

    The driver flags (``PeriodPreparer``'s) have no default: they pick the recipe and
    the period the seed copies from. A step whose component is missing raises
    ``KeyError``; the F3.5 copied check (``base_calibrated`` without
    ``solve_check``) raises ``NotImplementedError``."""
    if period == "check" and base_calibrated and not solve_check:
        raise NotImplementedError("no array recipe for the check copied from the base")
    steps = _RECIPES.get((period, mode))
    if steps is None:
        raise NotImplementedError(
            f"no array recipe for period {period!r}, mode {mode!r}"
        )
    ctx = _Ctx(
        mp.copy(),
        period,
        params,
        closure,
        sp_ref,
        mode,
        mute_welfare,
        seed_from_prior,
        seed_prior(period, base_calibrated=base_calibrated, solve_check=solve_check),
        cross,
    )
    for step in steps:
        step(ctx)
    return ctx.mp
