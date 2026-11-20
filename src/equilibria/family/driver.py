"""Family backend driver: the multiperiod GTAP run on arrays, without a Pyomo model.

Opt-in (``_solve_shock(..., backend="family")`` or ``EQUILIBRIA_GTAP_BACKEND=family``).
The route never falls back to Pyomo: a configuration it cannot run raises
``NotImplementedError`` naming it, before any work is done.

User shocks are not refused here: every ShockBlock instrument is applied through the
same ``apply_shock`` semantics as the Pyomo route.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from equilibria.family.capflex import require_ref_gdx
from equilibria.family.mp_arrays import MPArrays, Squared
from equilibria.family.period_newton import (  # noqa: F401 - re-exported
    _MUMPS_KEEP,
    _NEWTON_TOL,
    CODE_RESIDUAL,
    ICNTL6_FAMILY,
    RatioBorder,
    _newton,
    _period_system,
    _world_income,
)
from equilibria.family.phases import PhaseTimer, activate, phase, scope
from equilibria.family.run_setup import (  # noqa: F401 - re-exported
    _SETTLE_KEY_ENV_SKIP,
    _UNSUPPORTED_ENV,
    CACHE_ENV,
    SETTLE_CACHE_ENV,
    FamilyRun,
    _driver_closure,
    _settle_base_key,
    _settle_key,
    check_supported,
    default_cache_root,
    prepare_run,
    settled_check_point,
    solve_record,
)
from equilibria.solver.gc_tuning import (
    _gc_family_scope,
    _gc_freeze_from_env,
    _gc_freeze_now,
)

PERIODS = ("base", "check", "shock")


@dataclass
class FamilyResult:
    """What the array route returns: no Pyomo model, no fill.

    ``results[period]`` is ``{"code", "residual", "nit"}`` (the base is not solved:
    ``skipped_solve``); ``settled`` is the dict the calibration returned;
    ``settle`` the same stats for the settle's own calls; ``timing`` seconds per
    period; ``phases`` self seconds per fine phase (``settle_check.newton_J``, ...)."""

    mp: MPArrays
    results: dict[str, dict[str, Any]]
    settled: dict
    settle: dict[str, dict[str, Any]] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    phases: dict[str, float] = field(default_factory=dict)  # self seconds per phase
    phase_counts: dict[str, int] = field(default_factory=dict)
    # ``solve_gtap_family(keep_session=True)``: more shocks on the same run context
    session: Any = field(default=None, repr=False)
    # every Newton solve of the run (settle, check, shock, ratio iterations), in order:
    # ``{"solve", "code", "nit", "residual", "null_pivots", "gmin_retries", ...}``
    solves: list = field(default_factory=list)


def _solve_period(
    run: FamilyRun,
    mp: MPArrays,
    period: str,
    *,
    base_calibrated: bool,
    solve_check: bool,
    shock: Any = None,
    shock_levels: bool = False,
    tag: str | None = None,
    warm_start: bool = False,
    label: str | None = None,
) -> tuple[MPArrays, dict[str, Any]]:
    """One period of ``_solve_multiperiod_inner``: recipe, solve, post-solve.

    Returns the arrays the next period starts from (``mp`` itself is not modified).
    ``tag`` names the call in the phase timing (default: the period); ``label`` its
    record in ``run.solves`` (default: the tag). ``warm_start`` (shock only): the free
    cells restart from the shock values ``mp`` already holds (a previous shock solve)
    instead of the check's."""
    with scope(tag or period):
        return _solve_period_inner(
            run,
            mp,
            period,
            base_calibrated=base_calibrated,
            solve_check=solve_check,
            shock=shock,
            shock_levels=shock_levels,
            warm_start=warm_start,
            label=label or tag or period,
        )


def _solve_period_inner(
    run: FamilyRun,
    mp: MPArrays,
    period: str,
    *,
    base_calibrated: bool,
    solve_check: bool,
    shock: Any = None,
    shock_levels: bool = False,
    warm_start: bool = False,
    label: str = "",
) -> tuple[MPArrays, dict[str, Any]]:
    from equilibria.family.post_solve import post_solve

    mp1 = _period_recipe(
        run,
        mp,
        period,
        base_calibrated=base_calibrated,
        solve_check=solve_check,
        shock=shock,
        shock_levels=shock_levels,
        warm_start=warm_start,
    )
    if period == "base":  # skip_base_solve: the base is the calibrated benchmark
        with phase("post_solve"):
            post_solve(
                mp1,
                "base",
                backend=run.backend,
                state=run.state,
                params=run.params,
                regions=run.regions,
                if_sub=run.if_sub,
            )
        return mp1, {"code": 1, "residual": 0.0, "skipped_solve": True}
    sq, mp_sq, ss = _period_system(run, mp1, period)
    token, keep = _NEWTON_TOL.set(run.newton_tol), _MUMPS_KEEP.set(run.mumps)
    try:
        res, resid, code = _newton(ss, sq)
    finally:
        _NEWTON_TOL.reset(token)
        _MUMPS_KEEP.reset(keep)
    with phase("post_solve"):
        post_solve(
            mp_sq,
            period,
            backend=run.backend,
            state=run.state,
            params=run.params,
            regions=run.regions,
            if_sub=run.if_sub,
            sq=sq,
            x=res.x,
            code=code,
        )
    rec = solve_record(label or period, res, resid, code)
    run.solves.append(rec)
    return mp_sq, {
        "code": code,
        "residual": resid,
        "nit": res.nit,
        "solves": [rec],
    }


def _period_recipe(
    run: FamilyRun,
    mp: MPArrays,
    period: str,
    *,
    base_calibrated: bool,
    solve_check: bool,
    shock: Any = None,
    shock_levels: bool = False,
    warm_start: bool = False,
) -> MPArrays:
    """The recipe step of ``_solve_period_inner`` (new arrays)."""
    from equilibria.family.prep_recipe import run_recipe

    with phase("recipe"):
        return run_recipe(
            mp,
            period,
            params=run.params,
            closure=run.closure,
            sp_ref=run.ref,
            mode="gtap",
            backend=run.backend,
            mute_welfare=True,
            seed_from_prior=False,
            base_calibrated=base_calibrated,
            solve_check=solve_check,
            cross=run.cross,
            shock=shock if period == "shock" else None,
            shock_levels=shock_levels,
            nominal_savf=run.nominal_savf,
            savf_targets=run.savf_targets,
            warm=mp if warm_start else None,
        )


def calibrate_base_arrays(
    mp: MPArrays, run: FamilyRun
) -> tuple[dict[str, dict], dict[str, dict[str, Any]]]:
    """``FactorBlock.calibrate_base`` on arrays: the settle on a copy of ``mp``.

    The driver loop up to the check with ``settle_only`` (``base_calibrated=False``,
    ``solve_check=False``); the caller's ``mp`` is not touched. Returns ``(settled,
    results)``: the check slice minus the exogenous instruments, in the shape
    ``calibrate_base`` returns, and the per-period stats. No seed cache."""
    cur = mp.copy()
    results: dict[str, dict[str, Any]] = {}
    for period in ("base", "check"):
        cur, results[period] = _solve_period(
            run,
            cur,
            period,
            base_calibrated=False,
            solve_check=False,
            tag=f"settle_{period}",
        )
    return settled_check_point(cur), results


def calibrate_base_cached(
    mp: MPArrays, run: FamilyRun
) -> tuple[dict[str, dict], dict[str, dict[str, Any]]]:
    """:func:`calibrate_base_arrays`, through the opt-in disk cache of the settled point
    (``blocks/gtap/seed_cache`` storage). Only a settle whose every period converged
    is saved; ``EQUILIBRIA_SEED_CACHE_DISABLE=1`` (which the benchmark settings
    ``NEWTON_MUMPS`` set) turns it off like every seed cache. A hit returns the saved point and ``{"code": 1, "nit": 0, "cached":
    True}`` per period; it runs no settle solve, so the first solve of the run (the
    check) pays the one-time costs the settle paid."""
    from equilibria.blocks.gtap import seed_cache

    key = _settle_key(run)
    if key is not None:
        with phase("settle_cache_load"):
            cached = seed_cache.load(key)
        if cached is not None:
            stats = {"code": 1, "nit": 0, "cached": True}
            return cached, {p: dict(stats) for p in ("base", "check")}
    settled, settle = calibrate_base_arrays(mp, run)
    if key is not None and all(int(r.get("code", 0)) == 1 for r in settle.values()):
        seed_cache.save(key, settled)
    return settled, settle


def solve_gtap_family(
    params: Any,
    closure: Any,
    residual_region: str,
    *,
    ref_gdx: Any = None,
    shock: Any = None,
    shock_levels: bool = False,
    cache_root: Path | None = None,
    jax_cache: bool = True,
    nominal_trade_balance: bool | str = False,
    keep_session: bool = False,
    newton_tol: tuple[float, float] | None = None,
) -> FamilyResult:
    """Calibrated base, check and shock on arrays: ``_solve_shock``'s route without Pyomo.

    ``shock`` is a user shock ``{instrument: {cell: pct}}`` (levels with
    ``shock_levels``) for any ShockBlock instrument, applied as ``shocks.apply_shock``
    does; ``None`` leaves the default +10 % import tariff. ``ref_gdx`` is not read under
    capFix; under capFlex it gives ``risk`` (``family.capflex``) and is required (a
    missing file raises ``NotImplementedError`` before any work). capFlex also needs
    ``gdxdump`` on PATH and ``scripts/gtap`` (``_diff_core``) importable.

    ``nominal_trade_balance`` (default off, ALTERTAX's GEMPACK ``del_tbal = 0``): in the
    shock ``savf`` of every non-residual region is held at its check value and its
    ``eq_savf`` row is dropped, instead of ``savf = pigbl * savf_bar`` (real).
    ``"ratio"`` is GEMPACK's ``del_tbalry(r) = 0`` instead: the same hook, but ``savf[r]``
    keeps its ratio to world income, ``savf_shock = savf_check * W / W_check`` with
    ``W = sum(regy)``; ``W`` depends on the solution, so it is one more unknown of the
    shock's Newton system (:class:`RatioBorder`): one solve.

    ``newton_tol``: ``(reltol, ftol)`` for every Newton solve of the run (scaled and
    absolute residual), instead of ``EQUILIBRIA_GTAP_TR_RELTOL`` / ``_FTOL``.

    ``keep_session``: the result carries a :class:`FamilySession` (``result.session``)
    that solves the shock period again at other shocks on the same context."""
    if nominal_trade_balance not in (False, True, "nominal", "ratio"):
        raise ValueError(
            f"nominal_trade_balance={nominal_trade_balance!r}: expected False, True, "
            "'nominal' or 'ratio'"
        )
    check_supported(
        if_sub=bool(getattr(closure, "if_sub", False)),
        savf_flag=str(getattr(closure, "savf_flag", "capFix")),
        mode="gtap",
        skip_base_solve=True,
        mute_welfare=True,
        seed_from_prior=False,
        solve_check=True,
        shock_steps=None,
        shock_method="nonlinear",
    )
    if nominal_trade_balance and getattr(closure, "savf_flag", "capFix") != "capFix":
        raise NotImplementedError(
            "nominal_trade_balance replaces the capFix eq_savf row; savf_flag="
            f"{getattr(closure, 'savf_flag', None)!r} has none to replace"
        )
    if getattr(closure, "savf_flag", "capFix") == "capFlex":
        require_ref_gdx(ref_gdx)
    timing: dict[str, float] = {}
    timer = PhaseTimer()
    prev = activate(timer)
    try:
        # as ``solve_multiperiod``: freeze the live graph (EQUILIBRIA_GTAP_GC_FREEZE,
        # "0" -> off) and raise the gen-0 threshold (EQUILIBRIA_GTAP_GC_THRESHOLD:
        # unset -> 50000, 0 -> off) for the solve; inside a caller's family scope
        # (ALTERTAX) the caller's setup stays as it is
        with _gc_family_scope():
            return _solve_family(
                params,
                closure,
                residual_region,
                shock,
                shock_levels,
                cache_root,
                jax_cache,
                timing,
                timer,
                ref_gdx,
                nominal_trade_balance,
                keep_session=keep_session,
                newton_tol=newton_tol,
            )
    finally:
        activate(prev)


def _solve_shock_ratio(
    run: FamilyRun,
    mp: MPArrays,
    *,
    shock: Any,
    shock_levels: bool,
    warm_start: bool = False,
    label: str = "shock",
) -> tuple[MPArrays, dict[str, Any]]:
    """The shock period under ``trade_balance="ratio"``: recipe and system as a plain
    shock solve (savf held at the check's value), then ONE Newton solve of the system
    bordered by :class:`RatioBorder`; the post-solve sees the savf targets of the
    solution's ``W``. Stats as ``_solve_period`` (``ratio="bordered"``)."""
    from equilibria.family.post_solve import post_solve

    sv = mp.vars["savf"]
    chk = {r: float(sv.value[sv.pos((r, "check"))]) for r in run.nominal_savf}
    w_chk = _world_income(mp, "check")
    with scope("shock"):
        mp1 = _period_recipe(
            run,
            mp,
            "shock",
            base_calibrated=True,
            solve_check=True,
            shock=shock,
            shock_levels=shock_levels,
            warm_start=warm_start,
        )
        sq, mp_sq, ss = _period_system(run, mp1, "shock")
        if not chk:
            raise ValueError("trade_balance='ratio' needs the nominal_savf regions")
        with phase("ratio_border"):
            border = RatioBorder(ss, sq, chk, w_chk)
        if len(border.g_ix) != len(run.regions):
            raise ValueError(
                f"trade_balance='ratio': {len(border.g_ix)} regy shock columns for "
                f"{len(run.regions)} regions"
            )
        token, keep = _NEWTON_TOL.set(run.newton_tol), _MUMPS_KEEP.set(run.mumps)
        try:
            res, resid, code = _newton(ss, sq, border)
        finally:
            _NEWTON_TOL.reset(token)
            _MUMPS_KEEP.reset(keep)
        w = float(res.x[-1])
        x = np.asarray(res.x[:-1], dtype=float)
        sv1 = mp_sq.vars["savf"]
        for r, value in border.targets(w).items():
            sv1.value[sv1.pos((r, "shock"))] = value
        with phase("post_solve"):
            post_solve(
                mp_sq,
                "shock",
                backend=run.backend,
                state=run.state,
                params=run.params,
                regions=run.regions,
                if_sub=run.if_sub,
                sq=sq,
                x=x,
                code=code,
            )
    rec = {**solve_record(label, res, resid, code), "ratio": "bordered"}
    run.solves.append(rec)
    return mp_sq, {
        "code": code,
        "residual": resid,
        "nit": res.nit,
        "solves": [rec],
        "ratio": "bordered",
        "ratio_iterations": 1,
        "world_income": w,
    }


@dataclass
class FamilySession:
    """The run context of a finished ``solve_gtap_family`` (``keep_session=True``): the
    shock period solved again at other shocks, with the same base and check. Used for the
    sub-steps of a shock path (``family.altertax.path``)."""

    run: FamilyRun
    after_check: MPArrays  # the arrays the shock period of the run started from
    ratio: bool  # trade_balance="ratio": the bordered solve (RatioBorder) each time

    def solve_shock(
        self, shock: Any, *, shock_levels: bool = False, start: MPArrays | None = None
    ) -> tuple[MPArrays, dict[str, Any]]:
        """The shock period at ``shock``, as the run solved its own. ``start``: arrays
        of a previous shock solve of this session; the free cells start from its shock
        values (warm start) instead of the check's. Returns ``(arrays, stats)``."""
        warm = start is not None
        mp = start if start is not None else self.after_check
        if self.ratio:
            return _solve_shock_ratio(
                self.run, mp, shock=shock, shock_levels=shock_levels, warm_start=warm
            )
        return _solve_period(
            self.run,
            mp,
            "shock",
            base_calibrated=True,
            solve_check=True,
            shock=shock,
            shock_levels=shock_levels,
            warm_start=warm,
        )

    def shock_system(
        self,
        shock: Any,
        *,
        start: MPArrays | None = None,
        savf_targets: Mapping[str, float] | None = None,
        shock_levels: bool = False,
    ) -> tuple[Any, Squared, MPArrays]:
        """The shock period at ``shock`` prepared as :meth:`solve_shock` prepares one
        solve (recipe warm from ``start``, square, system build), not solved: ``(ss,
        sq, mp_sq)``, ``ss.x0`` the start point. ``savf_targets`` (``{region: savf}``)
        holds the ``nominal_savf`` regions there (``trade_balance="ratio"``)."""
        run = self.run
        run.savf_targets = None if savf_targets is None else dict(savf_targets)
        try:
            with scope("shock"):
                mp1 = _period_recipe(
                    run,
                    start if start is not None else self.after_check,
                    "shock",
                    base_calibrated=True,
                    solve_check=True,
                    shock=shock,
                    shock_levels=shock_levels,
                    warm_start=start is not None,
                )
                sq, mp_sq, ss = _period_system(run, mp1, "shock")
        finally:
            run.savf_targets = None
        return ss, sq, mp_sq

    def finish_shock(self, mp_sq: MPArrays, sq: Squared, x: np.ndarray) -> MPArrays:
        """``post_solve`` of the shock period at the point ``x`` of ``sq`` (``mp_sq``
        modified and returned), as after a converged solve."""
        from equilibria.family.post_solve import post_solve

        run = self.run
        with scope("shock"), phase("post_solve"):
            return post_solve(
                mp_sq,
                "shock",
                backend=run.backend,
                state=run.state,
                params=run.params,
                regions=run.regions,
                if_sub=run.if_sub,
                sq=sq,
                x=x,
                code=1,
            )


def _solve_family(
    params,
    closure,
    residual_region,
    shock,
    shock_levels,
    cache_root,
    jax_cache,
    timing,
    timer,
    ref_gdx=None,
    nominal_trade_balance=False,
    keep_session=False,
    newton_tol=None,
) -> FamilyResult:
    t0 = time.perf_counter()
    run, mp = prepare_run(
        params,
        closure,
        residual_region,
        cache_root=cache_root,
        jax_cache=jax_cache,
        freeze=_gc_freeze_from_env(),
        ref_gdx=ref_gdx,
        newton_tol=newton_tol,
    )
    if nominal_trade_balance:
        run.nominal_savf = tuple(r for r in run.regions if r != run.residual_region)
    ratio = nominal_trade_balance == "ratio"
    timing["prepare_s"] = time.perf_counter() - t0
    # records, cross records and MPArrays are now loaded and long-lived: hide them
    # from the full passes too (the entry-point freeze predates them)
    _gc_freeze_now(_gc_freeze_from_env())

    t = time.perf_counter()
    settled, settle = calibrate_base_cached(mp, run)
    timing["calibrate_s"] = time.perf_counter() - t

    cur = dataclasses.replace(mp, settled_seed=settled, base_calibrated=True)
    results: dict[str, dict[str, Any]] = {}
    after_check = cur
    for period in PERIODS:
        if period == "shock":
            after_check = cur
        t = time.perf_counter()
        if ratio and period == "shock":
            cur, results[period] = _solve_shock_ratio(
                run, cur, shock=shock, shock_levels=shock_levels
            )
        else:
            cur, results[period] = _solve_period(
                run,
                cur,
                period,
                base_calibrated=True,
                solve_check=True,
                shock=shock,
                shock_levels=shock_levels,
            )
        timing[f"{period}_s"] = time.perf_counter() - t
    run.mumps.clear()  # the shock's factors: no later solve of the run reuses them
    timing["total_s"] = time.perf_counter() - t0
    return FamilyResult(
        cur,
        results,
        settled,
        settle,
        timing,
        dict(timer.seconds),
        dict(timer.counts),
        session=FamilySession(run, after_check, ratio) if keep_session else None,
        solves=list(run.solves),
    )
