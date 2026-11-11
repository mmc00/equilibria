"""Family backend driver: the multiperiod GTAP run on arrays, without a Pyomo model.

Opt-in (``_solve_shock(..., backend="family")`` or ``EQUILIBRIA_GTAP_BACKEND=family``).
The route never falls back to Pyomo: a configuration it cannot run raises
``NotImplementedError`` naming it, before any work is done.

User shocks are not refused here: every ShockBlock instrument is applied through the
same ``apply_shock`` semantics as the Pyomo route.
"""

from __future__ import annotations

import dataclasses
import os
import time
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from equilibria.family.capflex import require_ref_gdx, stamp_capflex_risk
from equilibria.family.mp_arrays import MPArrays, Squared
from equilibria.family.phases import PhaseTimer, activate, phase, scope
from equilibria.solver.gc_tuning import (
    _gc_freeze_from_env,
    _gc_freeze_now,
    _gc_frozen_permanent_graph,
    _gc_relaxed_threshold,
    _gc_threshold_from_env,
)

# Settings that change the Newton-TR system or add debug side effects the family route
# does not reproduce. Any of them set (non-empty) is refused.
_UNSUPPORTED_ENV: tuple[str, ...] = (
    "EQUILIBRIA_GTAP_REF_MODEL",
    "EQUILIBRIA_GTAP_JAC_DUMP",
    "EQUILIBRIA_GTAP_WARM_START",
    "EQUILIBRIA_GTAP_DIRECT_DEGEN_FIX",
    "EQUILIBRIA_GTAP_BTF_PROBE",
    # Pyomo turns it into ``shock_steps`` (``_solve_shock_in_steps``)
    "EQUILIBRIA_GTAP_SHOCK_CONTINUATION",
    # NEOS takes precedence over the scipy solver in ``path_capi``
    "EQUILIBRIA_GTAP_SOLVE_NEOS",
    # rescale the equations the Pyomo route hands to Newton (the family rows are raw)
    "EQUILIBRIA_GTAP_BENCH_SCALE",
    "EQUILIBRIA_GTAP_ROW_EQUIL",
    # seeds the exported .nl from a GDX
    "EQUILIBRIA_SEED_FROM_GDX",
)


def check_supported(
    *,
    if_sub: bool,
    savf_flag: str,
    mode: str,
    skip_base_solve: bool,
    mute_welfare: bool,
    seed_from_prior: bool,
    solve_check: bool,
    shock_steps: tuple[float, ...] | None,
    shock_method: str,
    env: Mapping[str, str] | None = None,
) -> None:
    """Raise ``NotImplementedError`` naming the first setting the family route cannot run."""
    env = os.environ if env is None else env

    def refuse(what: str) -> None:
        raise NotImplementedError(f"family backend: {what} is not supported")

    if savf_flag == "capFlex":
        if not if_sub:
            refuse(
                "savf_flag='capFlex' with if_sub=False (capFlex runs with ifSUB=1 only)"
            )
    elif savf_flag != "capFix":
        refuse(f"savf_flag={savf_flag!r} (only 'capFix' and 'capFlex')")
    if mode != "gtap":
        refuse(f"mode={mode!r} (only 'gtap')")
    if not skip_base_solve:
        refuse("skip_base_solve=False (the base is not solved)")
    if seed_from_prior:
        refuse("seed_from_prior=True")
    if not mute_welfare:
        refuse("mute_welfare=False")
    if not solve_check:
        refuse("solve_check=False (the route runs base, check and shock)")
    if shock_steps:
        refuse("shock_steps")
    if shock_method != "nonlinear":
        refuse(f"shock_method={shock_method!r} (only 'nonlinear')")
    if env.get("EQUILIBRIA_GTAP_SOLVER") != "scipy_newton_tr":
        refuse(
            f"EQUILIBRIA_GTAP_SOLVER={env.get('EQUILIBRIA_GTAP_SOLVER')!r} "
            "(only 'scipy_newton_tr')"
        )
    linsolve = env.get("EQUILIBRIA_GTAP_TR_LINSOLVE", "direct").lower()
    if linsolve not in ("mumps", "direct"):
        refuse(f"EQUILIBRIA_GTAP_TR_LINSOLVE={linsolve!r} (only 'mumps' or 'direct')")
    if not env.get("EQUILIBRIA_GTAP_SOLVE_NLP"):
        refuse("EQUILIBRIA_GTAP_SOLVE_NLP unset (the NLP-square route is required)")
    if env.get("EQUILIBRIA_GTAP_MUMPS_ICNTL6"):
        refuse(
            "EQUILIBRIA_GTAP_MUMPS_ICNTL6 (the family route fixes MUMPS "
            f"ICNTL(6)={ICNTL6_FAMILY} by design)"
        )
    for name in _UNSUPPORTED_ENV:
        if env.get(name):
            refuse(f"{name}")
    for name in sorted(env):
        if name.startswith("EQUILIBRIA_DEBUG_") and env.get(name):
            refuse(f"{name}")

    from equilibria.blocks.gtap.overwrite import overwrite

    if overwrite.active():
        refuse("@overwrite hooks")


CACHE_ENV = "EQUILIBRIA_FAMILY_CACHE"
PERIODS = ("base", "check", "shock")
ICNTL6_FAMILY = 7  # MUMPS ICNTL(6) on the family squared system (Pyomo keeps 0)
CODE_RESIDUAL = 1e-5  # code 1 needs ||F||_inf below this, as ``_ok_sr``


def solve_record(label: str, res: Any, resid: float, code: int) -> dict[str, Any]:
    """What a run reports of one Newton solve: its ``code`` (1 = converged), ``nit``,
    the final ``residual``, the MUMPS null pivots summed over its factorizations
    (``null_pivots``; ``null_pivot_factors`` counts the factorizations that had any),
    the GMIN refactorizations they triggered (``gmin_retries``) and the factorizations
    that raised (``factor_failures``)."""
    piv = [int(k) for k in getattr(res, "null_pivots", ()) or ()]
    return {
        "solve": label,
        "code": int(code),
        "nit": int(getattr(res, "nit", 0)),
        "residual": float(resid),
        "converged": bool(getattr(res, "success", False)),
        "null_pivots": sum(piv),
        "null_pivot_factors": sum(1 for k in piv if k > 0),
        "gmin_retries": int(getattr(res, "gmin_factors", 0)),
        "factor_failures": int(getattr(res, "factor_failures", 0)),
    }


def default_cache_root() -> Path:
    """Where the records and the compiled programs live (``EQUILIBRIA_FAMILY_CACHE``)."""
    env = os.environ.get(CACHE_ENV)
    return Path(env) if env else Path.home() / ".cache" / "equilibria" / "family"


@dataclass
class FamilyRun:
    """What every solve call of one run reads. Constant, except ``savf_targets``, which
    ``FamilySession.shock_system`` sets around one shock build and resets."""

    params: Any
    closure: Any  # the driver's base_closure (the recipe and the squaring read it)
    backend: Any  # single-period recording
    state: Any  # composed state (levels, fixed cells, Parameters)
    cross: Any  # cross-period recording
    ref: Any  # SPRef: the single-period reference as arrays
    masks: Any  # ConditionalMasks
    regions: list[str]
    if_sub: bool
    # derived once per run, filled by the first solve call that needs them
    # regions whose savf the shock holds nominal (``nominal_trade_balance``); () = off
    nominal_savf: tuple[str, ...] = ()
    # {region: value} the held savf takes in the shock (``trade_balance="ratio"``); None
    # = the check's value
    savf_targets: Any = field(default=None, repr=False)
    residual_region: str = ""  # the resolved residual region of the run
    # (reltol, ftol) of every Newton solve of the run; None = the environment's
    # EQUILIBRIA_GTAP_TR_RELTOL / _FTOL
    newton_tol: tuple[float, float] | None = None
    cell_index: Any = field(default=None, repr=False)
    adjacency: dict = field(default_factory=dict, repr=False)
    # one record per Newton solve of the run, in order (:func:`solve_record`)
    solves: list = field(default_factory=list, repr=False)


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


def _driver_closure(closure: Any) -> Any:
    """``base_closure`` of ``_solve_multiperiod_inner`` for the route's closure."""
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=bool(getattr(closure, "if_sub", False)),
        numeraire=getattr(closure, "numeraire", "pnum"),
        unit_sigma=getattr(closure, "unit_sigma", "gams"),
    )


def prepare_run(
    params: Any,
    closure: Any,
    residual_region: str,
    *,
    cache_root: Path | None = None,
    jax_cache: bool = True,
    freeze: bool = False,
    ref_gdx: Any = None,
    newton_tol: tuple[float, float] | None = None,
) -> tuple[FamilyRun, MPArrays]:
    """The run context and the multiperiod arrays, from the disk cache where possible.

    Loads (or records, once) the single-period and cross-period families, composes the
    state once, and builds ``MPArrays`` from it: what ``build_block_model`` does, with no
    Pyomo model. ``params`` takes the closure's value-added subsidy basis and
    ``unit_sigma`` in place, as ``build_block_model`` does. ``freeze`` moves the loaded records to the GC's permanent
    generation before ``mp_build``; only for a caller inside a context that unfreezes.
    Under capFlex ``ref_gdx`` gives ``params._capflex_risk`` (see ``family.capflex``), set
    before the cache keys are taken."""
    from equilibria.family import cross_period
    from equilibria.family.disk_cache import enable_jax_cache, load_or_build
    from equilibria.family.prep_recipe import sp_reference_arrays
    from equilibria.family.prep_square import ConditionalMasks
    from equilibria.templates.gtap.residual_region import resolve_residual_region

    residual_region = resolve_residual_region(
        params, params.sets, closure, residual_region
    )
    params.set_va_subsidy_basis(getattr(closure, "va_subsidy_basis", "gams"))
    params.set_unit_sigma(getattr(closure, "unit_sigma", "gams"))
    stamp_capflex_risk(params, closure, ref_gdx)
    root = Path(cache_root) if cache_root is not None else default_cache_root()
    if jax_cache:
        with phase("jax_cache_setup"):
            enable_jax_cache(root)
    backend, state = load_or_build(params, closure, residual_region, root)
    cross = cross_period.load_or_record(
        params,
        closure,
        residual_region,
        root,
        sets=backend.sets,
        var_domains=backend.var_domains,
    )
    # the loaded records are long-lived and large: freeze them before mp_build and the
    # later steps churn allocations over them (the entry-point freeze predates them)
    _gc_freeze_now(freeze)
    with phase("mp_build"):
        mp = MPArrays.build(state, backend, cross, params=params, base_calibrated=False)
    with phase("sp_reference"):
        ref = sp_reference_arrays(params, closure, residual_region, state=state)
    with phase("masks"):
        masks = ConditionalMasks.from_params(params, mp)
    run = FamilyRun(
        params=params,
        closure=_driver_closure(closure),
        backend=backend,
        state=state,
        cross=cross,
        ref=ref,
        masks=masks,
        regions=[str(r) for r in params.sets.r],
        if_sub=bool(getattr(closure, "if_sub", False)),
        residual_region=str(residual_region),
        newton_tol=newton_tol,
    )
    return run, mp


# (reltol, ftol) of the solve in progress; set by ``_solve_period_inner`` from the run
_NEWTON_TOL: ContextVar[tuple[float, float] | None] = ContextVar(
    "_NEWTON_TOL", default=None
)


def _newton(ss: Any, sq: Squared, border: Any = None) -> tuple[Any, float, int]:
    """Newton-TR on the squared system: ``(NewtonResult, ||F||_inf, code)``.

    ``border`` (a :class:`RatioBorder`): the system bordered with the world-income
    unknown and its row (``trade_balance="ratio"``); ``res.x`` then ends with ``W``.

    The settings are read from the environment as the Pyomo path reads them (the
    ``NEWTON_MUMPS`` set); the family pairing is the system's own column order, and
    MUMPS runs with ICNTL(6)=7. ``code`` is 1 iff Newton succeeded and the residual of
    the squared system is below ``CODE_RESIDUAL`` (``_ok_sr``), else 2 as the Pyomo path.
    ``_NEWTON_TOL`` (the run's ``newton_tol``, set around the call) replaces the
    environment's two stopping tolerances."""
    from equilibria.solver.newton_tr import newton_tr

    linsolve = os.environ.get("EQUILIBRIA_GTAP_TR_LINSOLVE", "direct").lower()
    lb = np.array([sq.cells[c][2] for c in sq.col_names], dtype=float)
    ub = np.array([sq.cells[c][3] for c in sq.col_names], dtype=float)
    F, x0 = ss.F, np.asarray(ss.x0, dtype=float)
    J = lambda z: ss.J(z)  # noqa: E731 - looked up at the first Jacobian
    row_names, col_names = list(sq.row_names), list(sq.col_names)
    if border is not None:
        lb, ub = np.append(lb, -np.inf), np.append(ub, np.inf)
        F, J, x0 = border.F, border.J, np.append(x0, border.w0)
        row_names.append(border.ROW)
        col_names.append(border.COL)
    lb[~np.isfinite(lb)] = -1e20  # as the Pyomo path
    ub[~np.isfinite(ub)] = 1e20
    n = len(x0)
    # the relative test needs RELTOL > 0, then fscale applies
    tol = _NEWTON_TOL.get()
    if tol is None:
        reltol, ftol = (
            float(os.environ.get("EQUILIBRIA_GTAP_TR_RELTOL", "0") or 0.0),
            None,
        )
    else:
        reltol, ftol = tol
    with phase("fscale"):
        fscale = ss.fscale() if reltol > 0.0 else None
    if fscale is not None and border is not None:
        fscale = np.append(fscale, border.fscale_row)

    def timed_F(z):
        with phase("newton_F"):
            return F(np.clip(z, lb, ub))

    def timed_J(z):
        with phase("newton_J"):
            return J(np.clip(z, lb, ub))

    with phase("newton_other"):  # analysis, factor, solve and the TR logic
        res = newton_tr(
            timed_F,
            timed_J,
            np.clip(x0, lb, ub),
            lb,
            ub,
            colperm=np.arange(n),  # row k is paired with column k
            fscale=fscale,
            reltol=reltol,
            ftol=ftol,
            icntl6=ICNTL6_FAMILY if linsolve == "mumps" else None,
            row_names=row_names,
            col_names=col_names,
            linsolve=linsolve,
            maxit=int(os.environ.get("EQUILIBRIA_GTAP_SCIPY_MAXITER", "300")),
            # GMIN lands on other entries under the family pairing than under Pyomo's
            # and moves weakly determined cells by ~1e-7: factor without it and
            # regularize only when MUMPS reports null pivots.
            gmin=float(os.environ.get("EQUILIBRIA_GTAP_GMIN", "0") or 0.0),
            gmin_policy="on_null_pivot",
            # analysed on the pattern alone (Pyomo's wrapper), ICNTL(6)=7 cannot use
            # the values and met a spurious null pivot on the 20x41 capFlex settle
            mumps_values_at_analysis=True,
        )
    with phase("residual_check"):
        resid = float(np.linalg.norm(F(np.clip(res.x, lb, ub)), np.inf))
    return res, resid, 1 if (bool(res.success) and resid < CODE_RESIDUAL) else 2


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
    token = _NEWTON_TOL.set(run.newton_tol)
    try:
        res, resid, code = _newton(ss, sq)
    finally:
        _NEWTON_TOL.reset(token)
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


def _period_system(run: FamilyRun, mp1: MPArrays, period: str) -> tuple[Any, Any, Any]:
    """The square and system-build steps of ``_solve_period_inner`` on the recipe's
    arrays ``mp1``: ``(sq, mp_sq, ss)``."""
    from equilibria.family.cell_index import CellIndex
    from equilibria.family.prep_square import square
    from equilibria.family.squared_system import ArrayIncidence, SquaredSystem

    with phase("square"):
        if run.cell_index is None or not run.cell_index.fits(mp1):
            run.cell_index = CellIndex(mp1)
        if period not in run.adjacency:
            run.adjacency[period] = ArrayIncidence(
                run.backend, run.state, period, run.cross
            )
        sq, mp_sq = square(
            mp1,
            period,
            params=run.params,
            closure=run.closure,
            incidence=run.adjacency[period],
            index=run.cell_index,
            exogenous=run.ref.exogenous,
            masks=run.masks,
            return_mp=True,
        )
    with phase("system_build"):
        ss = SquaredSystem.build(
            run.backend, run.state, mp_sq, period, sq, cross=run.cross
        )
    return sq, mp_sq, ss


def settled_check_point(mp: MPArrays) -> dict[str, dict]:
    """The check slice without the exogenous instruments, ``{var: {body: value}}``
    (``FactorBlock._settled_check_point``): ``body`` is the index without the period,
    a bare label for one index. Variables indexed by the period alone are not in it."""
    from equilibria.templates.gtap.instruments import is_free_cell

    settled: dict[str, dict] = {}
    for name, v in mp.vars.items():
        if len(v.keys) == 0 or len(v.keys[0]) < 2:
            continue
        libres = mp.endogenous_instrument_cells.get(name)
        exo = name in mp.exogenous
        cells = settled.setdefault(name, {})
        for i in np.flatnonzero(v.period == "check"):
            k = v.keys[i]
            if exo and (not libres or not is_free_cell(k, libres)):
                continue
            body = k[:-1]
            cells[body[0] if len(body) == 1 else body] = float(v.value[i])
        if not cells:
            del settled[name]
    return settled


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
        # unset -> 50000, 0 -> off) for the solve
        with (
            _gc_frozen_permanent_graph(_gc_freeze_from_env()),
            _gc_relaxed_threshold(_gc_threshold_from_env()),
        ):
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


def _world_income(mp: MPArrays, period: str) -> float:
    v = mp.vars["regy"]
    return float(sum(v.value[i] for i in np.flatnonzero(v.period == period)))


class RatioBorder:
    """GEMPACK ``del_tbalry(r) = 0`` inside the shock's Newton system. ``savf[r]`` of
    every non-residual region is held at ``savf_check[r] * W / W_check`` (``W`` = world
    income = ``sum(regy)``), the level form of ``INCOMEWLD * del_tbalry = TB * y - TBAL *
    ywld`` with ``del_tbalry = 0`` (gtapv7.tab, E_del_tbalry). ``W`` is one more unknown
    (the last), with its own row ``W - g.x = 0`` (``g`` the ``regy`` shock columns);
    ``F`` writes the savf targets of ``W`` into the system's fixed slots
    (``SquaredSystem.set_fixed``) before evaluating it. The Jacobian is bordered: ``[[J,
    v], [-g, 1]]``, ``v = dF/dW`` through the targets (finite difference, once, at the
    start). The solution is the fixed point the outer iterations of a plain solve at
    ``T(W)`` reach (tested), in one Newton solve."""

    ROW = "eq_ratio_world_income"
    COL = "ratio_world_income"

    def __init__(self, ss: Any, sq: Squared, chk: Mapping[str, float], w_chk: float):
        self.ss, self.chk, self.w_chk = ss, dict(chk), float(w_chk)
        g = np.fromiter(
            (c.startswith("regy[") and c.endswith(",shock]") for c in sq.col_names),
            dtype=bool,
            count=len(sq.col_names),
        )
        self.g_ix = np.flatnonzero(g)
        self.n = len(sq.col_names)
        x0 = np.asarray(ss.x0, dtype=float)
        self.w0 = float(x0[self.g_ix].sum())
        self.fscale_row = max(1.0, abs(self.w0))
        self._w: float | None = None
        f0 = self._F_at(x0, self.w0)
        d = 1e-6 * self.w_chk
        v = (self._F_at(x0, self.w0 + d) - f0) / d
        self._F_at(x0, self.w0)  # the slots back at the start
        self.v_rows = np.flatnonzero(v)
        self.v_vals = v[self.v_rows]

    def targets(self, w: float) -> dict[str, float]:
        """``{region: savf}`` at world income ``w``."""
        return {r: x * w / self.w_chk for r, x in self.chk.items()}

    def _set(self, w: float) -> None:
        if w != self._w:
            self.ss.set_fixed(
                "savf", {(r, "shock"): x for r, x in self.targets(w).items()}
            )
            self._w = w

    def _F_at(self, x: np.ndarray, w: float) -> np.ndarray:
        self._set(w)
        return self.ss.F(x)

    def F(self, z: np.ndarray) -> np.ndarray:
        x, w = z[:-1], float(z[-1])
        return np.append(self._F_at(x, w), w - x[self.g_ix].sum())

    def J(self, z: np.ndarray) -> Any:
        import scipy.sparse as sp

        self._set(float(z[-1]))
        r, c, d = self.ss.J_entries(z[:-1])
        n, ng, nv = self.n, len(self.g_ix), len(self.v_rows)
        rows = np.concatenate([r, self.v_rows, np.full(ng + 1, n)])
        cols = np.concatenate([c, np.full(nv, n), self.g_ix, [n]])
        data = np.concatenate([d, self.v_vals, -np.ones(ng), [1.0]])
        return sp.csr_matrix((data, (rows, cols)), shape=(n + 1, n + 1))


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
        token = _NEWTON_TOL.set(run.newton_tol)
        try:
            res, resid, code = _newton(ss, sq, border)
        finally:
            _NEWTON_TOL.reset(token)
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
    settled, settle = calibrate_base_arrays(mp, run)
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
