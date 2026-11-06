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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from equilibria.family.mp_arrays import MPArrays, Squared

# Settings that change the Newton-TR system or add debug side effects the family route
# does not reproduce. Any of them set (non-empty) is refused.
_UNSUPPORTED_ENV: tuple[str, ...] = (
    "EQUILIBRIA_GTAP_REF_MODEL",
    "EQUILIBRIA_GTAP_JAC_DUMP",
    "EQUILIBRIA_GTAP_WARM_START",
    "EQUILIBRIA_GTAP_DIRECT_DEGEN_FIX",
    "EQUILIBRIA_GTAP_BTF_PROBE",
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

    if not if_sub:
        refuse("ifSUB=0 (closure.if_sub must be True)")
    if savf_flag != "capFix":
        refuse(f"savf_flag={savf_flag!r} (only 'capFix')")
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


def default_cache_root() -> Path:
    """Where the records and the compiled programs live (``EQUILIBRIA_FAMILY_CACHE``)."""
    env = os.environ.get(CACHE_ENV)
    return Path(env) if env else Path.home() / ".cache" / "equilibria" / "family"


@dataclass
class FamilyRun:
    """What every solve call of one run reads and never changes."""

    params: Any
    closure: Any  # the driver's base_closure (the recipe and the squaring read it)
    backend: Any  # single-period recording
    state: Any  # composed state (levels, fixed cells, Parameters)
    cross: Any  # cross-period recording
    ref: Any  # SPRef: the single-period reference as arrays
    masks: Any  # ConditionalMasks
    regions: list[str]
    if_sub: bool


@dataclass
class FamilyResult:
    """What the array route returns: no Pyomo model, no fill.

    ``results[period]`` is ``{"code", "residual", "nit"}`` (the base is not solved:
    ``skipped_solve``); ``settled`` is the dict the calibration returned;
    ``settle`` the same stats for the settle's own calls; ``timing`` seconds per phase."""

    mp: MPArrays
    results: dict[str, dict[str, Any]]
    settled: dict
    settle: dict[str, dict[str, Any]] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)


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
    )


def prepare_run(
    params: Any,
    closure: Any,
    residual_region: str,
    *,
    cache_root: Path | None = None,
    jax_cache: bool = True,
) -> tuple[FamilyRun, MPArrays]:
    """The run context and the multiperiod arrays, from the disk cache where possible.

    Loads (or records, once) the single-period and cross-period families, composes the
    state once, and builds ``MPArrays`` from it: what ``build_block_model`` does, with no
    Pyomo model. ``params`` takes the closure's value-added subsidy basis in place, as
    ``build_block_model`` does."""
    from equilibria.family import cross_period
    from equilibria.family.disk_cache import enable_jax_cache, load_or_build
    from equilibria.family.prep_recipe import sp_reference_arrays
    from equilibria.family.prep_square import ConditionalMasks
    from equilibria.templates.gtap.residual_region import resolve_residual_region

    residual_region = resolve_residual_region(
        params, params.sets, closure, residual_region
    )
    params.set_va_subsidy_basis(getattr(closure, "va_subsidy_basis", "gams"))
    root = Path(cache_root) if cache_root is not None else default_cache_root()
    if jax_cache:
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
    mp = MPArrays.build(state, backend, cross, params=params, base_calibrated=False)
    run = FamilyRun(
        params=params,
        closure=_driver_closure(closure),
        backend=backend,
        state=state,
        cross=cross,
        ref=sp_reference_arrays(params, closure, residual_region, state=state),
        masks=ConditionalMasks.from_params(params, mp),
        regions=[str(r) for r in params.sets.r],
        if_sub=bool(getattr(closure, "if_sub", False)),
    )
    return run, mp


def _newton(ss: Any, sq: Squared) -> tuple[Any, float, int]:
    """Newton-TR on the squared system: ``(NewtonResult, ||F||_inf, code)``.

    The settings are read from the environment as the Pyomo path reads them (the
    ``NEWTON_MUMPS`` set); the family pairing is the system's own column order, and
    MUMPS runs with ICNTL(6)=7. ``code`` is 1 iff Newton succeeded and the residual of
    the squared system is below ``CODE_RESIDUAL`` (``_ok_sr``)."""
    from equilibria.solver.newton_tr import newton_tr

    linsolve = os.environ.get("EQUILIBRIA_GTAP_TR_LINSOLVE", "direct").lower()
    n = len(ss.x0)
    lb = np.array([sq.cells[c][2] for c in sq.col_names], dtype=float)
    ub = np.array([sq.cells[c][3] for c in sq.col_names], dtype=float)
    lb[~np.isfinite(lb)] = -1e20  # as the Pyomo path
    ub[~np.isfinite(ub)] = 1e20
    # the relative test needs RELTOL > 0, then fscale applies
    reltol = float(os.environ.get("EQUILIBRIA_GTAP_TR_RELTOL", "0") or 0.0)
    fscale = ss.fscale() if reltol > 0.0 else None
    res = newton_tr(
        lambda z: ss.F(np.clip(z, lb, ub)),
        lambda z: ss.J(np.clip(z, lb, ub)),
        np.clip(ss.x0, lb, ub),
        lb,
        ub,
        colperm=np.arange(n),  # row k is paired with column k
        fscale=fscale,
        reltol=reltol,
        icntl6=ICNTL6_FAMILY if linsolve == "mumps" else None,
        row_names=sq.row_names,
        col_names=sq.col_names,
        linsolve=linsolve,
        maxit=int(os.environ.get("EQUILIBRIA_GTAP_SCIPY_MAXITER", "300")),
        # GMIN lands on other entries under the family pairing than under Pyomo's
        # and moves weakly determined cells by ~1e-7: factor without it and
        # regularize only when MUMPS reports null pivots.
        gmin=float(os.environ.get("EQUILIBRIA_GTAP_GMIN", "0") or 0.0),
        gmin_policy="on_null_pivot",
    )
    resid = float(np.linalg.norm(ss.F(np.clip(res.x, lb, ub)), np.inf))
    return res, resid, int(bool(res.success) and resid < CODE_RESIDUAL)


def _solve_period(
    run: FamilyRun,
    mp: MPArrays,
    period: str,
    *,
    base_calibrated: bool,
    solve_check: bool,
    shock: Any = None,
    shock_levels: bool = False,
) -> tuple[MPArrays, dict[str, Any]]:
    """One period of ``_solve_multiperiod_inner``: recipe, solve, post-solve.

    Returns the arrays the next period starts from (``mp`` itself is not modified)."""
    from equilibria.family.post_solve import post_solve
    from equilibria.family.prep_recipe import run_recipe
    from equilibria.family.prep_square import square
    from equilibria.family.squared_system import SquaredSystem, period_incidence

    mp1 = run_recipe(
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
    )
    if period == "base":  # skip_base_solve: the base is the calibrated benchmark
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
    sq, mp_sq = square(
        mp1,
        period,
        params=run.params,
        closure=run.closure,
        incidence=period_incidence(run.backend, run.state, period, run.cross),
        exogenous=run.ref.exogenous,
        masks=run.masks,
        return_mp=True,
    )
    ss = SquaredSystem.build(run.backend, run.state, mp_sq, period, sq, cross=run.cross)
    res, resid, code = _newton(ss, sq)
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
    return mp_sq, {"code": code, "residual": resid, "nit": res.nit}


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
            run, cur, period, base_calibrated=False, solve_check=False
        )
    return settled_check_point(cur), results


def solve_gtap_family(
    params: Any, closure: Any, residual_region: str, *, ref_gdx: Any = None
) -> Any:
    """Run base, check and shock on arrays. Filled in by the next task."""
    raise NotImplementedError(
        "family backend: solve_gtap_family is not implemented yet"
    )
