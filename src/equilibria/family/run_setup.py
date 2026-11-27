"""The run context of the family backend and its preparation (ADR-0011).

``prepare_run`` loads or records the families of a GTAP run (no Pyomo model) and
returns its ``FamilyRun``; ``check_supported`` refuses, before any work, what the
family route cannot run; the settle-cache key and the settled check point of
``FactorBlock.calibrate_base``. Moved here from the old family driver (PR 3d of
gtap-on-core): the core template uses them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from equilibria.family.capflex import stamp_capflex_risk
from equilibria.family.disk_cache import _family_code_digest
from equilibria.family.mp_arrays import MPArrays
from equilibria.family.period_newton import ICNTL6_FAMILY
from equilibria.family.phases import phase
from equilibria.solver.gc_tuning import (
    _gc_freeze_now,
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
    # the MUMPS analysis of the last solve, for the next solve of the same pattern
    # (``newton_tr(mumps_keep=)``: the check on the settle's); emptied when the run ends
    mumps: dict = field(default_factory=dict, repr=False)
    # one record per Newton solve of the run, in order (:func:`solve_record`)
    solves: list = field(default_factory=list, repr=False)
    # key of the opt-in settle cache over what ``prepare_run`` sees (``_settle_key``
    # adds the nominal-savf rule); None = do not cache
    settle_key: str | None = None
    # the records' cache folder (``disk_cache``), where ``square`` keeps its pairings;
    # None = keep them in memory only
    records_dir: Path | None = None
    # ``ref`` as the core route's ``Reference`` (``GTAPFamilyPeriodModel.sp_reference``),
    # built by its first call and shared by every model of the run
    sp_reference: Any = field(default=None, repr=False)
    # {(variable, period): (reference positions, period positions)} of ``ref``'s cells
    # (``GTAPFamilyPeriodModel.replicate_reference``); the keys never change in a run
    ref_positions: dict = field(default_factory=dict, repr=False)


def _driver_closure(closure: Any) -> Any:
    """The base closure the old Pyomo multiperiod driver used for the route's closure."""
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
    Under capFlex ``params._capflex_risk`` comes from the data (see ``family.capflex``),
    set before the cache keys are taken."""
    from equilibria.family import cross_period
    from equilibria.family.disk_cache import cache_key, enable_jax_cache, load_or_build
    from equilibria.family.prep_square import ConditionalMasks
    from equilibria.family.sp_arrays import sp_reference_arrays
    from equilibria.templates.gtap.residual_region import resolve_residual_region

    residual_region = resolve_residual_region(
        params, params.sets, closure, residual_region
    )
    params.set_va_subsidy_basis(getattr(closure, "va_subsidy_basis", "gams"))
    params.set_unit_sigma(getattr(closure, "unit_sigma", "gams"))
    stamp_capflex_risk(params, closure)
    root = Path(cache_root) if cache_root is not None else default_cache_root()
    if jax_cache:
        with phase("jax_cache_setup"):
            enable_jax_cache(root)
    # one key for the single- and cross-period records (and the pairings next to them)
    with phase("sp_cache_key"):
        key = cache_key(params, closure, residual_region)
    backend, state = load_or_build(params, closure, residual_region, root, key=key)
    cross = cross_period.load_or_record(
        params,
        closure,
        residual_region,
        root,
        sets=backend.sets,
        var_domains=backend.var_domains,
        key=key,
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
        settle_key=_settle_base_key(
            params, closure, residual_region, ref_gdx, newton_tol
        ),
        records_dir=None if key is None else root / key,
    )
    return run, mp


SETTLE_CACHE_ENV = "EQUILIBRIA_FAMILY_SETTLE_CACHE"
# where the caches live, and the switch itself: they do not change the settle
_SETTLE_KEY_ENV_SKIP = frozenset(
    {
        SETTLE_CACHE_ENV,
        "EQUILIBRIA_FAMILY_CACHE",
        "EQUILIBRIA_SEED_CACHE",
        "EQUILIBRIA_SEED_CACHE_DISABLE",
    }
)


def _settle_base_key(
    params: Any,
    closure: Any,
    residual_region: str,
    ref_gdx: Any,
    newton_tol: tuple[float, float] | None,
) -> str | None:
    """The settle cache key over data, closure, residual region, ref GDX content, the
    build + solve code (``blocks/gtap/fingerprint``) and ``equilibria/family``, the
    Newton tolerances and every ``EQUILIBRIA_*`` variable (linear solver, tolerances,
    ...). ``None`` unless ``EQUILIBRIA_FAMILY_SETTLE_CACHE=1``, under an @overwrite
    hook, or when an input cannot be covered."""
    if os.environ.get(SETTLE_CACHE_ENV) != "1":
        return None
    from equilibria.blocks.gtap.fingerprint import file_digest, fingerprint
    from equilibria.blocks.gtap.overwrite import overwrite

    if overwrite.active():
        return None
    gdx = "nogdx"
    if ref_gdx is not None:
        gdx = file_digest(ref_gdx)
        if gdx is None:
            return None
    env = sorted(
        (k, v)
        for k, v in os.environ.items()
        if k.startswith("EQUILIBRIA_") and k not in _SETTLE_KEY_ENV_SKIP
    )
    return fingerprint(
        "family-seed",
        params=params,
        closure=closure,
        residual_region=str(residual_region),
        ref_gdx=gdx,
        family_code=_family_code_digest(),
        newton_tol=newton_tol,
        env=env,
    )


def _settle_key(run: FamilyRun) -> str | None:
    """``run.settle_key`` plus the nominal-savf rule (set on the run after
    ``prepare_run``)."""
    if run.settle_key is None:
        return None
    import hashlib

    tail = hashlib.sha256(repr(tuple(run.nominal_savf)).encode()).hexdigest()[:8]
    return f"{run.settle_key}-{tail}"


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
