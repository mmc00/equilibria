"""GTAP multi-period driver: replicates GAMS loop(tsim) + iterloop.gms.

Solves base → check → shock sequentially on the FULL multi-period model `m`.
Each period is solved by PATH via freeze_inactive_periods: all vars outside the
active period are fixed at their current values, making the full `m` effectively
single-period-for-PATH while the inactive-period equations remain as satisfied
rows referencing fixed vars (exactly the GAMS holdfixed=1 structure).

After each period solves, the solved values remain fixed for subsequent periods,
replicating GAMS iterloop.gms:149-178 `var.fx(tsim-1) = var.l(tsim-1)`.

Public API
----------
solve_multiperiod(m, params, closure, *, ref_gdx=None) -> dict
    Returns {period: {"code": int, "residual": float}} for each period.

freeze_period(m, period)
    Fixes all Var families at the given period so they serve as holdfixed
    parameters for subsequent periods.

freeze_inactive_periods(m, active_period)
    Freezes all var cells NOT belonging to active_period, leaving only
    active_period cells free (subject to PATH solve).
"""
from __future__ import annotations

import copy
import importlib.util as _iu
import sys
from pathlib import Path
from typing import Any, Optional

# Root of the repository (four levels up from this file):
#   src/equilibria/templates/gtap/ → src/equilibria/templates/ → src/equilibria/ →
#   src/ → repository root
ROOT = Path(__file__).resolve().parents[4]

PERIODS = ("base", "check", "shock")


# ---------------------------------------------------------------------------
# Lazy-load run_gtap (mirrors how diff_altertax.py does it)
# ---------------------------------------------------------------------------
def _load_run_gtap():
    spec = _iu.spec_from_file_location(
        "run_gtap", str(ROOT / "scripts" / "gtap" / "run_gtap.py")
    )
    rg = _iu.module_from_spec(spec)
    sys.modules["run_gtap"] = rg
    spec.loader.exec_module(rg)
    return rg


# ---------------------------------------------------------------------------
# freeze_period — replicate iterloop.gms var.fx(tsim-1) = var.l(tsim-1)
# ---------------------------------------------------------------------------
def freeze_period(m, period: str) -> int:
    """Fix every VarData at index (..., period) at its current value.

    Replicates GAMS iterloop.gms:149-178 holdfixed block for the given period.
    Returns the count of VarData entries that were fixed.
    """
    from pyomo.environ import Var

    n_fixed = 0
    for v in m.component_objects(Var, active=True):
        for idx in v:
            # idx is a tuple whose last element is the period label,
            # or a scalar string equal to the period label.
            t = idx[-1] if isinstance(idx, tuple) else idx
            if t != period:
                continue
            vd = v[idx]
            try:
                val = float(vd.value) if vd.value is not None else 1.0
                vd.fix(val)
                n_fixed += 1
            except Exception:
                pass
    return n_fixed


# ---------------------------------------------------------------------------
# freeze_inactive_periods — freeze vars AND deactivate constraints for inactive periods
# ---------------------------------------------------------------------------
def freeze_inactive_periods(m, active_period: str) -> int:
    """Fix all VarData cells NOT belonging to active_period and deactivate their constraints.

    This makes the full multi-period model `m` look like a single-period system
    to PATH:
      - active_period vars are free; inactive-period vars are fixed.
      - active_period constraints are active (Jacobian rows for PATH).
      - inactive-period constraints are deactivated (no rows → square system).

    The Fisher cross-period equations (eq_rgdpmp, eq_pgdpmp, eq_pabs, eq_pfact,
    eq_pwfact) are ALWAYS deactivated for inactive periods since those rows
    reference fixed base-period vars and would be trivially satisfied (not useful
    as PATH rows); the active period's Fisher rows remain active.

    Returns count of VarData entries frozen.
    """
    from pyomo.environ import Var, Constraint

    # 1. Fix vars for inactive periods; unfix active period vars.
    n_fixed = 0
    for v in m.component_objects(Var, active=True):
        for idx in v:
            t = idx[-1] if isinstance(idx, tuple) else idx
            if t == active_period:
                # Unfix active period vars so PATH can move them.
                vd = v[idx]
                if vd.fixed:
                    try:
                        vd.unfix()
                    except Exception:
                        pass
                continue
            vd = v[idx]
            if vd.fixed:
                continue  # already frozen — idempotent
            try:
                val = float(vd.value) if vd.value is not None else 1.0
                vd.fix(val)
                n_fixed += 1
            except Exception:
                pass

    # 2. Activate/deactivate constraints per period.
    #    Constraint index last element = period label.
    for con_comp in m.component_objects(Constraint):
        for idx in list(con_comp):
            # Determine the period for this constraint cell.
            if isinstance(idx, tuple):
                t = idx[-1]
            elif isinstance(idx, str):
                t = idx
            else:
                # Non-period constraint (e.g., scalar with no period dim) — skip
                continue

            if t not in PERIODS:
                # Index doesn't end with a period label → skip (not a time-indexed con)
                continue

            cd = con_comp[idx]
            if t == active_period:
                if not cd.active:
                    cd.activate()
            else:
                if cd.active:
                    cd.deactivate()

    return n_fixed


# ---------------------------------------------------------------------------
# _replicate_sp_fixing — copy fixed-var state from single-period model to mp
# ---------------------------------------------------------------------------
def _replicate_sp_fixing(m, sp_model, active_period: str) -> int:
    """Copy the fixed-variable state from sp_model to m for active_period.

    In single-period, GTAPModelEquations.build_model() internally fixes many
    structural zeros (afeall, p_rai, chiSave, etc.) via apply_production_scaling
    and data-driven fixing. The multi-period model doesn't replicate these.

    This function mirrors that fixing: for each var fixed in sp_model (before
    any closure), fix the corresponding m var at the same value for active_period.

    Returns count of vars fixed.
    """
    from pyomo.environ import Var

    n_fixed = 0
    for sp_v in sp_model.component_objects(Var, active=True):
        vname = sp_v.name
        mp_v = getattr(m, vname, None)
        if mp_v is None:
            continue
        for sp_idx in sp_v.index_set():
            sp_vd = sp_v[sp_idx]
            if not sp_vd.fixed:
                continue  # only replicate fixed vars
            # Build multi-period index
            if sp_v.is_indexed():
                if isinstance(sp_idx, tuple):
                    mp_idx = (*sp_idx, active_period)
                else:
                    mp_idx = (sp_idx, active_period)
            else:
                mp_idx = (active_period,)
            try:
                mp_vd = mp_v[mp_idx]
                if not mp_vd.fixed:
                    val = float(sp_vd.value) if sp_vd.value is not None else 0.0
                    mp_vd.fix(val)
                    n_fixed += 1
            except (KeyError, TypeError):
                pass
    return n_fixed


# ---------------------------------------------------------------------------
# _replicate_sp_bounds — copy lower/upper bounds from single-period to mp
# ---------------------------------------------------------------------------
def _replicate_sp_bounds(m, sp_model, active_period: str) -> int:
    """Copy lb/ub bounds from sp_model to m for active_period vars.

    Single-period build_model sets meaningful lower bounds (e.g., ev, cv, xd,
    etc. get lb = 0.001 * initial_value > 0) to prevent PATH from driving
    variables to zero/negative.  The multi-period model's build_vars doesn't
    replicate these bounds, so PATH can violate them and cause ZeroDivisionError
    in CDE/Armington equations.

    This function mirrors those bounds: for each free var in sp_model with lb or
    ub, set the same bounds on the matching active-period var in m.

    Returns count of bounds replicated.
    """
    from pyomo.environ import Var

    n_bounds = 0
    for sp_v in sp_model.component_objects(Var, active=True):
        vname = sp_v.name
        mp_v = getattr(m, vname, None)
        if mp_v is None:
            continue
        for sp_idx in sp_v.index_set():
            sp_vd = sp_v[sp_idx]
            if sp_vd.lb is None and sp_vd.ub is None:
                continue  # no bounds to replicate
            # Build multi-period index
            if sp_v.is_indexed():
                if isinstance(sp_idx, tuple):
                    mp_idx = (*sp_idx, active_period)
                else:
                    mp_idx = (sp_idx, active_period)
            else:
                mp_idx = (active_period,)
            try:
                mp_vd = mp_v[mp_idx]
                if sp_vd.lb is not None:
                    mp_vd.setlb(sp_vd.lb)
                if sp_vd.ub is not None:
                    mp_vd.setub(sp_vd.ub)
                n_bounds += 1
            except (KeyError, TypeError):
                pass
    return n_bounds


# ---------------------------------------------------------------------------
# _apply_imptx_shock — multiply all imptx entries by (1+factor) tariff-power
# ---------------------------------------------------------------------------
def _apply_imptx_shock(params, factor: float = 0.10) -> None:
    """Apply tm_pct-mode shock to params.taxes.imptx in-place.

    Mirrors GAMS: tm.fx = tm.l * (1+shock)
      → imptx_new = (1 + imptx_old) * (1 + factor) - 1
    """
    for key in list(params.taxes.imptx.keys()):
        # Skip diagonal (domestic sales, rp==r) — no tariff
        if len(key) == 3 and key[0] == key[2]:
            continue
        old = float(params.taxes.imptx[key] or 0.0)
        params.taxes.imptx[key] = (1.0 + old) * (1.0 + factor) - 1.0


# ---------------------------------------------------------------------------
# _seed_period_from_prior — warm-start active period vars from prior period
# ---------------------------------------------------------------------------
def _seed_period_from_prior(m, prior_period: str, active_period: str) -> int:
    """Copy Var values from prior_period to active_period as warm start.

    Returns number of values set.
    """
    from pyomo.environ import Var

    n_set = 0
    for v in m.component_objects(Var, active=True):
        for idx in v:
            t = idx[-1] if isinstance(idx, tuple) else idx
            if t != active_period:
                continue
            # Build the prior-period index by replacing the trailing t
            if isinstance(idx, tuple):
                prior_idx = (*idx[:-1], prior_period)
            else:
                prior_idx = prior_period
            try:
                prior_vd = v[prior_idx]
                cur_vd = v[idx]
                if prior_vd.value is not None and not cur_vd.fixed:
                    cur_vd.set_value(float(prior_vd.value))
                    n_set += 1
            except (KeyError, TypeError):
                pass
    return n_set


# ---------------------------------------------------------------------------
# solve_multiperiod — main public function
# ---------------------------------------------------------------------------
def solve_multiperiod(
    m,
    params,
    closure,
    *,
    ref_gdx=None,
) -> dict[str, dict[str, Any]]:
    """Replicate GAMS loop(tsim): solve base → check → shock on the FULL model m.

    Strategy (freeze_inactive_periods per period):
    For each period in (base, check, shock):
      1. freeze_inactive_periods(m, period) — pin all non-active-period vars
      2. seed active period from prior-period solved values (warm start)
      3. apply period-specific setup (altertax elasticities, imptx shock, pnum)
      4. call _run_path_capi_nonlinear_full(m, ...) ON `m` ITSELF (not a temp model)
      5. After solve, the active period vars stay fixed (freeze_inactive_periods
         for the next period will also freeze them — idempotent)

    The multi-period model `m` must already have:
      - build_vars, build_equations_intra (all 3 periods), build_equations_fisher

    Fisher rows (eq_rgdpmp, eq_pgdpmp, eq_pabs, eq_pfact, eq_pwfact) are live
    Jacobian rows in `m`; the PATH solve of each period sees them as constraints
    referencing the FIXED base-period vars (constants in Jacobian for check/shock).

    Parameters
    ----------
    m : Pyomo ConcreteModel built by GTAPMultiPeriodModel
    params : GTAPParameters (used for altertax elasticities / imptx shock)
    closure : GTAPClosureConfig or None
    ref_gdx : path to GAMS reference GDX (optional, not yet used)

    Returns
    -------
    dict mapping "base" | "check" | "shock" → {"code": int, "residual": float}
    """
    from equilibria.templates.gtap.altertax import apply_altertax_elasticities
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap import GTAPModelEquations

    run_gtap = _load_run_gtap()

    # Determine residual region from the multi-period model's attribute or params.
    res_region = getattr(m, "_residual_region", None)
    if res_region is None:
        res_region = list(params.sets.r)[-1]

    # Apply altertax elasticities (mirrors diff_altertax [1/3] betaCal setup).
    p_alt = apply_altertax_elasticities(params, in_place=False)

    # Build closures.
    _if_sub = getattr(closure, "if_sub", False) if closure is not None else False
    _numeraire = getattr(closure, "numeraire", "pnum") if closure is not None else "pnum"

    base_closure = GTAPClosureConfig(
        name="base", closure_type="MCP",
        capital_mobility="sluggish", fix_endowments=False,
        fix_taxes=False, fix_technology=False, if_sub=_if_sub,
        numeraire=_numeraire,
    )
    alt_closure = GTAPClosureConfig(
        name="altertax", closure_type="MCP",
        capital_mobility="mobile", fix_endowments=False,
        fix_taxes=True, fix_technology=True, if_sub=_if_sub,
        numeraire=_numeraire,
    )

    results: dict[str, dict[str, Any]] = {}

    # Store reference to m on itself for residual_region use
    m._residual_region = res_region

    # ── Phase 1: BASE period ─────────────────────────────────────────────────
    # Freeze check and shock periods; leave base free.
    freeze_inactive_periods(m, "base")

    # FIX B (B1): deactivate eq_xft[r,f,'base'] for every (r,f) cell where
    # eq_xfteq[r,f,'base'] is also active.  In the multi-period model the base
    # period uses a non-altertax closure (name="base"), so the deactivation block
    # inside _run_path_capi_nonlinear_full that fires only for name=="altertax"
    # never runs, leaving BOTH eq_xft and eq_xfteq active for the same xft var →
    # 14 surplus rows that make the system over-determined (code=0 / residual=inf).
    # Replicating the single-period altertax deactivation for the base period:
    # whenever eq_xfteq is active for (r,f,base), eq_xft is redundant and must go.
    _eq_xft_mp = getattr(m, "eq_xft", None)
    _eq_xfteq_mp = getattr(m, "eq_xfteq", None)
    if _eq_xft_mp is not None and _eq_xfteq_mp is not None:
        _n_xft_deact_base = 0
        for _r in m.r:
            for _f in m.f:
                _xfteq_idx = (_r, _f, "base")
                try:
                    _xfteq_cd = _eq_xfteq_mp[_xfteq_idx]
                except KeyError:
                    continue
                if not _xfteq_cd.active:
                    continue
                # eq_xfteq is active for this (r,f,'base') → deactivate eq_xft
                try:
                    _xft_cd = _eq_xft_mp[(_r, _f, "base")]
                except KeyError:
                    continue
                if _xft_cd.active:
                    _xft_cd.deactivate()
                    _n_xft_deact_base += 1
        if _n_xft_deact_base:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "base period: deactivated eq_xft for %d (r,f) pairs "
                "(eq_xfteq active → eq_xft redundant, no name==altertax gate)",
                _n_xft_deact_base,
            )

    # Replicate single-period build_model's structural fixing for base period.
    # build_model internally fixes ~500+ structural zeros (afeall, p_rai, chiSave,
    # etc.) that apply_conditional_fixing doesn't cover. Without this, aggressive
    # structural matching fixes the wrong 639 vars, breaking PATH convergence.
    _sp_ref_base = GTAPModelEquations(
        p_alt.sets, p_alt, base_closure, residual_region=res_region
    ).build_model()
    _replicate_sp_fixing(m, _sp_ref_base, "base")
    _replicate_sp_bounds(m, _sp_ref_base, "base")
    del _sp_ref_base

    # Solve base on m via PATH.
    r_base = run_gtap._run_path_capi_nonlinear_full(
        m, p_alt,
        enforce_post_checks=False,
        strict_path_capi=False,
        closure_config=base_closure,
        equation_scaling=True,
        solution_hint=None,
    )
    code_base = int(r_base.get("termination_code") or 0)
    res_base = float(r_base.get("residual") or float("inf"))
    results["base"] = {"code": code_base, "residual": res_base}

    # Freeze base period (holdfixed=1 for subsequent periods).
    freeze_period(m, "base")
    # Ensure pabs[r,'base'] is pinned (GAMS iterloop pabs.fx=1 at base).
    for r in m.r:
        try:
            pabs_bd = m.pabs[r, "base"]
            if not pabs_bd.fixed:
                pabs_bd.fix(1.0)
        except (KeyError, AttributeError):
            pass

    # ── Phase 2: CHECK period ────────────────────────────────────────────────
    # Freeze base and shock; leave check free.
    freeze_inactive_periods(m, "check")

    # Warm-start check from base solved values.
    _seed_period_from_prior(m, "base", "check")

    # phiP[check] = pcons[base] = 1.0 (GAMS convention).
    for _r in p_alt.sets.r:
        try:
            if hasattr(p_alt, "calibration") and hasattr(p_alt.calibration, "phip"):
                p_alt.calibration.phip[(_r,)] = 1.0
        except Exception:
            pass

    # Unfix regy[r,'check'] (GAMS regYeq.regY endogenous in compStat).
    for _r in p_alt.sets.r:
        try:
            if hasattr(m, "regy") and m.regy[_r, "check"].fixed:
                m.regy[_r, "check"].unfix()
        except Exception:
            pass

    # Replicate single-period structural fixing for check period.
    _sp_ref_chk = GTAPModelEquations(
        p_alt.sets, p_alt, alt_closure, residual_region=res_region,
    ).build_model()
    _replicate_sp_fixing(m, _sp_ref_chk, "check")
    _replicate_sp_bounds(m, _sp_ref_chk, "check")
    del _sp_ref_chk

    # Solve check on m.
    r_chk = run_gtap._run_path_capi_nonlinear_full(
        m, p_alt,
        enforce_post_checks=False,
        strict_path_capi=False,
        closure_config=alt_closure,
        equation_scaling=True,
        solution_hint=None,
    )
    code_chk = int(r_chk.get("termination_code") or 0)
    res_chk = float(r_chk.get("residual") or float("inf"))
    results["check"] = {"code": code_chk, "residual": res_chk}

    # Freeze check period.
    freeze_period(m, "check")
    for r in m.r:
        try:
            pabs_cd = m.pabs[r, "check"]
            if not pabs_cd.fixed:
                pabs_cd.fix(float(pabs_cd.value or 1.0))
        except (KeyError, AttributeError):
            pass

    # ── Phase 3: SHOCK period ────────────────────────────────────────────────
    # Apply +10% imptx shock to params (tm_pct mode).
    params_shock = copy.deepcopy(p_alt)
    _apply_imptx_shock(params_shock, factor=0.10)

    # Freeze base and check; leave shock free.
    freeze_inactive_periods(m, "shock")

    # Warm-start shock from check solved values.
    _seed_period_from_prior(m, "check", "shock")

    # Unfix regy[r,'shock'] (same as check period).
    for _r in params_shock.sets.r:
        try:
            if hasattr(m, "regy") and m.regy[_r, "shock"].fixed:
                m.regy[_r, "shock"].unfix()
        except Exception:
            pass

    # Fix pnum for shock period (numeraire anchor).
    try:
        if hasattr(m, "pnum"):
            # pnum is scalar in single-period; in multi-period it's indexed by t
            pnum_shock = m.pnum["shock"]
            if not pnum_shock.fixed:
                pnum_shock.fix(1.5)
    except (KeyError, AttributeError, TypeError):
        pass

    # Replicate single-period structural fixing for shock period.
    _sp_ref_shk = GTAPModelEquations(
        params_shock.sets, params_shock, alt_closure, residual_region=res_region,
    ).build_model()
    _replicate_sp_fixing(m, _sp_ref_shk, "shock")
    _replicate_sp_bounds(m, _sp_ref_shk, "shock")
    del _sp_ref_shk

    # Solve shock on m with shocked params.
    r_shk = run_gtap._run_path_capi_nonlinear_full(
        m, params_shock,
        enforce_post_checks=False,
        strict_path_capi=False,
        closure_config=alt_closure,
        equation_scaling=True,
        solution_hint=None,
    )
    code_shk = int(r_shk.get("termination_code") or 0)
    res_shk = float(r_shk.get("residual") or float("inf"))
    results["shock"] = {"code": code_shk, "residual": res_shk}

    # Freeze shock as well (for completeness / report purposes).
    freeze_period(m, "shock")

    return results
