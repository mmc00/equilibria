"""GTAP multi-period driver: replicates GAMS loop(tsim) + iterloop.gms.

Solves base → check → shock sequentially (the altertax CD three-period pipeline).
Each period is solved by PATH via a temporary single-period model; results are
written back to the multi-period model `m` and the period is then frozen.

This mirrors what GAMS does in loop(tsim): each iteration creates an effective
single-period solve, then the solved period is holdfixed (var.fx(tsim-1)=var.l)
before the next period is solved.

Public API
----------
solve_multiperiod(m, params, closure, *, ref_gdx=None) -> dict
    Returns {period: {"code": int, "residual": float}} for each period.

freeze_period(m, period)
    Fixes all Var families at the given period so they serve as holdfixed
    parameters for subsequent periods.
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
# _seed_sp_from_mp — seed single-period model from multi-period model for a period
# ---------------------------------------------------------------------------
def _seed_sp_from_mp(sp_model, mp_model, period: str) -> int:
    """Copy Var values from mp_model[...,period] to sp_model Vars.

    Returns number of values set.
    """
    from pyomo.environ import Var

    n_set = 0
    for sp_v in sp_model.component_objects(Var, active=True):
        vname = sp_v.name
        mp_v = getattr(mp_model, vname, None)
        if mp_v is None:
            continue
        if sp_v.is_indexed():
            for k in sp_v.index_set():
                kt = (*((k,) if not isinstance(k, tuple) else k), period)
                try:
                    mp_vd = mp_v[kt]
                    sp_vd = sp_v[k]
                    val = float(mp_vd.value) if mp_vd.value is not None else 1.0
                    if not sp_vd.fixed:
                        sp_vd.set_value(val)
                        n_set += 1
                except (KeyError, TypeError):
                    pass
        else:
            # Scalar Var in single-period → indexed by (period,) in multi-period
            try:
                mp_vd = mp_v[(period,)]
                sp_vd = sp_v[None]
                val = float(mp_vd.value) if mp_vd.value is not None else 1.0
                if not sp_vd.fixed:
                    sp_vd.set_value(val)
                    n_set += 1
            except (KeyError, TypeError):
                pass
    return n_set


# ---------------------------------------------------------------------------
# _write_sp_to_mp — write single-period solve results back to multi-period model
# ---------------------------------------------------------------------------
def _write_sp_to_mp(sp_model, mp_model, period: str) -> int:
    """Copy Var values from solved sp_model to mp_model[...,period].

    Returns number of values written.
    """
    from pyomo.environ import Var

    n_set = 0
    for sp_v in sp_model.component_objects(Var, active=True):
        vname = sp_v.name
        mp_v = getattr(mp_model, vname, None)
        if mp_v is None:
            continue
        if sp_v.is_indexed():
            for k in sp_v.index_set():
                kt = (*((k,) if not isinstance(k, tuple) else k), period)
                try:
                    sp_vd = sp_v[k]
                    mp_vd = mp_v[kt]
                    val = float(sp_vd.value) if sp_vd.value is not None else 1.0
                    if not mp_vd.fixed:
                        mp_vd.set_value(val)
                        n_set += 1
                except (KeyError, TypeError):
                    pass
        else:
            try:
                sp_vd = sp_v[None]
                mp_vd = mp_v[(period,)]
                val = float(sp_vd.value) if sp_vd.value is not None else 1.0
                if not mp_vd.fixed:
                    mp_vd.set_value(val)
                    n_set += 1
            except (KeyError, TypeError):
                pass
    return n_set


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
# solve_multiperiod — main public function
# ---------------------------------------------------------------------------
def solve_multiperiod(
    m,
    params,
    closure,
    *,
    ref_gdx=None,
) -> dict[str, dict[str, Any]]:
    """Replicate GAMS loop(tsim): solve base → check → shock with prior-period holdfixed.

    Strategy (mirrors diff_altertax.py three-period pipeline):
    For each period, builds a fresh single-period model seeded from mp_model,
    solves it via _run_path_capi_nonlinear_full, writes results back to mp_model,
    then freezes the period in mp_model for subsequent periods.

    The multi-period model `m` serves as the persistent state store — its Var
    values for each period are updated from single-period solve results, and
    freeze_period() pins them as holdfixed anchors for subsequent Fisher rows.

    Parameters
    ----------
    m : Pyomo ConcreteModel built by GTAPMultiPeriodModel
        Must already have build_vars, build_equations_intra (all 3 periods),
        and build_equations_fisher called on it.
    params : GTAPParameters (used for altertax elasticities / imptx shock)
    closure : GTAPClosureConfig or None
    ref_gdx : path to GAMS reference GDX (optional warm-start seed, not yet used)

    Returns
    -------
    dict mapping "base" | "check" | "shock" → {"code": int, "residual": float}
    """
    from equilibria.templates.gtap import GTAPModelEquations
    from equilibria.templates.gtap.altertax import apply_altertax_elasticities
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_parity_pipeline import GTAPVariableSnapshot

    run_gtap = _load_run_gtap()

    # Determine residual region from the multi-period model's attribute or params.
    res_region = getattr(m, "_residual_region", None)
    if res_region is None:
        res_region = list(params.sets.r)[-1]

    # Apply altertax elasticities (mirrors diff_altertax [1/3] betaCal setup).
    p_alt = apply_altertax_elasticities(params, in_place=False)

    # Build closures.
    _closure_name = getattr(closure, "name", None) if closure is not None else None
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

    # ── Phase 1: BASE period ─────────────────────────────────────────────────
    # Build single-period base model with betaCal init (mirrors diff_altertax [1/3]).
    eq_b = GTAPModelEquations(
        p_alt.sets, p_alt, base_closure, residual_region=res_region
    )
    m_b = eq_b.build_model()

    # Seed from multi-period model base values (init from build_vars).
    _seed_sp_from_mp(m_b, m, "base")

    # Solve base via PATH wrapper.
    r_base = run_gtap._run_path_capi_nonlinear_full(
        m_b, p_alt,
        enforce_post_checks=False,
        strict_path_capi=False,
        closure_config=base_closure,
        equation_scaling=True,
        solution_hint=None,
    )
    code_base = int(r_base.get("termination_code") or 0)
    res_base = float(r_base.get("residual") or float("inf"))
    results["base"] = {"code": code_base, "residual": res_base}

    # Write base solve results back to multi-period model.
    _write_sp_to_mp(m_b, m, "base")

    # Freeze base period in multi-period model (holdfixed=1 for base).
    freeze_period(m, "base")
    # Mandatory: ensure pabs[r,'base'] is pinned (GAMS iterloop pabs.fx=1 at base).
    for r in m.r:
        try:
            pabs_bd = m.pabs[r, "base"]
            if not pabs_bd.fixed:
                pabs_bd.fix(1.0)
        except (KeyError, AttributeError):
            pass

    # ── Phase 2: CHECK period ────────────────────────────────────────────────
    # Build single-period check model with altertax closure and t0_snapshot=m_b.
    # phiP[check] = pcons[base] = 1.0 (GAMS convention).
    for _r in p_alt.sets.r:
        try:
            if hasattr(p_alt, "calibration") and hasattr(p_alt.calibration, "phip"):
                p_alt.calibration.phip[(_r,)] = 1.0
        except Exception:
            pass

    eq_chk = GTAPModelEquations(
        p_alt.sets, p_alt, alt_closure,
        residual_region=res_region,
        t0_snapshot=m_b,
    )
    m_chk = eq_chk.build_model()

    # Override phip=1.0 on built model.
    for _r in p_alt.sets.r:
        try:
            if hasattr(m_chk, "phip"):
                m_chk.phip[_r].set_value(1.0)
        except Exception:
            pass

    # Unfix regy (GAMS regYeq.regY endogenous in compStat).
    for _r in p_alt.sets.r:
        try:
            if hasattr(m_chk, "regy") and m_chk.regy[_r].fixed:
                m_chk.regy[_r].unfix()
        except Exception:
            pass

    # Seed check from multi-period model check values (init = base init from build_vars).
    _seed_sp_from_mp(m_chk, m, "check")

    # Solve check via PATH wrapper (warm from base solved state).
    warm_b = GTAPVariableSnapshot.from_python_model(m_chk)
    r_chk = run_gtap._run_path_capi_nonlinear_full(
        m_chk, p_alt,
        enforce_post_checks=False,
        strict_path_capi=False,
        closure_config=alt_closure,
        equation_scaling=True,
        solution_hint=warm_b,
    )
    code_chk = int(r_chk.get("termination_code") or 0)
    res_chk = float(r_chk.get("residual") or float("inf"))
    results["check"] = {"code": code_chk, "residual": res_chk}

    # Write check solve results back to multi-period model.
    _write_sp_to_mp(m_chk, m, "check")

    # Freeze check period in multi-period model.
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

    # t0_snapshot=m_b (the base, not check): pf0/xf0 Fisher anchors from benchmark.
    eq_alt = GTAPModelEquations(
        params_shock.sets, params_shock, alt_closure,
        residual_region=res_region,
        t0_snapshot=m_b,
    )
    m_alt = eq_alt.build_model()

    # phiP[shock] = pcons[base] = 1.0 (same convention as check).
    for _r in params_shock.sets.r:
        try:
            if hasattr(m_alt, "phip"):
                m_alt.phip[_r].set_value(1.0)
        except Exception:
            pass

    # Unfix regy (same as check period).
    for _r in params_shock.sets.r:
        try:
            if hasattr(m_alt, "regy") and m_alt.regy[_r].fixed:
                m_alt.regy[_r].unfix()
        except Exception:
            pass

    # Seed shock from multi-period model shock values (init = check solved values
    # after _write_sp_to_mp for check).
    _seed_sp_from_mp(m_alt, m, "shock")

    # Solve shock via PATH wrapper (warm from check solved state).
    warm_chk = GTAPVariableSnapshot.from_python_model(m_alt)
    r_shk = run_gtap._run_path_capi_nonlinear_full(
        m_alt, params_shock,
        enforce_post_checks=False,
        strict_path_capi=False,
        closure_config=alt_closure,
        equation_scaling=True,
        solution_hint=warm_chk,
    )
    code_shk = int(r_shk.get("termination_code") or 0)
    res_shk = float(r_shk.get("residual") or float("inf"))
    results["shock"] = {"code": code_shk, "residual": res_shk}

    # Write shock solve results back to multi-period model.
    _write_sp_to_mp(m_alt, m, "shock")
    # Freeze shock as well (for completeness / report purposes).
    freeze_period(m, "shock")

    return results
