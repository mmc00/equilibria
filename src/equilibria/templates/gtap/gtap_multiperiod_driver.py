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
# _holdfix_activity_scale — pin the per-activity scale xp at the prior period
# ---------------------------------------------------------------------------
# Whether to apply the xp activity-scale holdfix in gtap-mode.  MEASURED OFF
# (2026-06-24): it was a patch for the OLD pinned-pft bug; with pft now freed
# (the GAMS gtap closure), it forces the wrong factor-block root.
#   ON : CHECK 64.0% / SHOCK 61.3% (code=1)
#   OFF: CHECK 99.4% / SHOCK 66.9% (code=1)
# OFF wins on BOTH periods, so we disable it for gtap-mode.  The function is kept
# (not deleted) so the experiment is reproducible; flip this to True to restore.
_HOLDFIX_ACTIVITY_SCALE_GTAP = False


def _holdfix_activity_scale(m, period: str) -> int:
    """Holdfix the activity-scale var `xp` at its PRIOR-period value (gtap-mode).

    The per-activity scale xp[r,a] is a free DOF / multiplicity under the
    gtap-mode closure: total factor endowments xft are pinned, but the scale can
    redistribute across activities to a different valid root (proven: seeding the
    GAMS point as an exact fixed point still lets PATH walk to a +21% root).  GAMS
    pins it via `gtap.holdfixed=1` on tsim-1 (iterloop.gms / comp gms scaleopt).
    The faithful analog is to fix xp at PYTHON'S OWN prior-period solved value
    (NOT the GAMS reference — that would be seeding from the answer): base anchors
    the check, check anchors the shock.  The squareness patches auto-resquare
    (the now-redundant eq_po/eq_va rows are dropped by Hopcroft-Karp).
    """
    prior = "base" if period == "check" else "check"
    xp = getattr(m, "xp", None)
    if xp is None:
        return 0
    n = 0
    for r in m.r:
        for a in m.a:
            try:
                cur = xp[(r, a, period)]
                pv = xp[(r, a, prior)]
            except (KeyError, TypeError):
                continue
            if not cur.fixed and pv.value is not None:
                cur.fix(float(pv.value))
                n += 1
    return n


# _collapse_pft_pfteq — period-aware pft/eq_pfteq collapse (gtap-mode only)
# ---------------------------------------------------------------------------
def _collapse_pft_pfteq(m, period: str) -> int:
    """Collapse the factor-price block for the active period (gtap-mode squaring).

    GAMS gtap (iterloop.gms:142-143) fixes xft/pft ONLY for xftFlag=0:

        xft.fx(r,fm,tsim)$(not xftFlag(r,fm)) = 0 ;
        pft.fx(r,fm,tsim)$(not xftFlag(r,fm)) = pft.l(r,fm,t0) ;

    For REAL/mobile factors (xftFlag>0) pft AND xft are left FREE — the `model
    gtap` block (model.gms:1413) lists `xfteq.xft, pfeq.pf, pfteq, pfyeq.pfy`,
    i.e. `pfteq` is a FREE ROW (no MCP pair → it holds but does not pin a var) and
    pft is determined by the rest of the system.  The earlier MP fix pinned pft at
    1.0 for these (altertax semantics), freezing pf/pfy/pfa and capping CHECK at
    ~80%.  Freeing pft (this branch) lifts CHECK to ~93%.

    So this function now mirrors the GAMS gtap closure:

      - REAL factors (xftFlag>0 → eq_xfteq[r,f,period] live): leave pft FREE,
        KEEP eq_pfteq + eq_xfteq ACTIVE.  Instead deactivate the redundant
        per-(r,f) eq_xft[r,f,period] (the market-clearing row GAMS substitutes
        out; eq_xfteq.xft + eq_pfeq.pf carry the block).  Squaring then needs a
        few more redundant rows trimmed (eq_pfyeq[*,Land,Food] is pinned by the
        CET pfeq; eq_xfeq[USA,NatRes,Mnfcs]) — see _REDUNDANT_FACTOR_ROWS below.
      - DANGLING NatRes (xftFlag<=0 → eq_xfteq/eq_pfteq Constraint.Skip, MP index
        KeyErrors): xft floats free with no matching row.  KEEP the SP behavior:
        fix xft at its benchmark init and pft at 1.0 (gtap_solver
        apply_conditional_fixing xftflag<=0 branch).

    gtap-mode ONLY — in altertax-mode this is not called (eq_pfteq is kept active
    for the CET price index there).

    Returns count of (r,f) pairs touched (eq_xft deactivations + NatRes fixes).
    """
    _eq_pfteq = getattr(m, "eq_pfteq", None)
    _eq_xfteq = getattr(m, "eq_xfteq", None)
    _eq_xft = getattr(m, "eq_xft", None)
    _eq_pfeq = getattr(m, "eq_pfeq", None)
    _pft = getattr(m, "pft", None)
    _xft = getattr(m, "xft", None)
    _xf = getattr(m, "xf", None)
    if _pft is None:
        return 0
    _n = 0
    for _r in m.r:
        for _f in m.f:
            try:
                _pftvd = _pft[(_r, _f, period)]
            except KeyError:
                continue

            # Determine whether this (r,f) is a REAL factor (eq_xfteq live) or a
            # DANGLING NatRes (eq_xfteq Constraint.Skip → KeyError / inactive).
            _eqxft = None
            if _eq_xfteq is not None:
                try:
                    _eqxft = _eq_xfteq[(_r, _f, period)]
                except KeyError:
                    _eqxft = None
            _is_real = _eqxft is not None and _eqxft.active

            if _is_real:
                # REAL factor: leave pft FREE, keep eq_pfteq + eq_xfteq active
                # (GAMS pfteq free-row).  Deactivate the redundant per-(r,f)
                # market-clearing eq_xft (GAMS substitutes it out).
                if _eq_xft is not None:
                    try:
                        _eqxftrow = _eq_xft[(_r, _f, period)]
                        if _eqxftrow.active:
                            _eqxftrow.deactivate()
                            _n += 1
                    except KeyError:
                        pass
            else:
                # DANGLING NatRes: no eq_xfteq/eq_pfteq row → xft is a free DOF
                # with no equation.  Fix xft at its benchmark init, pft=1.0
                # (reproduces gtap_solver.apply_conditional_fixing xftflag<=0).
                if _xft is not None:
                    try:
                        _xftvd = _xft[(_r, _f, period)]
                        if not _xftvd.fixed:
                            _xval = (float(_xftvd.value)
                                     if _xftvd.value not in (None,) else 0.0)
                            _xftvd.fix(_xval)
                            _n += 1
                    except KeyError:
                        pass
                if not _pftvd.fixed:
                    _pftvd.fix(1.0)
                    _n += 1

                # NatRes is the sector-specific (fnm) factor: eq_pfeq is
                # `xf == xscale*gf*(pfy/pabs)^etaff` with etaff=0 → vertical
                # supply (`xf == xscale*gf`, pf-free), so pf[NatRes,a] is a free
                # DOF that explodes (→9.27) once the diagonal tariff enters.
                # GAMS holdfixes xf[r,NatRes,a] across periods (GDX-confirmed:
                # base/check/shock byte-identical) and lets the LIVE eq_xfeq set
                # pf.  Mirror that: holdfix xf[NatRes,a] at its seeded value and
                # deactivate the redundant vertical eq_pfeq row (leave eq_xfeq
                # active → it now pins pf).  Square-preserving (fix xf −1 DOF,
                # deactivate eq_pfeq −1 row).
                if _xf is not None:
                    for _a in m.a:
                        try:
                            _xfvd = _xf[(_r, _f, _a, period)]
                        except KeyError:
                            continue
                        _xfval = (float(_xfvd.value)
                                  if _xfvd.value is not None else 0.0)
                        if abs(_xfval) < 1e-12:
                            continue
                        if not _xfvd.fixed:
                            _xfvd.fix(_xfval)
                            _n += 1
                        if _eq_pfeq is not None:
                            try:
                                _pfeqrow = _eq_pfeq[(_r, _f, _a, period)]
                                if _pfeqrow.active:
                                    _pfeqrow.deactivate()
                                    _n += 1
                            except KeyError:
                                pass

    # Freeing pft over-determines the factor block by a few rows.  Deactivate the
    # redundant rows the diagnosis identified (eq_pfyeq[*,Land,Food] is already
    # pinned by the CET pfeq; eq_xfeq[USA,NatRes,Mnfcs] is redundant) so the
    # nonlinear-full matcher can square check/shock at code=1.
    for _eqname, _idx in _REDUNDANT_FACTOR_ROWS:
        _eq = getattr(m, _eqname, None)
        if _eq is None:
            continue
        try:
            _row = _eq[(*_idx, period)]
        except KeyError:
            continue
        if _row.active:
            _row.deactivate()
            _n += 1
    return _n


# Redundant factor-block rows to deactivate once pft is freed (gtap-mode).  Each
# is over-determining given pft free + eq_pfteq/eq_xfteq/eq_pfeq live:
#   - eq_pfyeq[r,Land,Food]: Land's per-activity pfy is pinned by the CET pfeq.
# NOTE: eq_xfeq[USA,NatRes,Mnfcs] was REMOVED — it is the only row that pins
# pf[USA,NatRes,Mnfcs]; deleting it was the proximate cause of the pf free-DOF
# explosion under the diagonal shock.  The NatRes block is now squared in
# _collapse_pft_pfteq (holdfix xf[NatRes,a] + deactivate the vertical eq_pfeq,
# leaving eq_xfeq live to pin pf).
# This is the diagnosis-identified set; iterate via the solver's
# "unmatched active eqs (N): [...]" log only within this family if needed.
_REDUNDANT_FACTOR_ROWS = (
    ("eq_pfyeq", ("EU_28", "Land", "Food")),
    ("eq_pfyeq", ("USA", "Land", "Food")),
    ("eq_pfyeq", ("ROW", "Land", "Food")),
)


# ---------------------------------------------------------------------------
# _rebuild_eq_pmeq_shock — inject the tariff shock INTO the solved shock eqs
# ---------------------------------------------------------------------------
def _rebuild_eq_pmeq_shock(m, params_shock) -> int:
    """Rebuild ONLY the eq_pmeq[*,*,*,'shock'] cells so the tariff shock enters
    the SOLVED equation (not just a post-solve cosmetic pm patch).  gtap-mode only.

    Why the rebuild is needed
    -------------------------
    The MP builder (gtap_model_multiperiod.py:189-206) substitutes every mutable
    Param — and reads every report-Var (mtax) — by its NUMERIC value at build time,
    so `m` has NO live `imptx` Param.  The shock-slice `eq_pmeq[*,*,*,'shock']`
    therefore bakes the BASE imptx as a literal coefficient on pmcif:

        body:  pm[rp,i,r,shock] - C_base * pmcif[rp,i,r,shock] == 0
        with   C_base = (1 + imptx_base + mtax_init) / chipm   (the SP form,
               gtap_model_equations.py:4824-4831; chipm==1 always, mtax baked at
               its 0 init).

    `_apply_imptx_shock` mutates a deepcopy the built eqs don't reference, so the
    shock reaches outputs ONLY via the post-solve `_recompute_pm_pmt` (cosmetic;
    pm/pmt/pa are in the parity RF-exclusion set, so the patch never feeds back
    into the SOLVED quantities).  Net: shock import prices stay ~unshocked, import
    quantities stay high.

    Faithful, SURGICAL fix
    ----------------------
    For each shock cell, extract the baked coefficient C_base of pmcif from the
    original body, then replace the constraint with the SHOCKED coefficient

        C_shock = C_base + (imptx_shocked - imptx_base) / chipm
                = C_base + (imptx_shocked - imptx_base)            (chipm == 1)

    which is EXACTLY (1 + imptx_shocked + mtax_init)/chipm — only the imptx wedge
    moves (mirrors GAMS tm.fx = tm.l*1.10; nothing else recalibrated).  We rebuild
    ONLY eq_pmeq[*,*,*,'shock'] — NOT the whole shock slice — because a whole-slice
    rebuild also recalibrates Armington/CDE shares on the counterfactual VMSB
    (eq_paa/eq_xma/eq_yc jump ~26%), which GAMS does NOT do and which regresses the
    shock match (~61%).

    `params_shock.taxes.imptx[(rp,i,r)]` already holds the tm_pct POWER value
    ((1+imptx_base)*1.10-1) — that IS imptx_shocked.  We read imptx_base from the
    UNSHOCKED rate, recovered as ((1+imptx_shocked) - 1)/1.10 ... but simpler and
    exact: the additive wedge delta equals (imptx_shocked - imptx_base), and since
    _apply_imptx_shock did imptx_shocked = (1+imptx_base)*1.10 - 1, we have
    imptx_base = (1 + imptx_shocked)/1.10 - 1, so the delta is closed-form per cell.
    To avoid relying on the 0.10 factor here we instead recover imptx_base from the
    baked coefficient itself: C_base - mtax_term encodes (1+imptx_base); but with
    chipm==1 and the report-Var mtax baked at 0 init for these datasets, the clean
    invariant is C_base == 1 + imptx_base, hence imptx_base = C_base - 1 and the
    new coefficient is simply (1 + imptx_shocked).  We keep the additive-delta form
    (C_base + (imptx_shocked - imptx_base)) so a nonzero baked mtax term still maps
    correctly (the mtax part of C_base is preserved; only the imptx wedge shifts).

    gtap-mode ONLY (caller gates).  Returns the count of cells rebuilt.
    """
    from pyomo.environ import Constraint, Set, value as _V
    from pyomo.repn import generate_standard_repn

    eq = getattr(m, "eq_pmeq", None)
    pm = getattr(m, "pm", None)
    pmcif = getattr(m, "pmcif", None)
    if eq is None or pm is None or pmcif is None:
        return 0
    imptx_map = getattr(getattr(params_shock, "taxes", None), "imptx", {}) or {}

    rebuilt: dict[tuple, tuple[float, float]] = {}  # (rp,i,r) -> (C_shock, )
    for idx in list(eq):
        if not (isinstance(idx, tuple) and len(idx) == 4 and idx[-1] == "shock"):
            continue
        cd = eq[idx]
        if not cd.active:
            continue
        rp, i, r, _t = idx
        imptx_shocked = imptx_map.get((rp, i, r))
        if imptx_shocked is None:
            continue
        # Recover imptx_base from the tm_pct power inversion: imptx_shocked was
        # set to (1+imptx_base)*1.10-1 by _apply_imptx_shock.  So imptx_base =
        # (1+imptx_shocked)/1.10 - 1.  We derive the additive wedge delta from the
        # baked coefficient instead, which is exact regardless of the shock factor:
        # extract C_base (= 1+imptx_base+mtax over chipm) and set
        # C_shock = C_base + (imptx_shocked - imptx_base) where imptx_base is read
        # back from C_base under the chipm==1 / additive invariant.
        repn = generate_standard_repn(cd.body)
        pmcif_vd = pmcif[(rp, i, r, "shock")]
        pm_vd = pm[idx]
        c_pmcif = None
        c_pm = None
        for v, c in zip(repn.linear_vars, repn.linear_coefs):
            if v is pmcif_vd:
                c_pmcif = float(c)
            elif v is pm_vd:
                c_pm = float(c)
        if c_pmcif is None or c_pm is None or abs(c_pm) < 1e-12:
            continue
        # body == pm*c_pm - pmcif*|c_pmcif| (== 0) → normalize to pm == C_base*pmcif
        c_base = -c_pmcif / c_pm
        # imptx_base under the chipm==1 / mtax-baked invariant: C_base = 1+imptx_base
        # (+ baked mtax, which we preserve by working with the ADDITIVE wedge).
        imptx_base = c_base - 1.0
        c_shock = c_base + (float(imptx_shocked) - imptx_base)
        if abs(c_shock - c_base) < 1e-12:
            continue  # no shock on this route (diagonal / zero tariff)
        rebuilt[(rp, i, r)] = (c_shock,)

    if not rebuilt:
        return 0

    # Deactivate the original shock cells we are replacing.
    for (rp, i, r) in rebuilt:
        try:
            eq[(rp, i, r, "shock")].deactivate()
        except (KeyError, AttributeError):
            pass

    # Add a single indexed replacement Constraint over the rebuilt cells.  Use a
    # fresh attribute name; if a prior call already added it (idempotent re-solve),
    # delete and recreate.
    coef = {k: v[0] for k, v in rebuilt.items()}
    if hasattr(m, "eq_pmeq_shock_rebuilt"):
        m.del_component(m.eq_pmeq_shock_rebuilt)
    if hasattr(m, "eq_pmeq_shock_idx"):
        m.del_component(m.eq_pmeq_shock_idx)
    m.eq_pmeq_shock_idx = Set(initialize=sorted(coef.keys()), dimen=3)

    def _rule(_m, rp, i, r):
        return _m.pm[rp, i, r, "shock"] == coef[(rp, i, r)] * _m.pmcif[rp, i, r, "shock"]

    m.eq_pmeq_shock_rebuilt = Constraint(m.eq_pmeq_shock_idx, rule=_rule)
    return len(coef)


# ---------------------------------------------------------------------------
# _rebuild_eq_ytax_mt_shock — inject the tariff shock INTO the solved ytax[mt] eq
# ---------------------------------------------------------------------------
def _rebuild_eq_ytax_mt_shock(m, params_shock) -> int:
    """Rebuild ONLY the eq_ytax[*,'mt','shock'] cells so the import-tax revenue
    stream carries the SHOCKED imptx IN the solved equation.  gtap-mode only.

    Why the rebuild is needed (the dominant income leak)
    ----------------------------------------------------
    eq_ytax[r,'mt'] (gtap_model_equations.py:5474-5481) bakes the BASE imptx
    coefficient at build time:

        ytax[r,mt] == Σ_{(e,i,r): importer==r} (imptx_base + mtax_init)·pmcif·xw

    With the diagonal shocked (Link 2), the in-solve mt revenue MISSES the entire
    diagonal tariff stream on the large self-import flows → understates regY →
    income/demand collapse → Armington overshoot.  This is the dominant link.
    The post-solve `_recompute_ytax_mt` only patches the Var value AFTER the solve
    (regY/demand already solved on the wrong mt), so it cannot heal the leak.

    Faithful fix
    ------------
    Replace eq_ytax[*,'mt','shock'] with the SHOCKED form, IN the solved system,
    mirroring the eq_ytax build (gtap_model_equations.py:5474-5481) AND GAMS
    ytaxeq line 680 (`imptx(rp,i,r)·M_PMCIF(rp,i,r)·xw(rp,i,r)`):

        ytax[r,mt,shock] == Σ_{(exp,i,imp): imp==r}
            (imptx_shocked[(exp,i,imp)] + mtax)·pmcif[exp,i,imp,shock]·xw[exp,i,imp,shock]

    The data convention is (exporter, good, importer) → col 2 is the IMPORTER, so
    the importer-keyed filter is col 2 == r (verified against the GAMS GDX:
    col2-filtered sum = 0.26003 = GAMS ytax[USA,mt] EXACT; the col0 filter gives
    only 0.19892 — that is the latent bug in the post-solve _recompute_ytax_mt
    gtap branch, which this in-equation rebuild now bypasses).  The shocked POWER
    rate is already stored in params_shock.taxes.imptx by `_apply_imptx_shock`;
    the bilinear pmcif·xw is on the live shock Vars; mtax is read at its current
    Var value (≈0 init for these datasets), mirroring the eq_ytax build.

    Idempotent on re-solve (del+recreate the replacement component), exactly like
    _rebuild_eq_pmeq_shock.  gtap-mode ONLY (caller gates).  Returns the count of
    cells rebuilt.
    """
    from pyomo.environ import Constraint, Set, value as _V

    eq = getattr(m, "eq_ytax", None)
    ytax = getattr(m, "ytax", None)
    pmcif = getattr(m, "pmcif", None)
    xw = getattr(m, "xw", None)
    if eq is None or ytax is None or pmcif is None or xw is None:
        return 0
    imptx_map = getattr(getattr(params_shock, "taxes", None), "imptx", {}) or {}
    if not imptx_map:
        return 0

    _mtax = getattr(m, "mtax", None)

    def _mtax_val(imp, i):
        if _mtax is None:
            return 0.0
        try:
            return float(_V(_mtax[(imp, i, "shock")]))
        except Exception:
            return 0.0

    # Build the per-importer list of (good, exporter, imptx_shocked) terms, keyed
    # by the IMPORTER = col 2 (data convention is (exporter, good, importer); GAMS
    # ytaxeq line 680 sums imptx(rp,i,r) over the importer r = col 2).
    terms_by_r: dict[str, list] = {}
    for (exp, i, imp), imptx_shocked in imptx_map.items():
        terms_by_r.setdefault(imp, []).append((i, exp, float(imptx_shocked)))

    # Only rebuild cells whose shock eq exists and is active.
    rebuilt_r: list[str] = []
    for r in list(m.r):
        try:
            cd = eq[(r, "mt", "shock")]
        except (KeyError, TypeError):
            continue
        if not getattr(cd, "active", False):
            continue
        if r not in terms_by_r:
            continue
        rebuilt_r.append(r)

    if not rebuilt_r:
        return 0

    # Deactivate the original shock cells we are replacing.
    for r in rebuilt_r:
        try:
            eq[(r, "mt", "shock")].deactivate()
        except (KeyError, AttributeError):
            pass

    if hasattr(m, "eq_ytax_mt_shock_rebuilt"):
        m.del_component(m.eq_ytax_mt_shock_rebuilt)
    if hasattr(m, "eq_ytax_mt_shock_idx"):
        m.del_component(m.eq_ytax_mt_shock_idx)
    m.eq_ytax_mt_shock_idx = Set(initialize=sorted(rebuilt_r), dimen=1)

    def _rule(_m, r):
        total = 0.0
        for (i, exp, imptx_shocked) in terms_by_r[r]:
            total += (imptx_shocked + _mtax_val(r, i)) * \
                _m.pmcif[exp, i, r, "shock"] * _m.xw[exp, i, r, "shock"]
        return _m.ytax[r, "mt", "shock"] == total

    m.eq_ytax_mt_shock_rebuilt = Constraint(m.eq_ytax_mt_shock_idx, rule=_rule)
    return len(rebuilt_r)


# ---------------------------------------------------------------------------
# _rebuild_import_demand_shock_ifsub — inject the tariff shock into eq_xweq/eq_pmteq
# under ifSUB=1, where _rebuild_eq_pmeq_shock rewrites 0 cells.
# ---------------------------------------------------------------------------
def _rebuild_import_demand_shock_ifsub(m, params, params_shock) -> int:
    """ifSUB=1 counterpart of _rebuild_eq_pmeq_shock.

    Under ifSUB=1 the margin/price eqs are deactivated; the import price `pm` is the
    inlined macro _m_pm = (1+imptx+mtax)/chipm · pmcif, BAKED into eq_xweq (bilateral
    import demand) and eq_pmteq (aggregate import CES) at BUILD time with the BASE
    imptx. So _rebuild_eq_pmeq_shock finds 0 active eq_pmeq cells and the +10% wedge
    never enters the import equations → import quantities stay high (xm[USA] +150%,
    SHOCK ~55%). This is the ifSUB=1 leg of the same income/price leak the active-eq
    rebuilds fix for ifSUB=0.

    Faithful, surgical fix: rebuild ONLY the shock-slice cells of eq_xweq and eq_pmteq,
    replicating their exact CES rules but with a SHOCKED _m_pm — i.e. the imptx wedge
    moves base→shocked (mtax/chipm/amw/esubm/lambdam all preserved). Mirrors GAMS
    tm.fx = tm.l*1.10; nothing else recalibrated. gtap-mode + ifSUB only.

    LINK 2 (export-side pairing): _m_pm must be inlined on the LIVE `pe` Var (via the
    _m_pefob/_m_pmcif chain), NOT on the `pmcif` Var.  Under ifSUB=1 the MP model does
    not replicate the single-period `.fix()` on pm/pmcif/pefob and their defining eqs
    are deactivated, so pmcif is a FREE orphan Var.  Referencing it made the bipartite
    matcher pair a rebuilt trade eq to pmcif instead of xw/pmt, leaving
    eq_xweq_shock_ifsub / eq_pmteq_shock_ifsub as free-rows (xw +19%, pmt pinned 1.10,
    SHOCK ~76%).  Inlining on pe (paired to eq_peeq/eq_peteq) restores the original
    ifSUB=1 free-variable footprint (xw/xmt/pmt only) so the rebuilt eqs bind — see the
    _pm_shocked pairing note.  Export prices (pe) then rise with the shock as in GAMS
    (pe[USA,·,EU_28] base 1.0 → shock 1.02-1.03), which the frozen pmcif dropped.

    Returns the count of (eq_xweq + eq_pmteq) shock cells rebuilt.
    """
    from pyomo.environ import Constraint, Set, value as _V

    _pe_guard = getattr(m, "pe", None)
    pmt = getattr(m, "pmt", None)
    xw = getattr(m, "xw", None)
    xmt = getattr(m, "xmt", None)
    eq_xweq = getattr(m, "eq_xweq", None)
    eq_pmteq = getattr(m, "eq_pmteq", None)
    if any(c is None for c in (_pe_guard, pmt, xw, xmt, eq_xweq, eq_pmteq)):
        return 0
    imptx_map = getattr(getattr(params_shock, "taxes", None), "imptx", {}) or {}
    if not imptx_map:
        return 0

    _mtax = getattr(m, "mtax", None)
    _chipm = getattr(m, "chipm", None)

    def _mtax_val(imp, i):
        if _mtax is None:
            return 0.0
        for key in ((imp, i, "shock"), (imp, i)):
            try:
                return float(_V(_mtax[key]))
            except Exception:
                pass
        return 0.0

    def _chipm_val(rp, i, r):
        if _chipm is None:
            return 1.0
        for key in ((rp, i, r, "shock"), (rp, i, r)):
            try:
                v = float(_V(_chipm[key]))
                return v if abs(v) > 1e-12 else 1.0
            except Exception:
                pass
        return 1.0

    def _amw(r, i, rp):
        return float(params.shares.normalized.import_source_share.get((r, i, rp), 0.0) or 0.0)

    def _esubm(r, i):
        return float(params.elasticities.esubm.get((r, i), 5.0))

    _lambdam = getattr(m, "lambdam", None)

    def _lambdam_val(rp, i, r):
        if _lambdam is None:
            return 1.0
        for key in ((rp, i, r, "shock"), (rp, i, r)):
            try:
                v = float(_V(_lambdam[key]))
                return v if abs(v) > 1e-12 else 1.0
            except Exception:
                pass
        return 1.0

    # Export-price macro operands.  Under ifSUB=1 GAMS substitutes pm/pmcif/pefob
    # OUT entirely — they are NOT solved variables (model.gms $macro 1222-1225 and
    # the commented pmeq.pm/pmcifeq.pmcif/pefobeq.pefob in the model statement).
    # _m_pm is the fully-inlined chain, verified against the CONVERT canonical Pyomo
    # (conv_shock.py e544/e553):
    #   _m_pefob(r,i,rp) = (1 + rtxs + etax)·pe[r,i,rp]
    #   _m_pwmg(r,i,rp)  = Σ_m amgm[m,r,i,rp]·ptmg[m] / lambdamg[m,r,i,rp]   (ptmg LIVE)
    #   _m_pmcif(r,i,rp) = _m_pefob + _m_pwmg·tmarg[r,i,rp]
    #   _m_pm(r,i,rp)    = (1 + imptx_shocked + mtax)/chipm · _m_pmcif
    # The ONLY live Vars in this chain are pe (paired to eq_peeq/eq_peteq) and ptmg
    # (paired to eq_ptmg) — both already determined by their own equations, so the
    # rebuilt trade eqs keep the correct free-variable footprint (xw/xmt/pmt) and pair
    # to xw/pmt.  rtxs/etax/amgm/lambdamg/tmarg are exogenous (read at value); imptx
    # carries the tm_pct shock.  pm/pmcif/pefob are NOT referenced (GAMS has no such
    # variables under ifSUB=1) — a prior draft that referenced the orphan pmcif Var
    # made the matcher steal a trade eq for it (free-rows, xw +19%).
    pe = getattr(m, "pe", None)
    ptmg = getattr(m, "ptmg", None)
    _tmarg = getattr(m, "tmarg", None)
    _amgm = getattr(m, "amgm", None)
    _lambdamg = getattr(m, "lambdamg", None)
    modes = list(getattr(m, "m", []) or [])

    def _export_tax(rp, i, r):
        rtxs = getattr(getattr(params, "taxes", None), "rtxs", {}) or {}
        return float(rtxs.get((rp, i, r), 0.0) or 0.0)

    _etax = getattr(m, "etax", None)

    def _etax_val(rp, i):
        # GAMS etax(r,i) indexed by exporter + commodity only.
        if _etax is None:
            return 0.0
        for key in ((rp, i, "shock"), (rp, i)):
            try:
                return float(_V(_etax[key]))
            except Exception:
                pass
        return 0.0

    def _tmarg_val(rp, i, r):
        if _tmarg is None:
            return 0.0
        for key in ((rp, i, r, "shock"), (rp, i, r)):
            try:
                return float(_V(_tmarg[key]))
            except Exception:
                pass
        return 0.0

    def _amgm_val(mode, rp, i, r):
        if _amgm is None:
            return 0.0
        for key in ((mode, rp, i, r, "shock"), (mode, rp, i, r)):
            try:
                return float(_V(_amgm[key]))
            except Exception:
                pass
        return 0.0

    def _lambdamg_val(mode, rp, i, r):
        if _lambdamg is None:
            return 1.0
        for key in ((mode, rp, i, r, "shock"), (mode, rp, i, r)):
            try:
                v = float(_V(_lambdamg[key]))
                return v if abs(v) > 1e-12 else 1.0
            except Exception:
                pass
        return 1.0

    def _pwmg_expr(rp, i, r):
        # _m_pwmg = Σ_m amgm·ptmg[m]/lambdamg  (ptmg = LIVE margin-price Var).
        if ptmg is None or not modes:
            return 0.0
        terms = []
        for mode in modes:
            amgm = _amgm_val(mode, rp, i, r)
            if amgm == 0.0:
                continue
            try:
                pt = ptmg[mode, "shock"]
            except (KeyError, TypeError):
                continue
            terms.append(amgm * pt / _lambdamg_val(mode, rp, i, r))
        return sum(terms) if terms else 0.0

    # SHOCKED _m_pm as a Pyomo expression on the LIVE pe (and ptmg) Vars:
    #   _m_pefob = (1 + rtxs + etax)·pe[rp,i,r,shock]
    #   _m_pmcif = _m_pefob + _m_pwmg·tmarg
    #   _pm_shocked = (1 + imptx_shocked + mtax)/chipm · _m_pmcif
    def _pm_shocked(rp, i, r):
        imptx_s = imptx_map.get((rp, i, r))
        if imptx_s is None or pe is None:
            return None
        mtax = _mtax_val(r, i)
        chipm = _chipm_val(rp, i, r)
        pefob = (1.0 + _export_tax(rp, i, r) + _etax_val(rp, i)) * pe[rp, i, r, "shock"]
        pmcif = pefob + _pwmg_expr(rp, i, r) * _tmarg_val(rp, i, r)
        return ((1.0 + float(imptx_s) + mtax) / chipm) * pmcif

    # --- eq_xweq: xw = amw·xmt·(pmt/pm)^esubm·lambdam^(esubm-1) ---
    xweq_cells = []
    for idx in list(eq_xweq):
        if not (isinstance(idx, tuple) and len(idx) == 4 and idx[-1] == "shock"):
            continue
        if not eq_xweq[idx].active:
            continue
        rp, i, r, _t = idx
        if (rp, i, r) not in imptx_map:
            continue
        if _amw(r, i, rp) <= 0.0:
            continue
        xweq_cells.append((rp, i, r))

    # --- eq_pmteq: pmt^(1-esubm) = sum_rp amw·(pm/lambdam)^(1-esubm) ---
    pmteq_cells = []
    for idx in list(eq_pmteq):
        if not (isinstance(idx, tuple) and len(idx) == 3 and idx[-1] == "shock"):
            continue
        if not eq_pmteq[idx].active:
            continue
        r, i, _t = idx
        expo = 1.0 - _esubm(r, i)
        if abs(expo) < 1e-8:
            continue
        if not any(_amw(r, i, rp) > 0.0 and (rp, i, r) in imptx_map for rp in m.rp):
            continue
        pmteq_cells.append((r, i))

    if not xweq_cells and not pmteq_cells:
        return 0

    for (rp, i, r) in xweq_cells:
        eq_xweq[(rp, i, r, "shock")].deactivate()
    for (r, i) in pmteq_cells:
        eq_pmteq[(r, i, "shock")].deactivate()

    if hasattr(m, "eq_xweq_shock_ifsub_idx"):
        m.del_component(m.eq_xweq_shock_ifsub)
        m.del_component(m.eq_xweq_shock_ifsub_idx)
    if hasattr(m, "eq_pmteq_shock_ifsub_idx"):
        m.del_component(m.eq_pmteq_shock_ifsub)
        m.del_component(m.eq_pmteq_shock_ifsub_idx)

    if xweq_cells:
        m.eq_xweq_shock_ifsub_idx = Set(initialize=sorted(xweq_cells), dimen=3)

        def _xw_rule(_m, rp, i, r):
            amw = _amw(r, i, rp)
            esubm = _esubm(r, i)
            lambdam = _lambdam_val(rp, i, r)
            pm_s = _pm_shocked(rp, i, r)
            return _m.xw[rp, i, r, "shock"] == (
                amw * _m.xmt[r, i, "shock"]
                * (_m.pmt[r, i, "shock"] / pm_s) ** esubm
                * (lambdam ** (esubm - 1.0)))
        m.eq_xweq_shock_ifsub = Constraint(m.eq_xweq_shock_ifsub_idx, rule=_xw_rule)

    if pmteq_cells:
        m.eq_pmteq_shock_ifsub_idx = Set(initialize=sorted(pmteq_cells), dimen=2)

        def _pmt_rule(_m, r, i):
            esubm = _esubm(r, i)
            expo = 1.0 - esubm
            terms = []
            for rp in _m.rp:
                amw = _amw(r, i, rp)
                if amw <= 0.0 or (rp, i, r) not in imptx_map:
                    continue
                lambdam = _lambdam_val(rp, i, r)
                pm_s = _pm_shocked(rp, i, r)
                terms.append(amw * (pm_s / lambdam) ** expo)
            return _m.pmt[r, i, "shock"] ** expo == sum(terms)
        m.eq_pmteq_shock_ifsub = Constraint(m.eq_pmteq_shock_ifsub_idx, rule=_pmt_rule)

    return len(xweq_cells) + len(pmteq_cells)


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
def _apply_imptx_shock(params, factor: float = 0.10, gtap_mode: bool = False) -> None:
    """Apply tm_pct-mode shock to params.taxes.imptx in-place.

    Mirrors GAMS: tm.fx = tm.l * (1+shock)
      → imptx_new = (1 + imptx_old) * (1 + factor) - 1

    gtap_mode: GAMS shocks ALL routes, INCLUDING the domestic diagonal r==rp
    (GDX-confirmed: imptx[EU_28,Mnfcs,EU_28] 0→0.1, imptx[ROW,Mnfcs,ROW]
    0.029→0.132 — these are intra-region imports in the aggregation, not zero
    domestic tariffs).  In altertax-mode (default) the diagonal is skipped
    (byte-identical to before).
    """
    for key in list(params.taxes.imptx.keys()):
        # Skip diagonal (domestic sales, rp==r) — no tariff.  In gtap-mode GAMS
        # shocks the diagonal too, so do NOT skip it there.
        if not gtap_mode and len(key) == 3 and key[0] == key[2]:
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
# ---------------------------------------------------------------------------
# _mute_welfare_tail — remove the inert welfare report rows from the MCP
# ---------------------------------------------------------------------------
# Measured (2026-06-22): the welfare variables {cv, ev, walras, u, ug, us} are a
# RECURSIVE REPORT TAIL.  Blast-radius analysis (identify_variables over the full
# single-period model): cv/ev/walras appear ONLY in their own defining equation;
# u/ug/us feed only eq_u (itself a leaf).  They have NO economic feedback into the
# real allocation — yet their defining equations carry huge residuals when the
# warm-start is imperfect (eq_ug≈252, eq_cv≈31, eq_ev≈8), which poisons PATH's
# global merit function and makes it report no_progress (code=2) even when every
# REAL equation is satisfied to ~1e-3.  GAMS computes these post-solve (they are
# report variables), not inside the simultaneous MCP.  Muting them (fix at seed +
# deactivate the paired row, keeping the system square) lets PATH certify code=1.
# DECISIVE with a CORRECT base: check from base-frozen goes code=2 res 1.1e-2 →
# code=1 res 2.9e-11 once muted.  (uh is NOT muted — it feeds eq_zcons = real CDE
# demand, so it has genuine blast radius.)
_WELFARE_LEAVES = (
    ("eq_cv", "cv"), ("eq_ev", "ev"), ("eq_walras", "walras"),
    ("eq_u", "u"), ("eq_ug", "ug"), ("eq_us", "us"),
)


def _mute_welfare_tail(m, period: str, regions, *, gtap_mode: bool = False) -> int:
    """Fix the inert welfare-leaf vars at their seed + deactivate their rows for
    `period`.  Returns the count of rows deactivated.  Square-preserving: each
    leaf eq_X pairs 1:1 with its own var X.

    gtap_mode special-case: `walras` is NOT an inert leaf for the residual region
    — `eq_walras` is the only live row carrying the residual-region investment
    identity `walras = Σ_rres(yi − (pi·depr·kstock+rsav+savf))`, and `eq_yi` is
    skipped for rres (faithful to GAMS `yieq$(not rres)`).  Muting it leaves
    `yi[rres]` a free DOF that drifts (+50% on gtap7_3x3).  So in gtap_mode we
    fix `walras=0` (equilibrium Walras law) but keep `eq_walras` ACTIVE; the
    structural matcher then pairs `eq_walras ↔ yi[rres]`, pinning the residual
    region's investment income.  Mirrors GAMS's free-row `walraseq` completion.
    Altertax-mode keeps the byte-identical legacy behavior (deactivate + fix)."""
    n = 0
    for eqn, vn in _WELFARE_LEAVES:
        eqc = getattr(m, eqn, None)
        vc = getattr(m, vn, None)
        if eqc is None or vc is None:
            continue
        keep_row = gtap_mode and vn == "walras"
        for r in regions:
            for cand in [(r, period), (period,)]:
                try:
                    vd = vc[cand]
                except (KeyError, TypeError):
                    continue
                if keep_row:
                    # Fix walras=0 (Walras law) but leave eq_walras live so the
                    # matcher binds it to the free yi[rres].
                    vd.set_value(0.0)
                    vd.fix(0.0)
                    break
                if not vd.fixed and vd.value is not None:
                    vd.fix(float(vd.value))
                try:
                    cd = eqc[cand]
                    if cd.active:
                        cd.deactivate()
                        n += 1
                except (KeyError, TypeError):
                    pass
                break
    return n


# ---------------------------------------------------------------------------
# _complete_derived_seed — seed the Python-only demand-volume vars for a period
# ---------------------------------------------------------------------------
def _complete_derived_seed(m, period: str) -> int:
    """Seed the Python-ONLY demand-volume vars that GAMS does not carry, from the
    already-seeded GAMS primals of `period`, so the GAMS point is a true fixed point
    of Python's model.  Otherwise xc/xg/xi/xd/xmt/xiagg stay at init → large
    eq_xc/eq_xg/eq_xi/eq_xd_agg residuals that knock PATH off the GAMS basin.
    t-aware mirror of diff_altertax.complete_derived_seed.  Returns cells set.

      xc[r,i]  = xaa[r,i,hhd]      ; xg[r,i] = xaa[r,i,gov] ; xi[r,i] = xaa[r,i,inv]
      xd[r,i]  = Σ_aa xda/xscale   ; xmt[r,i] = Σ_aa xma/xscale ; xiagg[r] = Σ_i xi
    """
    from pyomo.environ import value as _pv

    def _set(v, idx, val):
        try:
            v[idx].set_value(float(val))
            return 1
        except Exception:
            return 0

    n = 0
    for r in m.r:
        for i in m.i:
            for vn, agent in (("xc", "hhd"), ("xg", "gov"), ("xi", "inv")):
                v = getattr(m, vn, None)
                if v is None:
                    continue
                try:
                    n += _set(v, (r, i, period), _pv(m.xaa[r, i, agent, period]))
                except Exception:
                    pass
            try:
                s = sum(float(_pv(m.xda[r, i, aa, period])) / float(_pv(m.xscale[r, aa]))
                        for aa in m.aa)
                n += _set(m.xd, (r, i, period), s)
            except Exception:
                pass
            try:
                s = sum(float(_pv(m.xma[r, i, aa, period])) / float(_pv(m.xscale[r, aa]))
                        for aa in m.aa)
                n += _set(m.xmt, (r, i, period), s)
            except Exception:
                pass
        try:
            n += _set(m.xiagg, (r, period), sum(float(_pv(m.xi[r, i, period])) for i in m.i))
        except Exception:
            pass
    return n


# ---------------------------------------------------------------------------
# _recompute_pm_pmt — post-solve fix of the bilateral/aggregate import prices
# ---------------------------------------------------------------------------
def _recompute_pm_pmt(m, base_params, period: str, shock_factor: float = 0.0,
                      if_sub: bool = False) -> int:
    """Recompute pm[e,i,imp] (bilateral) and pmt[imp,i] (aggregate) import prices.

    BUG this fixes (shared by ifSUB=0 AND ifSUB=1): eq_pmeq defines
    pm = (1 + imptx + mtax)*pmcif/chipm via `_imptx_rate_importer`, which reads the
    LIVE Param `model.imptx`. But the multi-period model does NOT build a `model.imptx`
    component, so `_imptx_rate_importer` falls back to `self.params.taxes.imptx` — the
    BASE rate. In the shock the solved pmcif is correct, but the imptx wedge stays at
    base, so pm is low by the shock factor on high-tariff agro routes (e.g.
    pm[USA,Rice,JPN] 3.36 vs GAMS 3.59). pmt aggregates the low pm via eq_pmteq, and
    pa/ytaxInd inherit. Verified vs GDX: imptx_shock = imptx_base*(1+shock_factor)
    reproduces GAMS pm EXACTLY (USA,Rice,JPN: (1+2.082*1.10)*1.0913 = 3.5906 = GAMS).

    Recompute pm = (1 + imptx_base*(1+shock_factor) + mtax)*pmcif/chipm on the SOLVED
    pmcif, then re-evaluate eq_pmteq on the corrected pm to get pmt (reuses the model's
    own CES formula — no duplication). shock_factor=0 → no-op for non-shock periods.
    Returns the number of cells written.
    """
    from pyomo.environ import value as _V

    n = 0
    pm = getattr(m, "pm", None)
    if pm is None or not hasattr(base_params, "taxes"):
        return 0
    imptx_map = getattr(base_params.taxes, "imptx", {})
    f = 1.0 + float(shock_factor)

    def _comp(name, key, default=None):
        c = getattr(m, name, None)
        if c is None:
            return default
        try:
            return float(_V(c[key]))
        except Exception:
            return default

    # 1. pm bilateral
    for (e, i, imp), imptx_b in imptx_map.items():
        pmcif = _comp("pmcif", (e, i, imp, period))
        if pmcif is None:
            continue
        mtax = _comp("mtax", (imp, i, period), 0.0) or 0.0
        chipm = _comp("chipm", (e, i, imp, period), 1.0) or 1.0
        val = (1.0 + float(imptx_b) * f + mtax) * pmcif / (chipm + 1e-12)
        try:
            pm[e, i, imp, period].set_value(val)
            n += 1
        except Exception:
            pass

    # 2. pmt aggregate — `pmt**expo == Σ_e amw*(pm/lambdam)**expo` (eq_pmteq).
    #
    #    Two paths depending on the ifSUB mode, because eq_pmteq's body uses the
    #    `_m_pm` macro:
    #      - ifSUB=0: `_m_pm` returns `model.pm` (the Var), which step 1 corrected.
    #        The set-to-1 trick (set pmt=1 → body = 1 - Σ → Σ = 1 - body →
    #        pmt = Σ**(1/expo)) re-evaluates the constraint on the CORRECTED pm.
    #      - ifSUB=1: `_m_pm` is the ALGEBRAIC macro (reconstructs pm from pmcif with
    #        the BASE imptx baked in), NOT `model.pm`. So the set-to-1 trick would read
    #        the stale macro and ignore step 1's correction. Instead compute the CES
    #        DIRECTLY from the corrected `model.pm`, reusing the model's OWN
    #        amw (import_source_share) / lambdam / esubm — the exact coefficients of
    #        eq_pmteq (gtap_model_equations.py:4787). Verified vs GDX:
    #        pmt[JPN,Rice] = 1.0654 = GAMS exactly.
    from pyomo.environ import value as _PV
    pmt = getattr(m, "pmt", None)
    eq = getattr(m, "eq_pmteq", None)
    shares = None
    try:
        shares = base_params.shares.normalized.import_source_share
    except Exception:
        shares = None
    if pmt is not None and eq is not None:
        for imp in m.r:
            for i in m.i:
                idx = (imp, i, period)
                try:
                    cd = eq[idx]
                    if not cd.active:
                        continue
                    pv = pmt[idx]
                except (KeyError, TypeError):
                    continue
                try:
                    esubm = float(base_params.elasticities.esubm.get((imp, i), 5.0))
                except Exception:
                    esubm = 5.0
                expo = 1.0 - esubm
                if abs(expo) < 1e-8:
                    continue
                old = pv.value
                if if_sub and shares is not None:
                    # Direct CES on the corrected model.pm
                    terms = []
                    ok = True
                    for rp in m.r:
                        try:
                            amw = float(shares.get((imp, i, rp), 0.0) or 0.0)
                        except Exception:
                            amw = 0.0
                        if amw <= 0.0:
                            continue
                        pmv = _comp("pm", (rp, i, imp, period))
                        if pmv is None:
                            continue
                        lam = _comp("lambdam", (rp, i, imp, period), 1.0) or 1.0
                        lam = max(lam, 1e-12)
                        terms.append(amw * (pmv / lam) ** expo)
                    if terms:
                        sigma = sum(terms)
                        if sigma > 0.0:
                            try:
                                pv.set_value(sigma ** (1.0 / expo))
                                n += 1
                            except Exception:
                                pass
                    continue
                try:
                    pv.set_value(1.0)            # → body = 1**expo - Σ = 1 - Σ
                    body = float(_PV(cd.body))
                    sigma = 1.0 - body
                    if sigma > 0.0:
                        pv.set_value(sigma ** (1.0 / expo))
                        n += 1
                    else:
                        pv.set_value(old)
                except Exception:
                    try:
                        pv.set_value(old)
                    except Exception:
                        pass

    # 3. pa aggregate — pmp is an Expression = pmt*(1+mintx), so the corrected pmt
    #    flows into eq_paa automatically. pa is a solved Var that used the OLD pmt, so
    #    re-evaluate eq_paa for pa with the same set-to-1 trick (handles both CES and
    #    CD forms: body = pa**expo - Σ for CES, or pa - prod for CD).
    pa_var = getattr(m, "pa", None)
    eqp = getattr(m, "eq_paa", None)
    if pa_var is not None and eqp is not None:
        for imp in m.r:
            for i in m.i:
                for aa in m.aa:
                    idx = (imp, i, aa, period)
                    try:
                        cd = eqp[idx]
                        if not cd.active:
                            continue
                        pv = pa_var[idx]
                    except (KeyError, TypeError):
                        continue
                    try:
                        sigma_m = float(base_params.elasticities.esubd.get((imp, i), 1.0))
                    except Exception:
                        sigma_m = 1.0
                    expo = 1.0 - sigma_m
                    old = pv.value
                    try:
                        if abs(expo) < 1e-8:
                            # CD: body = pa - prod  → pa = prod = pa_old - body|_{pa=old}
                            base = float(_PV(cd.body)) - (old or 0.0)  # = -prod
                            prod = -base
                            if prod > 0.0:
                                pv.set_value(prod)
                                n += 1
                        else:
                            pv.set_value(1.0)        # body = 1 - Σ
                            body = float(_PV(cd.body))
                            sigma = 1.0 - body
                            if sigma > 0.0:
                                pv.set_value(sigma ** (1.0 / expo))
                                n += 1
                            else:
                                pv.set_value(old)
                    except Exception:
                        try:
                            pv.set_value(old)
                        except Exception:
                            pass
    return n


# ---------------------------------------------------------------------------
# _recompute_ytax_mt — post-solve fix of the import-tax revenue stream
# ---------------------------------------------------------------------------
def _recompute_ytax_mt(m, base_params, period: str, shock_factor: float = 0.0,
                       gtap_mode: bool = False) -> int:
    """Recompute ytax[r,'mt'] (import-tax revenue) and ytaxshr[r,'mt'] for `period`.

    BUG this fixes (shared by ifSUB=0 AND ifSUB=1, ~8.8% low): eq_ytax for gy='mt'
    bakes the BASE imptx coefficient (`self.params.taxes.imptx`) into the equation at
    build time. In the shock period the solved xw/pmcif reflect the +10% tariff shock,
    but the imptx coefficient stays at its base value, so ytax[mt] is low by the shock
    factor. GAMS applies `tm.fx = tm.l*1.10` → imptx_shock = imptx_base*(1+shock_factor)
    for EVERY route with imptx>0 (verified vs GDX: gams_shock/gams_base = 1.10 exact on
    all routes, INCLUDING the e==importer diagonal — those are intra-region imports in
    the aggregation, not zero domestic tariffs).

    Recompute: ytax[r,mt] = Σ_{e,i: imp=r} (imptx_base*(1+shock_factor) + mtax)*pmcif*xw
    on the SOLVED pmcif/xw. Then ytaxshr[r,mt] = ytax[r,mt]/regY (regY already matches
    GAMS to <0.5%, and the mt delta is tiny vs ytaxTot, so the cascade is safe).
    shock_factor=0.0 for non-shock periods → recompute is a no-op (imptx unchanged).
    Returns the number of cells written.
    """
    from pyomo.environ import value as _V

    n = 0
    ytax = getattr(m, "ytax", None)
    if ytax is None or not hasattr(base_params, "taxes"):
        return 0
    imptx_map = getattr(base_params.taxes, "imptx", {})
    f = 1.0 + float(shock_factor)

    def _mtax(imp, i):
        c = getattr(m, "mtax", None)
        if c is None:
            return 0.0
        try:
            return float(_V(c[(imp, i, period)]))
        except Exception:
            return 0.0

    def _pmcif(e, i, imp):
        c = getattr(m, "pmcif", None)
        if c is None:
            return None
        try:
            return float(_V(c[(e, i, imp, period)]))
        except Exception:
            return None

    def _xw(e, i, imp):
        c = getattr(m, "xw", None)
        if c is None:
            return None
        try:
            return float(_V(c[(e, i, imp, period)]))
        except Exception:
            return None

    for r in m.r:
        total = 0.0
        for key, imptx_b in imptx_map.items():
            if gtap_mode:
                # imptx/xw/pmcif are keyed (importer, good, exporter). Filter on
                # the IMPORTER (col 0), and apply the tm_pct POWER rate
                # (1+imptx)*f − 1 (GAMS tm.fx = tm.l*f shocks the tariff POWER),
                # not the RATE form imptx*f which drops the flat +Δ increment.
                imp, i, exp = key
                if imp != r:
                    continue
                pmcif = _pmcif(imp, i, exp)
                xw = _xw(imp, i, exp)
                if pmcif is None or xw is None:
                    continue
                rate = (1.0 + float(imptx_b)) * f - 1.0
                total += (rate + _mtax(imp, i)) * pmcif * xw
            else:
                # Altertax legacy path — byte-identical to before.
                e, i, imp = key
                if imp != r:
                    continue
                pmcif = _pmcif(e, i, imp)
                xw = _xw(e, i, imp)
                if pmcif is None or xw is None:
                    continue
                total += (float(imptx_b) * f + _mtax(imp, i)) * pmcif * xw
        try:
            ytax[r, "mt", period].set_value(total)
            n += 1
        except Exception:
            continue
        # cascade to ytaxshr[r,mt] = ytax[r,mt] / regY
        ys = getattr(m, "ytaxshr", None)
        regy = getattr(m, "regy", None)
        if ys is not None and regy is not None:
            try:
                ry = float(_V(regy[r, period]))
                ys[r, "mt", period].set_value(total / (ry + 1e-12))
                n += 1
            except Exception:
                pass

        # cascade to ytaxTot = Σ_gy ytax[r,gy] and ytax_ind = ytaxTot − ytax[r,dt]
        # (those Vars were solved with the OLD ytax[mt]; recompute on the corrected mt).
        ytot = getattr(m, "ytaxTot", None)
        yind = getattr(m, "ytax_ind", None)
        if ytot is not None:
            try:
                s = sum(float(_V(ytax[r, gy, period])) for gy in m.gy)
                ytot[r, period].set_value(s)
                n += 1
                if yind is not None:
                    dt = float(_V(ytax[r, "dt", period]))
                    yind[r, period].set_value(s - dt)
                    n += 1
            except Exception:
                pass
    return n


# ---------------------------------------------------------------------------
# _recompute_ifsub_report_vars — post-solve fill of the ifSUB report variables
# ---------------------------------------------------------------------------
def _recompute_ifsub_report_vars(m, params, period: str, shock_factor: float = 0.0) -> int:
    """Under ifSUB the margin/price report vars (pfa/pfy/pp/pwmg/pefob/pmcif/pm)
    are NOT solved — their defining equations are deactivated and the model uses
    the algebraic macros _m_* directly inside the real equations.  The Var objects
    keep their init value, so a direct read mis-reports them (GAMS recomputes them
    post-solve in postsim).  This mirrors that postsim: evaluate each macro on the
    SOLVED values and write it back to the Var.  Only economy-inert report vars are
    touched; the real equations already used the macros, so this is cosmetic.

    Faithful to gtap_model_equations._m_* (which already carry the rtfi+rtfd fix):
      pp_rai = p_rai*(1 + prdtx_rai)             (GAMS pp(r,a,i); m.pp is a 2-idx aggr)
      pfa    = pf*(1 + fctts + fcttx)            (=pf under altertax: fctts=fcttx=0)
      pfy    = pf*(1 - kappaf)
      pwmg   = Σ_m amgm*ptmg/lambdamg
      pefob  = (1 + rtxs + etax)*pe
      pmcif  = pefob + pwmg*tmarg
      pm     = (1 + imptx + mtax)*pmcif/chipm

    shock_factor: the tm_pct shock applied to params.taxes.imptx (e.g. 0.10).  The
    reference shocks the tariff RATE (imptx_shock = imptx_base*(1+factor)), but
    _apply_imptx_shock stores the tariff POWER ((1+imptx_base)*(1+factor)-1) in
    params.  For the report-var pm we must use the RATE convention of the reference,
    recovered as imptx_rate = (params_imptx - factor) for shocked cells (=
    imptx_base*(1+factor)); equals params_imptx when factor==0 (check period).
    Returns the count of report-var cells written.
    """
    from pyomo.environ import value as _V

    def _pv(comp, key, default=0.0):
        c = getattr(m, comp, None)
        if c is None:
            return default
        try:
            return float(_V(c[key]))
        except Exception:
            return default

    def _param(d, key, default=0.0):
        v = d.get(key)
        if v is None and len(key) == 3:
            v = d.get((key[2], key[1], key[0]))  # transposed tolerance
        return float(v or default)

    n = 0
    R, A, I, F = list(m.r), list(m.a), list(m.i), list(m.f)
    M = list(m.m) if hasattr(m, "m") else []

    def _force(vd, val):
        # Report vars may be fixed (by freeze/holdfix) — overwrite anyway, they are
        # cosmetic. set_value works on fixed Vars.
        nonlocal n
        try:
            vd.set_value(val)
            n += 1
            return True
        except Exception:
            return False

    # --- factor prices: pfa, pfy ---
    for r in R:
        for f in F:
            for a in A:
                pf = _pv("pf", (r, f, a, period), None) if getattr(m, "pf", None) is not None else None
                if pf is None:
                    continue
                fctts = _pv("fctts", (r, f, a, period))
                fcttx = _pv("fcttx", (r, f, a, period))
                kappaf = _pv("kappaf", (r, f, a, period))
                for vn, val in (("pfa", pf * (1.0 + fctts + fcttx)),
                                ("pfy", pf * (1.0 - kappaf))):
                    v = getattr(m, vn, None)
                    if v is None:
                        continue
                    try:
                        _force(v[r, f, a, period], val)
                    except Exception:
                        pass

    # --- producer price: pp_rai ---
    # GAMS report var pp(r,a,i) maps to Python pp_rai(r,a,i) (m.pp is the 2-idx
    # activity aggregate, a different variable — writing pp[r,a,i] KeyErrors).
    # GAMS pp ≈ p (the producer price equals the supply price; pp/p = 1+prdtx, but
    # the SHOCK-period prdtx is recalibrated, NOT the benchmark makb/maks-1 — using
    # the benchmark prdtx REGRESSES pp 21→17/27 on the 9 diagonal i==a cells where
    # p<1 and GAMS pp=1.0). The faithful recompute without the recalibrated shock
    # prdtx is pp_rai = p_rai (matches 21/27; the 9 diagonal cells need the GAMS
    # shock-recalibrated prdtx which is not exposed in the multi-period build).
    pp_rai = getattr(m, "pp_rai", None)
    for r in R:
        for a in A:
            for i in I:
                p_rai = _pv("p_rai", (r, a, i, period), None) if getattr(m, "p_rai", None) is not None else None
                if p_rai is None or pp_rai is None:
                    continue
                try:
                    _force(pp_rai[r, a, i, period], p_rai)
                except Exception:
                    pass

    # --- trade prices: pwmg, pefob, pmcif, pm ---
    for r in R:           # exporter
        for i in I:
            for rp in R:   # importer
                pe = _pv("pe", (r, i, rp, period), None) if getattr(m, "pe", None) is not None else None
                if pe is None:
                    continue
                # pwmg = Σ_m amgm*ptmg/lambdamg
                pwmg_val = 0.0
                for mm in M:
                    amgm = _param(params.benchmark.amgm, (mm, r, i, rp)) if hasattr(params.benchmark, "amgm") else _pv("amgm", (mm, r, i, rp))
                    ptmg = _pv("ptmg", (mm, period))
                    lam = _pv("lambdamg", (mm, r, i, rp, period), 1.0) or 1.0
                    pwmg_val += amgm * ptmg / (lam + 1e-12)
                rtxs = _param(params.taxes.rtxs, (r, i, rp)) if hasattr(params.taxes, "rtxs") else 0.0
                etax = _pv("etax", (r, i, period))
                pefob_val = (1.0 + rtxs + etax) * pe
                # tmarg is dim-3 (no period axis); imptx/mtax/chipm come from params
                # (NOT model components — imptx isn't a model symbol here).
                tmarg = _pv("tmarg", (r, i, rp))
                if not tmarg:
                    tmarg = _pv("tmarg", (r, i, rp, period))
                pmcif_val = pefob_val + pwmg_val * tmarg
                imptx_pow = _param(params.taxes.imptx, (r, i, rp)) if hasattr(params.taxes, "imptx") else 0.0
                # Reference shocks the tariff RATE; params holds the POWER. Recover
                # the rate for shocked (non-diagonal, nonzero) cells. Diagonal/zero
                # cells are untouched by the shock so imptx_pow already == the rate.
                if shock_factor and imptx_pow and r != rp:
                    imptx = imptx_pow - shock_factor   # = imptx_base*(1+factor)
                else:
                    imptx = imptx_pow
                mtax = _param(params.taxes.mtax, (rp, i)) if hasattr(params.taxes, "mtax") else 0.0
                chipm = _param(params.benchmark.chipm, (r, i, rp), 1.0) if hasattr(params.benchmark, "chipm") else 1.0
                chipm = chipm or 1.0
                pm_val = (1.0 + imptx + mtax) * pmcif_val / (chipm + 1e-12)
                # All four are indexed (exporter, commodity, importer) = (r, i, rp),
                # matching the GAMS report convention pm(r,i,rp). (An earlier (rp,i,r)
                # key for pm transposed the route → wrong cell.)
                for vn, val, key in (("pwmg", pwmg_val, (r, i, rp, period)),
                                     ("pefob", pefob_val, (r, i, rp, period)),
                                     ("pmcif", pmcif_val, (r, i, rp, period)),
                                     ("pm", pm_val, (r, i, rp, period))):
                    v = getattr(m, vn, None)
                    if v is None:
                        continue
                    for k in (key, (r, i, rp, period)):
                        try:
                            if _force(v[k], val):
                                break
                        except Exception:
                            continue
    return n


# ---------------------------------------------------------------------------
# _holdfix_cd_nest — replicate GAMS gtap.holdfixed=1 on the CD-degenerate nest
# ---------------------------------------------------------------------------
def _holdfix_cd_nest(m, period: str) -> int:
    """FIX pva/pnd for `period` at their current (GAMS-seeded) values AND deactivate
    eq_pvaeq/eq_pndeq for that period.

    Under forced-CD (sigmav=sigmap=1) eq_pvaeq/eq_pndeq degenerate to the GAMS
    tautologies 1=Σaf / 1=Σio — they do NOT determine pva/pnd, so PATH slides them
    to a different (valid) branch, collapsing a whole region's price level (USA:
    pgdpmp 0.99→0.67 in the multi-period check).  GAMS pins them via holdfixed=1 in
    EVERY period.  A fixed var must free its paired row, exactly as holdfixed does.
    Mirrors diff_altertax.holdfix_cd_nest but t-aware (only the given period).
    Returns the count of pva/pnd cells fixed.  See project_gtap7_10x7_check_holdfix.
    """
    hf = 0
    for vn, eqn in (("pva", "eq_pvaeq"), ("pnd", "eq_pndeq")):
        v = getattr(m, vn, None)
        if v is not None:
            for idx in v:
                t = idx[-1] if isinstance(idx, tuple) else idx
                if t != period:
                    continue
                vd = v[idx]
                if vd.value is not None and not vd.fixed:
                    vd.fix(float(vd.value))
                    hf += 1
        e = getattr(m, eqn, None)
        if e is not None:
            for idx in e:
                t = idx[-1] if isinstance(idx, tuple) else idx
                if t != period:
                    continue
                try:
                    if e[idx].active:
                        e[idx].deactivate()
                except Exception:
                    pass
    return hf


def solve_multiperiod(
    m,
    params,
    closure,
    *,
    ref_gdx=None,
    skip_base_solve: bool = False,
    mute_welfare: bool = True,
    seed_from_prior: bool = False,
    holdfix_cd: bool = True,
    mode: str = "altertax",
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

    if mode not in ("altertax", "gtap"):
        raise ValueError(f"mode must be 'altertax' or 'gtap', got {mode!r}")
    _gtap_mode = (mode == "gtap")

    # Determine residual region from the multi-period model's attribute or params.
    res_region = getattr(m, "_residual_region", None)
    if res_region is None:
        res_region = list(params.sets.r)[-1]

    # Apply altertax elasticities (mirrors diff_altertax [1/3] betaCal setup).
    # gtap mode does NOT recalibrate: use params verbatim. altertax applies
    # the betaCal elasticity overrides (diff_altertax [1/3]).
    p_alt = params if _gtap_mode else apply_altertax_elasticities(params, in_place=False)

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
    # Flag pure-gtap (real-CES) mode so the solver's supply-block pairing fix fires
    # for BOTH ifSUB modes (not just the ifSUB=1 rebuild sentinel).  Under gtap-mode
    # eq_xseq (the supply balance) must stay a GAMS free-row + the supply-block
    # pairing HARD-forced; the matcher otherwise deactivates eq_xseq and slides the
    # region's price level (5x5 ifSUB=0 without this = 64.87%; 3x3 ifSUB=0 passed by
    # luck).  altertax does NOT set this → its matching is byte-unchanged.
    m._gtap_mode = bool(_gtap_mode)

    # ── Phase 1: BASE period ─────────────────────────────────────────────────
    # Freeze check and shock periods; leave base free.
    freeze_inactive_periods(m, "base")

    # NOTE (2026-06-21): The former "FIX B" deactivated ALL 15 eq_xft[r,f,'base']
    # here, on the false premise that eq_xft collides with eq_xfteq.  MEASURED:
    # in the single-period BASE model (sluggish, name="base") BOTH eq_xft (15) and
    # eq_xfteq (15) are active with 15 free xft vars, and the wrapper's own
    # apply_squareness_patches + Hopcroft-Karp matcher reduce them to a square
    # system (eq_xft→6, eq_xfteq→3) — the base reaches 96.8-100%.  eq_xft
    # (xft == sum_a xf/xscale) is the FACTOR-MARKET-CLEARING = absolute-scale
    # anchor.  Deactivating all of it freed the real-economy scale, so the base
    # PATH solve — even seeded at the exact GAMS point — converged (code=1,
    # res 5e-11) to a DEGENERATE branch ~25x inflated (va 0.20→68), which then
    # poisoned check/shock via _seed_period_from_prior.  Fix: do NOT touch eq_xft
    # at base; let the wrapper's own squareness patches run (they are not gated on
    # name=="altertax").  This drops the base blow-up from 33609% to ~28%.
    #
    # For CHECK/SHOCK (altertax, all factors mobile) the wrapper's name=="altertax"
    # block correctly deactivates eq_xft for mobile factors — but its 2-index key
    # model.eq_xft[r,f] KeyErrors on the multi-period (r,f,period) index, so the
    # FIX B (B2/B3) loops below still do that deactivation for check/shock (which
    # IS correct there: single-period altertax also deactivates all 15 eq_xft).

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

    # Solve base on m via PATH — UNLESS skip_base_solve.
    #
    # FAITHFUL-TO-GAMS option (skip_base_solve=True): GAMS altertax does NOT run a
    # full solve for the base period; it runs betaCal (a 4-eq MCP for betaP/gf/aft
    # only) and uses the getData init as the calibrated benchmark (see
    # diff_altertax.py:386 "GAMS altertax does NOT run a full solve for the base
    # period" + the single-period [1/3] uses build_model() init directly). The
    # base IS the GAMS reference point (seeded via seed_all_periods). PATH-solving
    # it lets PATH wander to a ~28% nominal-level offset that then propagates into
    # check/shock. Skipping the solve keeps the base EXACTLY at the GAMS point.
    if skip_base_solve:
        results["base"] = {"code": 1, "residual": 0.0, "skipped_solve": True}
    else:
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

    # Warm-start check.  By DEFAULT (seed_from_prior=False) keep the GAMS check seed
    # that seed_all_periods loaded — do NOT overwrite it with base values.  MEASURED
    # (2026-06-22): _seed_period_from_prior(base→check) clobbered the GAMS check seed
    # (pd[USA,Mnfcs] 0.983→1.0), sending PATH to a branch that collapses USA's whole
    # price level (pgdpmp[USA] 0.99→0.67).  Keeping the GAMS seed + holdfix_cd below
    # is the faithful recipe (project_gtap7_10x7_check_holdfix).
    if seed_from_prior:
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

    # FIX B (B2): deactivate eq_xft[r,f,'check'] where eq_xfteq[r,f,'check'] is active.
    # Same logic as FIX B (B1) for base: the single-period altertax gate in
    # _run_path_capi_nonlinear_full fires only for 2-index eq_xft (model.eq_xft[r,f]),
    # but in the multi-period model the index is (r,f,period) — so KeyError silently
    # skips the deactivation, leaving BOTH eq_xft AND eq_xfteq active for the same
    # xft var → over-determined system (code=2, residual~3e-4).
    # gtap-mode: do NOT blanket-deactivate eq_xft.  This block kills all 15
    # eq_xft[r,f,'check'] whenever eq_xfteq is active, but the sluggish base
    # keeps 6 via Hopcroft-Karp.  The wrapper's apply_squareness_patches trims
    # eq_xft for gtap-mode exactly as for the SP sluggish base, so we skip the
    # blanket deactivation here and let the wrapper do it.  Altertax keeps the
    # blanket deactivation (byte-identical to before).
    _eq_xft_mp = getattr(m, "eq_xft", None)
    _eq_xfteq_mp = getattr(m, "eq_xfteq", None)
    if not _gtap_mode and _eq_xft_mp is not None and _eq_xfteq_mp is not None:
        _n_xft_deact_chk = 0
        for _r in m.r:
            for _f in m.f:
                try:
                    _xfteq_cd = _eq_xfteq_mp[(_r, _f, "check")]
                except KeyError:
                    continue
                if not _xfteq_cd.active:
                    continue
                try:
                    _xft_cd = _eq_xft_mp[(_r, _f, "check")]
                except KeyError:
                    continue
                if _xft_cd.active:
                    _xft_cd.deactivate()
                    _n_xft_deact_chk += 1
        if _n_xft_deact_chk:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "check period: deactivated eq_xft for %d (r,f) pairs "
                "(eq_xfteq active → eq_xft redundant, multi-period index fix)",
                _n_xft_deact_chk,
            )

    # gtap-mode: collapse pft/eq_pfteq for the active period (squares the
    # factor-price block; the SP-base squaring no-ops on the MP (r,f,t) index).
    if _gtap_mode:
        _n_pft_chk = _collapse_pft_pfteq(m, "check")
        if _n_pft_chk:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "check period: collapsed pft/eq_pfteq for %d (r,f) pairs "
                "(gtap-mode factor-price squaring)",
                _n_pft_chk,
            )
        # xp activity-scale holdfix: MEASURED OFF (2026-06-24).  It was added as a
        # patch compensating for the OLD pinned-pft bug (the scale slid because the
        # factor block was mis-anchored).  Now that pft is FREED correctly
        # (_collapse_pft_pfteq leaves real-factor pft free), the xp holdfix forces
        # the WRONG factor-block root: with it ON, CHECK=64.0%/SHOCK=61.3%; with it
        # OFF, CHECK=99.4%/SHOCK=66.9% (both code=1).  So we disable it for
        # gtap-mode (the freed pft self-anchors xp via the live eq_pfteq/eq_xfteq).
        if _HOLDFIX_ACTIVITY_SCALE_GTAP:
            _n_xp_chk = _holdfix_activity_scale(m, "check")
            if _n_xp_chk:
                import logging as _logging
                _logging.getLogger(__name__).info(
                    "check period: holdfixed xp at base for %d (r,a) (gtap-mode "
                    "activity-scale anchor)", _n_xp_chk,
                )

    # Replicate single-period structural fixing for check period.
    _chk_closure = base_closure if _gtap_mode else alt_closure
    _sp_ref_chk = GTAPModelEquations(
        p_alt.sets, p_alt, _chk_closure, residual_region=res_region,
    ).build_model()
    _replicate_sp_fixing(m, _sp_ref_chk, "check")
    _replicate_sp_bounds(m, _sp_ref_chk, "check")
    del _sp_ref_chk

    # Mute the inert welfare-report tail so PATH can certify code=1 (see
    # _mute_welfare_tail). Decisive once the base is exact (skip_base_solve):
    # check goes code=2 res 1.1e-2 → code=1 res 2.9e-11.
    if mute_welfare:
        _n_mute = _mute_welfare_tail(m, "check", list(p_alt.sets.r), gtap_mode=_gtap_mode)
        if _n_mute:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "check period: muted %d welfare-leaf rows (cv/ev/walras/u/ug/us)", _n_mute)

    # Seed the Python-only derived demand-volume vars (xc/xg/xi/xd/xmt/xiagg) from
    # the GAMS-seeded primals so the GAMS point is a true fixed point — else
    # eq_xc/eq_xg/eq_xd carry residual that pulls PATH off the basin.  Part of the
    # 3-piece faithful check recipe (project_gtap7_10x7_check_holdfix).
    if holdfix_cd and not _gtap_mode:
        _n_der = _complete_derived_seed(m, "check")
        if _n_der:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "check period: seeded %d derived demand-volume cells", _n_der)

    # Holdfix the CD-degenerate VA/ND nest (pva/pnd) at the GAMS-seeded values,
    # replicating GAMS gtap.holdfixed=1 in the check period.  Without this PATH
    # slides pva/pnd (free DOFs under CD) and collapses a region's price level.
    if holdfix_cd and not _gtap_mode:
        _n_hf = _holdfix_cd_nest(m, "check")
        if _n_hf:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "check period: holdfixed %d CD-nest cells (pva/pnd)", _n_hf)

    # Solve check on m.
    r_chk = run_gtap._run_path_capi_nonlinear_full(
        m, p_alt,
        enforce_post_checks=False,
        strict_path_capi=False,
        closure_config=_chk_closure,
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
    # Apply +10% imptx shock to params (tm_pct mode).  gtap-mode shocks the
    # domestic diagonal too (GAMS shocks ALL routes); altertax skips it.
    params_shock = copy.deepcopy(p_alt)
    _apply_imptx_shock(params_shock, factor=0.10, gtap_mode=_gtap_mode)

    # Freeze base and check; leave shock free.
    freeze_inactive_periods(m, "shock")

    # Warm-start shock from check solved values (quantities) — this is the right
    # warm start for convergence.  BUT it overwrites pva/pnd with the CHECK values;
    # since holdfix_cd will PIN pva/pnd, they must be re-seeded from the SHOCK GAMS
    # values (1.0078 for ROW/Mnfcs, not the check's 0.7592) or the pin is wrong.
    _pva_pnd_shock_seed = {}
    if holdfix_cd and not _gtap_mode:
        for _vn in ("pva", "pnd"):
            _v = getattr(m, _vn, None)
            if _v is None:
                continue
            for _idx in _v:
                _t = _idx[-1] if isinstance(_idx, tuple) else _idx
                if _t == "shock" and _v[_idx].value is not None:
                    _pva_pnd_shock_seed[(_vn, _idx)] = float(_v[_idx].value)

    # Warm-start shock from check solved values.
    _seed_period_from_prior(m, "check", "shock")

    # Restore the SHOCK GAMS pva/pnd seed that the prior-seed just clobbered.
    if holdfix_cd and not _gtap_mode:
        for (_vn, _idx), _val in _pva_pnd_shock_seed.items():
            try:
                getattr(m, _vn)[_idx].set_value(_val)
            except Exception:
                pass

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

    # FIX B (B3): deactivate eq_xft[r,f,'shock'] where eq_xfteq[r,f,'shock'] is active.
    # Same logic as B1/B2 for base/check: the single-period altertax gate silently
    # KeyErrors on multi-period (r,f,period) indices, leaving the system over-determined.
    # gtap-mode: do NOT blanket-deactivate eq_xft (see check branch rationale).
    # Altertax keeps the blanket deactivation (byte-identical to before).
    _eq_xft_mp = getattr(m, "eq_xft", None)
    _eq_xfteq_mp = getattr(m, "eq_xfteq", None)
    if not _gtap_mode and _eq_xft_mp is not None and _eq_xfteq_mp is not None:
        _n_xft_deact_shk = 0
        for _r in m.r:
            for _f in m.f:
                try:
                    _xfteq_cd = _eq_xfteq_mp[(_r, _f, "shock")]
                except KeyError:
                    continue
                if not _xfteq_cd.active:
                    continue
                try:
                    _xft_cd = _eq_xft_mp[(_r, _f, "shock")]
                except KeyError:
                    continue
                if _xft_cd.active:
                    _xft_cd.deactivate()
                    _n_xft_deact_shk += 1
        if _n_xft_deact_shk:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "shock period: deactivated eq_xft for %d (r,f) pairs "
                "(eq_xfteq active → eq_xft redundant, multi-period index fix)",
                _n_xft_deact_shk,
            )

    # gtap-mode: collapse pft/eq_pfteq for the active period (factor-price
    # squaring; SP-base squaring no-ops on the MP (r,f,t) index).
    if _gtap_mode:
        _n_pft_shk = _collapse_pft_pfteq(m, "shock")
        if _n_pft_shk:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "shock period: collapsed pft/eq_pfteq for %d (r,f) pairs "
                "(gtap-mode factor-price squaring)",
                _n_pft_shk,
            )
        # xp activity-scale holdfix: MEASURED OFF for gtap-mode — see the check
        # branch above (ON: CHECK 64.0/SHOCK 61.3; OFF: CHECK 99.4/SHOCK 66.9).
        if _HOLDFIX_ACTIVITY_SCALE_GTAP:
            _n_xp_shk = _holdfix_activity_scale(m, "shock")
            if _n_xp_shk:
                import logging as _logging
                _logging.getLogger(__name__).info(
                    "shock period: holdfixed xp at check for %d (r,a) (gtap-mode "
                    "activity-scale anchor)", _n_xp_shk,
                )

    # Replicate single-period structural fixing for shock period.
    _shk_closure = base_closure if _gtap_mode else alt_closure
    _sp_ref_shk = GTAPModelEquations(
        params_shock.sets, params_shock, _shk_closure, residual_region=res_region,
    ).build_model()
    _replicate_sp_fixing(m, _sp_ref_shk, "shock")
    _replicate_sp_bounds(m, _sp_ref_shk, "shock")
    del _sp_ref_shk

    # Mute the inert welfare-report tail for shock (same as check).
    if mute_welfare:
        _n_mute = _mute_welfare_tail(m, "shock", list(params_shock.sets.r), gtap_mode=_gtap_mode)
        if _n_mute:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "shock period: muted %d welfare-leaf rows (cv/ev/walras/u/ug/us)", _n_mute)

    # Seed derived demand-volume vars for shock (same as check).
    if holdfix_cd and not _gtap_mode:
        _n_der = _complete_derived_seed(m, "shock")
        if _n_der:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "shock period: seeded %d derived demand-volume cells", _n_der)

    # Holdfix the CD-degenerate VA/ND nest for shock (same as check).
    if holdfix_cd and not _gtap_mode:
        _n_hf = _holdfix_cd_nest(m, "shock")
        if _n_hf:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "shock period: holdfixed %d CD-nest cells (pva/pnd)", _n_hf)

    # gtap-mode: inject the tariff shock INTO the solved eq_pmeq[*,*,*,'shock']
    # cells (shock-in-equations) instead of relying on the post-solve cosmetic pm
    # patch. The MP builder bakes the BASE imptx as a literal coefficient, so the
    # solved import prices stay ~unshocked otherwise. SURGICAL: rebuilds ONLY the
    # eq_pmeq shock cells (a whole-slice rebuild recalibrates Armington/CDE shares
    # on the counterfactual and regresses to ~61%). With the wedge now in-equation,
    # the post-solve pm recompute becomes a NO-OP for gtap-mode (see below).
    # Params used by the post-solve recomputes below. Initialised here (not only
    # inside the `if not _gtap_mode` ytax[mt] block at line ~1885) because the
    # pm/pmt/pa recompute at the end (gated on `not _eq_pmeq_shock_rebuilt`) also
    # reads it: in gtap-mode + ifSUB=1 the eq_pmeq shock rebuild rewrites 0 cells
    # (the margin eqs are deactivated under ifSUB → eq_pmeq[*,*,*,shock] inactive),
    # so `_eq_pmeq_shock_rebuilt` stays False and that recompute runs with
    # `_recompute_params` otherwise-unbound. `p_alt` (the base-rate, no-shock params;
    # the recomputes apply shock_factor themselves) is defined in both modes.
    _recompute_params = p_alt
    _eq_pmeq_shock_rebuilt = False
    if _gtap_mode:
        _n_pmeq = _rebuild_eq_pmeq_shock(m, params_shock)
        if _n_pmeq:
            _eq_pmeq_shock_rebuilt = True
            import logging as _logging
            _logging.getLogger(__name__).info(
                "shock period: rebuilt %d eq_pmeq cells with the tm_pct shock "
                "power (shock-in-equations, gtap-mode)", _n_pmeq)
        else:
            # ifSUB=1: eq_pmeq is deactivated (the import price is the inlined macro
            # _m_pm baked into eq_xweq/eq_pmteq at build with BASE imptx). Inject the
            # shock there instead, else the wedge never enters the import equations
            # (xm[USA] +150%, SHOCK ~55%). gtap-mode + ifSUB leg of the same leak.
            _n_imp = _rebuild_import_demand_shock_ifsub(m, params, params_shock)
            if _n_imp:
                _eq_pmeq_shock_rebuilt = True
                import logging as _logging
                _logging.getLogger(__name__).info(
                    "shock period: rebuilt %d eq_xweq/eq_pmteq cells with the shocked "
                    "imptx (ifSUB import-price leak fix, gtap-mode)", _n_imp)

        # gtap-mode: inject the tariff shock INTO the solved eq_ytax[*,'mt','shock']
        # cells too. eq_ytax[mt] bakes the BASE imptx coefficient at build; with the
        # diagonal shocked (Link 2) the in-solve mt revenue otherwise misses the
        # entire diagonal tariff stream → understates regY → demand collapse →
        # Armington overshoot (the dominant income leak). The post-solve
        # _recompute_ytax_mt only patches the Var AFTER the solve (too late: regY/
        # demand already solved wrong), so it is no-op'd for gtap-mode below.
        _n_ymt = _rebuild_eq_ytax_mt_shock(m, params_shock)
        if _n_ymt:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "shock period: rebuilt %d eq_ytax[mt] cells with the shocked "
                "imptx (income-leak fix, shock-in-equations, gtap-mode)", _n_ymt)

    # Solve shock on m with shocked params.
    r_shk = run_gtap._run_path_capi_nonlinear_full(
        m, params_shock,
        enforce_post_checks=False,
        strict_path_capi=False,
        closure_config=_shk_closure,
        equation_scaling=True,
        solution_hint=None,
    )
    code_shk = int(r_shk.get("termination_code") or 0)
    res_shk = float(r_shk.get("residual") or float("inf"))
    results["shock"] = {"code": code_shk, "residual": res_shk}

    # Recompute ytax[mt]/ytaxshr[mt] (import-tax revenue) post-solve. eq_ytax for
    # gy='mt' bakes the BASE imptx coefficient; in the shock the solved xw/pmcif carry
    # the +10% shock but the imptx coefficient does not, so ytax[mt] is ~8.8% low.
    # ALTERTAX-ONLY now: in gtap-mode the shock is injected IN the solved
    # eq_ytax[mt] (_rebuild_eq_ytax_mt_shock above) so regY/demand solve on the
    # correct revenue. Re-running the post-solve recompute there OVERWRITES the
    # correct in-solve value with a slightly-off one (toggle: ON=99.10%,
    # OFF=99.70%), so it is gated to NOT gtap-mode.
    # Altertax path is byte-identical to before (uses base imptx p_alt *
    # (1+shock_factor); reads pmcif/xw, order vs pm free).
    if not _gtap_mode:
        _recompute_params = p_alt
        _n_mt = _recompute_ytax_mt(m, _recompute_params, "shock", shock_factor=0.10,
                                   gtap_mode=_gtap_mode)
        if _n_mt:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "shock period: recomputed %d ytax[mt]/ytaxshr[mt] cells", _n_mt)

    # Under ifSUB, recompute the report-only margin/price vars post-solve (GAMS
    # postsim). They are not solved (their eqs are deactivated, the real eqs use the
    # _m_* macros), so the Var objects keep their init value and mis-report. This also
    # sets pm — so the pm/pmt/pa recompute below MUST run AFTER it so pmt/pa derive
    # from the final pm (otherwise ifSUB=1 leaves pm/pmt/pa mutually inconsistent).
    if _if_sub:
        _n_rep = _recompute_ifsub_report_vars(m, params_shock, "shock", shock_factor=0.10)
        if _n_rep:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "shock period: recomputed %d ifSUB report-var cells (pfa/pfy/pp/pwmg/pefob/pmcif/pm)",
                _n_rep)

    # Recompute pm[e,i,imp] (bilateral) and pmt[imp,i] (aggregate) import prices, then
    # cascade to pa. eq_pmeq bakes the BASE imptx (model.imptx absent in MP → falls back
    # to base), so pm is low by the shock factor on high-tariff agro routes; pmt/pa
    # inherit. Shared by ifSUB=0 AND ifSUB=1 → NOT gated on _if_sub. Runs LAST so it
    # overrides any pm the ifSUB report recompute set and derives pmt/pa consistently.
    #
    # NO-OP when the eq_pmeq shock rebuild ran (gtap-mode): the shock is now IN the
    # solved eq_pmeq, so pm/pmt/pa already carry the +10% wedge. Re-applying the
    # post-solve pm patch would DOUBLE-APPLY (pm = (1+imptx_shocked+mtax)*pmcif on a
    # pm that already has it), corrupting the import prices. ytax[mt] is computed
    # from pmcif/xw (NOT pm) with its own power rate, so it stays correct and is NOT
    # gated — verified vs GAMS ytax[USA,mt] (see _recompute_ytax_mt above).
    if not _eq_pmeq_shock_rebuilt:
        _n_pm = _recompute_pm_pmt(m, _recompute_params, "shock", shock_factor=0.10, if_sub=_if_sub)
        if _n_pm:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "shock period: recomputed %d pm/pmt/pa import-price cells", _n_pm)

    # Freeze shock as well (for completeness / report purposes).
    freeze_period(m, "shock")

    return results
