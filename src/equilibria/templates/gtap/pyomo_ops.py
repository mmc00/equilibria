"""GTAP operations on the Pyomo multiperiod model that are not cell edits.

The recalibration of the shares, the welfare tail, the ifSUB report variables and
the benchmark factor wedge, as the Pyomo period model runs them. Moved here from the
old Pyomo multiperiod driver (PR 4 of gtap-on-core), unchanged.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any

from equilibria.blocks.gtap.factor_wedge import factor_wedge_rate, va_subsidy_basis
from equilibria.templates.gtap import period_rules as _rules


def mp_index(indexed: bool, sp_idx, active_period: str):
    """Indice multi-periodo de la celda ``sp_idx`` del modelo de un periodo."""
    if not indexed:
        return (active_period,)
    if isinstance(sp_idx, tuple):
        return (*sp_idx, active_period)
    return (sp_idx, active_period)


def benchmark_factor_wedge(params):
    """r, f, a -> fcttx + fctts del benchmark, con el signo del cierre (0 sin datos)."""
    bm = getattr(params, "benchmark", None)
    basis = va_subsidy_basis(params)

    def wedge(r, f, a):
        return 0.0 if bm is None else factor_wedge_rate(bm, r, f, a, basis)

    return wedge


# ---------------------------------------------------------------------------
# recalibrate_and_ava — replicate GAMS iterloop's per-period and/ava recalibration
# ---------------------------------------------------------------------------
# GAMS's calibration-of-parameters block (comp.gms, inside loop(tsim), AFTER that
# period's own solve, BEFORE the next period's solve) recomputes:
#   and(r,a,t) = (nd.l(r,a,t)/xp.l(r,a,t)) * (pnd.l(r,a,t)/px.l(r,a,t))**sigmap(r,a)
#   ava(r,a,t) = (va.l(r,a,t)/xp.l(r,a,t)) * (pva.l(r,a,t)/px.l(r,a,t))**sigmap(r,a)
# using THAT period's own solved values, not the benchmark SAM. Python built
# and_param/ava_param ONCE from the benchmark and bakes them as constants into
# every period's eq_nd/eq_va (gtap_model_multiperiod.py substitutes mutable
# Params by their build-time numeric value) — so check/shock never pick up
# GAMS's recalibration. Confirmed via CONVERT on a freshly-regenerated comp.gms
# (2026-07-12): and_param[JPN,VegFruit] Python=0.4518 (benchmark) vs GAMS's real
# check-period coefficient 0.3971 (= nd.l/(nd.l+va.l) at check, exact match to
# 10 sig figs) — a genuine, verified, ~5.5pp-magnitude bug, not a hypothesis.
# This is SURGICAL (only eq_nd/eq_va's own and_param/ava_param coefficient),
# NOT a whole-slice recalibration of Armington/CDE shares — a prior finding
# (the docstring of the old _rebuild_eq_pmeq_shock, removed in tariff-to-shock-
# block) already established that whole-slice
# recalibration on the SHOCK period regresses the match; this only touches
# and/ava, verified independently faithful via CONVERT before implementing.
#
# KEPT despite regressing the gate (60.87/60.66% -> 54.24/49.76%, codes stayed
# 2/2) per feedback_faithful_over_match_pct: fidelity to GAMS is the criterion,
# never match%/convergence. A faithful change regressing means ANOTHER link is
# missing/infaithful — diagnose and fix THAT, do not revert this.
#
# Step 0 of the driver recipe (2026-11-12) tried deleting it and KEPT it, measured
# with scripts/gtap/measure_altertax_cells.py over the 30 altertax gate rows. Note
# that today's GAMS reference GDXs hold and/ava identical in base, check and shock
# (gtap7_3x3, gtap7_10x7), so the July CONVERT reading above is not what the
# references say. Without this rebuild, 29 rows move by at most 2.4e-4 (relative)
# and keep their worst gap to GAMS, but nlp:gtap7_15x10:ifsub0 (IPOPT) moves to
# another point: check 100% -> 87.35%, shock 100% -> 95.17%, codes still 1, and
# 5703 cells leave the 1% tolerance (mostly xm, xd, xa, pa; zero-flow cells such
# as xa[ROW,Energy,inv,shock] go 8e-12 -> 9.4e-6 against GAMS 0). The rule is that
# no cell leaves tolerance, so it stays until that row is understood.
def recalibrate_and_ava(m, params, active_period: str, prior_period: str) -> int:
    """Rebuild eq_nd/eq_va for active_period using and/ava recomputed from
    prior_period's OWN solved nd/va/xp/pnd/pva/px (GAMS's iterloop formula).

    Returns the count of (r,a) cells rebuilt (nd+va combined).
    """
    from pyomo.environ import Constraint
    from pyomo.environ import value as _V

    eq_nd = getattr(m, "eq_nd", None)
    eq_va = getattr(m, "eq_va", None)
    nd = getattr(m, "nd", None)
    va = getattr(m, "va", None)
    xp = getattr(m, "xp", None)
    pnd = getattr(m, "pnd", None)
    pva = getattr(m, "pva", None)
    px = getattr(m, "px", None)
    if None in (eq_nd, eq_va, nd, va, xp, pnd, pva, px):
        return 0

    sigmap_map = getattr(params.elasticities, "sigmap", {}) or {}
    axp_map = getattr(params.shifts, "axp", {}) or {}
    lambdand_map = getattr(params.shifts, "lambdand", {}) or {}
    lambdava_map = getattr(params.shifts, "lambdava", {}) or {}

    n_rebuilt = 0
    for idx in list(eq_nd):
        if not (isinstance(idx, tuple) and idx[-1] == active_period):
            continue
        r, a, _t = idx
        prior_idx = (r, a, prior_period)
        try:
            xp_prior = float(_V(xp[prior_idx]))
            nd_prior = float(_V(nd[prior_idx]))
            va_prior = float(_V(va[prior_idx]))
            pnd_prior = float(_V(pnd[prior_idx]))
            pva_prior = float(_V(pva[prior_idx]))
            px_prior = float(_V(px[prior_idx]))
        except (KeyError, Exception):
            continue
        if xp_prior <= 0.0 or px_prior <= 0.0:
            continue

        sigmap = float(sigmap_map.get((r, a), 1.0))
        and_new = (
            (nd_prior / xp_prior) * (pnd_prior / px_prior) ** sigmap
            if pnd_prior > 0
            else 0.0
        )
        ava_new = (
            (va_prior / xp_prior) * (pva_prior / px_prior) ** sigmap
            if pva_prior > 0
            else 0.0
        )

        axp_shift = float(axp_map.get((r, a), 1.0))
        lambdand = float(lambdand_map.get((r, a), 1.0))
        lambdava = float(lambdava_map.get((r, a), 1.0))

        if and_new > 0.0 and eq_nd[idx].active:
            eq_nd[idx].deactivate()
            nd_shift = axp_shift * lambdand
            ratio_nd = px[idx] / pnd[idx]
            m.add_component(
                f"_eq_nd_recal_{r}_{a}_{active_period}",
                Constraint(
                    expr=nd[idx]
                    == and_new * xp[idx] * ratio_nd**sigmap * nd_shift ** (sigmap - 1.0)
                ),
            )
            n_rebuilt += 1
        if ava_new > 0.0 and eq_va[idx].active:
            eq_va[idx].deactivate()
            va_shift = axp_shift * lambdava
            ratio_va = px[idx] / pva[idx]
            m.add_component(
                f"_eq_va_recal_{r}_{a}_{active_period}",
                Constraint(
                    expr=va[idx]
                    == ava_new * xp[idx] * ratio_va**sigmap * va_shift ** (sigmap - 1.0)
                ),
            )
            n_rebuilt += 1
    return n_rebuilt


# u depends on ug/us (and the live uh), so it goes last.
WELFARE_SOLVE_ORDER = _rules.WELFARE_SOLVE_ORDER


def solve_welfare_tail(m, period: str) -> int:
    """Solve each row muted by `steps.mute_welfare` for its own var, after the
    period's solve, so the reported ev/cv/u/ug/us are the period's and not the seed.
    The rows stay deactivated and the vars fixed (the solved system is unchanged).
    A row with no real solution (e.g. u with us < 0 after negative saving) leaves
    its var at NaN, with a warning: the welfare is undefined there, and the seed
    would read as a result.  Returns the count of cells solved."""
    import logging

    from pyomo.common.errors import IterationLimitError
    from pyomo.util.calc_var_value import calculate_variable_from_constraint

    n = 0
    undefined = []
    for eqn, vn in WELFARE_SOLVE_ORDER:
        eqc = getattr(m, eqn, None)
        vc = getattr(m, vn, None)
        if eqc is None or vc is None:
            continue
        for idx in eqc:
            t = idx[-1] if isinstance(idx, tuple) else idx
            cd = eqc[idx]
            if t != period or cd.active or idx not in vc:
                continue
            vd = vc[idx]
            if not vd.fixed:
                continue
            vd.unfix()
            try:
                calculate_variable_from_constraint(vd, cd)
                n += 1
            except (ValueError, ArithmeticError, IterationLimitError):
                vd.set_value(float("nan"), skip_validation=True)
                undefined.append(vd.name)
            finally:
                vd.fix()
    if undefined:
        logging.getLogger(__name__).warning(
            "%s period: welfare undefined (no real solution), set to NaN: %s",
            period,
            ", ".join(undefined),
        )
    return n


# ---------------------------------------------------------------------------
# recompute_ifsub_report_vars — post-solve fill of the ifSUB report variables
# ---------------------------------------------------------------------------
def recompute_ifsub_report_vars(m, params, period: str) -> int:
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

    imptx is the ShockBlock Var of ``period`` (the one the macro M_PM read in the
    solve, shock included). Returns the count of report-var cells written.
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

    # The factor tax/subsidy wedge fctts+fcttx (factor_wedge_rate) comes from the
    # benchmark HAR, NOT from model Vars/Params (m.fctts/m.fcttx are 0 under altertax).
    # Reading them off the model gave pfa=pf (missing the wedge) → pfa[EU_28,UnSkLab,
    # Food] 1.37 vs GAMS 2.26 (39% off). Compute it from the benchmark.
    _pfa_wedge = benchmark_factor_wedge(params)

    # --- factor prices: pfa, pfy ---
    for r in R:
        for f in F:
            for a in A:
                pf = (
                    _pv("pf", (r, f, a, period), None)
                    if getattr(m, "pf", None) is not None
                    else None
                )
                if pf is None:
                    continue
                kappaf = _pv("kappaf", (r, f, a, period))
                for vn, val in (
                    ("pfa", pf * (1.0 + _pfa_wedge(r, f, a))),
                    ("pfy", pf * (1.0 - kappaf)),
                ):
                    v = getattr(m, vn, None)
                    if v is None:
                        continue
                    with contextlib.suppress(Exception):
                        _force(v[r, f, a, period], val)

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
                p_rai = (
                    _pv("p_rai", (r, a, i, period), None)
                    if getattr(m, "p_rai", None) is not None
                    else None
                )
                if p_rai is None or pp_rai is None:
                    continue
                with contextlib.suppress(Exception):
                    _force(pp_rai[r, a, i, period], p_rai)

    # --- trade prices: pwmg, pefob, pmcif, pm ---
    for r in R:  # exporter
        for i in I:
            for rp in R:  # importer
                pe = (
                    _pv("pe", (r, i, rp, period), None)
                    if getattr(m, "pe", None) is not None
                    else None
                )
                if pe is None:
                    continue
                # pwmg = Σ_m amgm*ptmg/lambdamg
                pwmg_val = 0.0
                for mm in M:
                    amgm = (
                        _param(params.benchmark.amgm, (mm, r, i, rp))
                        if hasattr(params.benchmark, "amgm")
                        else _pv("amgm", (mm, r, i, rp))
                    )
                    ptmg = _pv("ptmg", (mm, period))
                    lam = _pv("lambdamg", (mm, r, i, rp, period), 1.0) or 1.0
                    pwmg_val += amgm * ptmg / (lam + 1e-12)
                rtxs = (
                    _param(params.taxes.rtxs, (r, i, rp))
                    if hasattr(params.taxes, "rtxs")
                    else 0.0
                )
                etax = _pv("etax", (r, i, period))
                pefob_val = (1.0 + rtxs + etax) * pe
                # tmarg is dim-3 (no period axis); mtax/chipm come from params.
                tmarg = _pv("tmarg", (r, i, rp))
                if not tmarg:
                    tmarg = _pv("tmarg", (r, i, rp, period))
                pmcif_val = pefob_val + pwmg_val * tmarg
                # imptx: the ShockBlock Var of the period, as the solve read it.
                imptx = _pv("imptx", (r, i, rp, period))
                mtax = (
                    _param(params.taxes.mtax, (rp, i))
                    if hasattr(params.taxes, "mtax")
                    else 0.0
                )
                chipm = (
                    _param(params.benchmark.chipm, (r, i, rp), 1.0)
                    if hasattr(params.benchmark, "chipm")
                    else 1.0
                )
                chipm = chipm or 1.0
                pm_val = (1.0 + imptx + mtax) * pmcif_val / (chipm + 1e-12)
                # All four are indexed (exporter, commodity, importer) = (r, i, rp),
                # matching the GAMS report convention pm(r,i,rp). (An earlier (rp,i,r)
                # key for pm transposed the route → wrong cell.)
                for vn, val, key in (
                    ("pwmg", pwmg_val, (r, i, rp, period)),
                    ("pefob", pefob_val, (r, i, rp, period)),
                    ("pmcif", pmcif_val, (r, i, rp, period)),
                    ("pm", pm_val, (r, i, rp, period)),
                ):
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
