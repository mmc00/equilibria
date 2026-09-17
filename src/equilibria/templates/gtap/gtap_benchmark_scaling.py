"""Escalado de benchmark GTAP: lleva los seeds al punto de GAMS ``cal.gms``.

Estas cuatro funciones son el ULTIMO paso antes de construir ecuaciones: re-valuan
los niveles iniciales del modelo para que el warm-start coincida con GAMS. No
declaran ecuaciones ni variables —solo mutan ``VarData`` por nombre, con guarda
``hasattr``—, asi que corren igual sobre el modelo del monolito
(``GTAPModelEquations.build_model``) que sobre el compuesto por bloques
(``gtap_block_model.build_block_single_period``).

Vivian como metodos privados de ``GTAPModelEquations``. El composer de bloques
las necesitaba, y para llamarlas construia un ``GTAPModelEquations`` entero solo
para tomarle prestados dos metodos privados —un shim que hacia parecer que los
bloques dependen de las ecuaciones del monolito, cuando lo unico que comparten es
este escalado—. Extraerlas deja esa relacion explicita.

El monolito conserva los cuatro metodos como delegadores de una linea: sigue
siendo el ORACULO de paridad contra GAMS y su comportamiento no cambia.

El estado que necesitan —``params``, ``sets``, ``residual_region``,
``is_counterfactual``— viaja en un :class:`ScalingContext` en vez de en ``self``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from equilibria.templates.gtap.gtap_parameters import (
    GTAP_GOVERNMENT_AGENT,
    GTAP_HOUSEHOLD_AGENT,
    GTAP_INVESTMENT_AGENT,
    GTAP_MARGIN_AGENT,
)

if TYPE_CHECKING:
    from pyomo.environ import ConcreteModel

    from equilibria.templates.gtap.gtap_parameters import GTAPParameters
    from equilibria.templates.gtap.gtap_sets import GTAPSets

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScalingContext:
    """Lo que el escalado necesita saber del modelo, sin el modelo entero.

    ``GTAPModelEquations`` y el composer de bloques lo construyen igual; es la
    frontera que sustituye al ``self`` del monolito.
    """

    params: GTAPParameters | Any
    sets: GTAPSets | Any
    residual_region: str = "NAmerica"
    is_counterfactual: bool = False


def align_xi_xaa_post_scaling(model: ConcreteModel, ctx: ScalingContext) -> None:
    """Re-sync xi, xaa, xda, xma and all downstream aggregates with income-side xiagg.

    After apply_production_scaling, xiagg[r] = yi[r]/pi[r] uses the GAMS
    income-side identity (pi*depr*kstock + rsav + savf).  The xi variables
    were initialised from the demand-side SAM totals, which can differ by
    ~0.5% (e.g. EastAsia: demand 5.119 vs income 5.114).  This mismatch
    creates a ~2.6e-3 initial residual in eq_xi that PATH cannot reduce
    (code 2 / no_progress).

    Full cascade (mirrors the logic in apply_production_scaling but runs
    AFTER _refresh_macro_initial_state has set the income-side xiagg):

      xi, xaa[inv]       → eq_xi, eq_xaa_inv satisfied (residual = 0)
      xda[inv] *= k      → Armington shares preserved → eq_paa, eq_xda, eq_xma
      xma[inv] *= k      → same
      xd, xmt            → eq_xd_agg, eq_xmt_agg satisfied
      xds                → eq_pdeq satisfied
      xa                 → eq_xa satisfied
      xet (omega=inf)    → eq_xseq (supply identity xs=xds+xet) satisfied
      gw_share from xet  → eq_xw remains satisfied with old xw values
      gdpmp, rgdpmp      → eq_gdpmp, eq_pgdpmp satisfied

    The only remaining small residuals are in eq_xweq[rp,i,r] (Armington
    bilateral import demand), of order amw * Δxmt ≈ O(1e-4), easily handled
    by PATH in the first iteration.
    """
    from pyomo.environ import value as pyo_value

    # ---------- per-(r,i) xi and Armington updates -------------------------
    delta_xi: dict = {}
    for r in model.r:
        pi_val = max(float(pyo_value(model.pi[r])), 1e-8)
        xiagg = float(pyo_value(model.xiagg[r]))
        sigmai_raw = float(ctx.params.elasticities.esubi.get(r, 0.0))
        if abs(sigmai_raw - 1.0) < 1e-8:
            sigmai_raw = 1.01
        for i in model.i:
            share = float(pyo_value(model.i_share[r, i]))
            if share <= 0.0:
                delta_xi[(r, i)] = 0.0
                continue
            pa_inv = max(float(pyo_value(model.pa[r, i, GTAP_INVESTMENT_AGENT])), 1e-12)
            xi_old = float(pyo_value(model.xi[r, i]))
            xi_new = max(share * xiagg * (pi_val / pa_inv) ** sigmai_raw, 0.0)
            delta = xi_new - xi_old
            delta_xi[(r, i)] = delta

            # xi and xaa[inv]
            model.xi[r, i].set_value(xi_new)
            if hasattr(model, "xaa") and (r, i, GTAP_INVESTMENT_AGENT) in model.xaa:
                model.xaa[r, i, GTAP_INVESTMENT_AGENT].set_value(xi_new)

            # Scale xda/xma proportionally so Armington shares are preserved.
            if xi_old > 1e-12 and abs(delta) > 1e-14:
                k = xi_new / xi_old
                if hasattr(model, "xda") and (r, i, GTAP_INVESTMENT_AGENT) in model.xda:
                    old = float(pyo_value(model.xda[r, i, GTAP_INVESTMENT_AGENT]))
                    model.xda[r, i, GTAP_INVESTMENT_AGENT].set_value(max(old * k, 0.0))
                if hasattr(model, "xma") and (r, i, GTAP_INVESTMENT_AGENT) in model.xma:
                    old = float(pyo_value(model.xma[r, i, GTAP_INVESTMENT_AGENT]))
                    model.xma[r, i, GTAP_INVESTMENT_AGENT].set_value(max(old * k, 0.0))

    # ---------- recompute aggregates so all aggregation eqs are satisfied --
    for r in model.r:
        for i in model.i:
            # eq_xd_agg: xd = sum_aa(xda/xscale)
            if hasattr(model, "xd"):
                total_xd = sum(
                    float(pyo_value(model.xda[r, i, aa]))
                    / max(float(pyo_value(model.xscale[r, aa])), 1e-12)
                    for aa in model.aa
                )
                model.xd[r, i].set_value(max(total_xd, 1e-8))

            # eq_xmt_agg: xmt = sum_aa(xma/xscale)
            if hasattr(model, "xmt"):
                total_xmt = sum(
                    float(pyo_value(model.xma[r, i, aa]))
                    / max(float(pyo_value(model.xscale[r, aa])), 1e-12)
                    for aa in model.aa
                )
                model.xmt[r, i].set_value(max(total_xmt, 1e-8))

            # eq_pdeq: xds = sum_aa(xda/xscale for aa with domestic_share>0)
            # For simplicity use the same sum as xd (same result if all aa have share>0)
            if hasattr(model, "xds"):
                model.xds[r, i].set_value(max(float(pyo_value(model.xd[r, i])), 1e-8))

            # eq_xseq (omega=inf case): xs = xds + xet  →  xet = xs - xds
            # For finite omega, eq_xseq is a price eq: skip xet update.
            omega = ctx.params.elasticities.omegax.get((r, i), float("inf"))
            if omega == float("inf") and hasattr(model, "xet") and hasattr(model, "xs"):
                xs_val = float(pyo_value(model.xs[r, i]))
                xds_val = float(pyo_value(model.xds[r, i]))
                xet_old = max(float(pyo_value(model.xet[r, i])), 1e-12)
                xet_new = max(xs_val - xds_val, 0.0)
                if xet_new > 1e-12 and abs(xet_new - xet_old) > 1e-14:
                    model.xet[r, i].set_value(xet_new)
                    scale = xet_new / xet_old
                    # For omegaw=inf: eq_peteq says xet = sum(xw), eq_peeq says pe = pet.
                    # Scaling xw proportionally satisfies eq_peteq with xet_new.
                    # gw_share is NOT used in omegaw=inf equations, so leave unchanged.
                    if hasattr(model, "xw"):
                        omegaw = ctx.params.elasticities.omegaw.get(
                            (r, i), float("inf")
                        )
                        if omegaw == float("inf"):
                            for rp in model.rp:
                                if (r, i, rp) in model.xw:
                                    xw_old = float(pyo_value(model.xw[r, i, rp]))
                                    if xw_old > 0.0:
                                        model.xw[r, i, rp].set_value(xw_old * scale)

    # ---------- update gdpmp / rgdpmp: Δgdpmp = Σ_i Δxi per region ----------
    for r in model.r:
        delta_r = sum(delta_xi.get((r, i), 0.0) for i in model.i)
        if abs(delta_r) < 1e-14:
            continue
        old_gdpmp = float(pyo_value(model.gdpmp[r]))
        model.gdpmp[r].set_value(max(old_gdpmp + delta_r, 1e-8))
        old_rgdpmp = float(pyo_value(model.rgdpmp[r]))
        model.rgdpmp[r].set_value(max(old_rgdpmp + delta_r, 1e-8))

    updated = sum(1 for v in delta_xi.values() if abs(v) > 1e-14)
    if updated:
        logger.info(
            "align_xi_xaa_post_scaling: updated %d (r,i) investment pairs "
            "(xi/xaa/xda/xma/xd/xmt/xds/xa/xet/gw_share/gdpmp) to income-side xiagg",
            updated,
        )


def apply_production_scaling(model: ConcreteModel, ctx: ScalingContext) -> None:
    """Apply xScale to production variables after initialization.

    Following GAMS pattern (cal.gms lines 905-911):
    - Variables are initialized at benchmark (unscaled) values
    - After initialization, production-side variables are multiplied by xScale
    - This improves numerical conditioning for the solver

    Note: Python model has different indexing than GAMS:
    - GAMS: xd(r,i,a,t), xa(r,i,a,t), xm(r,i,a,t) - indexed by activity
    - Python: xd(r,i), xa(r,i) - NOT indexed by activity (aggregated)
    So we only scale variables indexed by activity: xf, xp, va, nd
    """
    from pyomo.environ import value

    final_demand_agents = (
        GTAP_HOUSEHOLD_AGENT,
        GTAP_GOVERNMENT_AGENT,
        GTAP_INVESTMENT_AGENT,
        GTAP_MARGIN_AGENT,
    )
    # Capture the base-year levels used by the compStat Fisher indices.
    # GAMS formulas mix current prices/quantities with the original t0 levels.
    base_pa = {
        (r, i, agent): float(value(model.pa[r, i, agent]))
        for r in model.r
        for i in model.i
        for agent in final_demand_agents
    }
    base_xaa = {
        (r, i, agent): float(value(model.xaa[r, i, agent]))
        for r in model.r
        for i in model.i
        for agent in final_demand_agents
    }
    base_pefob = {
        (r, i, rp): float(value(model.pefob[r, i, rp]))
        for r in model.r
        for i in model.i
        for rp in model.rp
    }
    base_pmcif = {
        (rp, i, r): float(value(model.pmcif[rp, i, r]))
        for rp in model.rp
        for i in model.i
        for r in model.r
    }
    base_xw = {
        (r, i, rp): float(value(model.xw[r, i, rp]))
        for r in model.r
        for i in model.i
        for rp in model.rp
    }
    base_pabs = {r: max(float(value(model.pabs[r])), 1e-8) for r in model.r}
    base_rgdpmp = {r: max(float(value(model.rgdpmp[r])), 1e-8) for r in model.r}

    # Scale factor demands (xf) - indexed by (r, f, a)
    for key in model.xf:
        r, f, a = key
        xscale_val = float(value(model.xscale[r, a]))
        if abs(xscale_val - 1.0) > 1e-12:
            xf_val = value(model.xf[key])
            if xf_val is not None:
                model.xf[key].set_value(xf_val * xscale_val)

    # Scale production aggregates - indexed by (r, a)
    for key in model.xp:
        r, a = key
        xscale_val = float(value(model.xscale[r, a]))
        if abs(xscale_val - 1.0) > 1e-12:
            xp_val = value(model.xp[key])
            if xp_val is not None:
                model.xp[key].set_value(xp_val * xscale_val)

    for key in model.va:
        r, a = key
        xscale_val = float(value(model.xscale[r, a]))
        if abs(xscale_val - 1.0) > 1e-12:
            va_val = value(model.va[key])
            if va_val is not None:
                model.va[key].set_value(va_val * xscale_val)

    for key in model.nd:
        r, a = key
        xscale_val = float(value(model.xscale[r, a]))
        if abs(xscale_val - 1.0) > 1e-12:
            nd_val = value(model.nd[key])
            if nd_val is not None:
                model.nd[key].set_value(nd_val * xscale_val)

    # GAMS also rescales activity-level Armington quantities on the
    # production side: xd(r,i,a), xm(r,i,a), xa(r,i,a).
    for var_name in ("xda", "xma", "xaa"):
        if not hasattr(model, var_name):
            continue
        var = getattr(model, var_name)
        for key in var:
            if not isinstance(key, tuple) or len(key) != 3:
                continue
            r, _i, aa = key
            if aa not in ctx.sets.a:
                continue
            xscale_val = float(value(model.xscale[r, aa]))
            if abs(xscale_val - 1.0) > 1e-12:
                level = value(var[key])
                if level is not None:
                    var[key].set_value(level * xscale_val)

    # Refresh Armington aggregates after scaling activity-level xda/xma/xaa.
    if hasattr(model, "xd"):
        for r in model.r:
            for i in model.i:
                total_xd = sum(
                    value(model.xda[r, i, aa]) / max(value(model.xscale[r, aa]), 1e-12)
                    for aa in model.aa
                )
                model.xd[r, i].set_value(max(total_xd, 1e-8))

    if hasattr(model, "xds") and hasattr(model, "xda"):
        for r in model.r:
            for i in model.i:
                total_xds = sum(
                    value(model.xda[r, i, aa]) / max(value(model.xscale[r, aa]), 1e-12)
                    for aa in model.aa
                )
                model.xds[r, i].set_value(max(total_xds, 1e-8))

    if hasattr(model, "xet") and hasattr(model, "xs") and hasattr(model, "xds"):
        for r in model.r:
            for i in model.i:
                has_export_route = any(
                    value(model.xw_flag[r, i, rp]) > 0.0 for rp in model.rp
                )
                if not has_export_route:
                    lb = model.xet[r, i].lb
                    if lb is not None and float(lb) > 0.0:
                        model.xet[r, i].setlb(0.0)
                    model.xet[r, i].set_value(0.0)
                    if hasattr(model, "xet_flag"):
                        model.xet_flag[r, i].set_value(0.0)
                    continue

                # Match GAMS cal.gms initialization:
                # xet.l = (ps.l*xs.l - pd.l*xds.l) / pet.l
                numerator = value(model.ps[r, i]) * value(model.xs[r, i]) - value(
                    model.pd[r, i]
                ) * value(model.xds[r, i])
                pet_val = max(value(model.pet[r, i]), 1e-12)
                xet_val = max(numerator / pet_val, 0.0)
                if xet_val <= 0.0:
                    # Fallback: benchmark VXSB total.  Needed when finite omegax
                    # (altertax) causes xs to initialize from maks-based levels
                    # while xds is demand-based (makb), making ps*xs < pd*xds.
                    xet_bench = sum(
                        float(
                            ctx.params.benchmark.vxsb.get(
                                (str(r), str(i), str(rp)), 0.0
                            )
                            or 0.0
                        )
                        for rp in model.rp
                    )
                    if xet_bench > 0.0:
                        xet_val = xet_bench
                    lb = model.xet[r, i].lb
                    if lb is not None and float(lb) > 0.0:
                        model.xet[r, i].setlb(0.0)
                model.xet[r, i].set_value(xet_val)
                if hasattr(model, "xet_flag"):
                    model.xet_flag[r, i].set_value(1.0 if xet_val > 1e-7 else 0.0)

    if hasattr(model, "gw_share") and hasattr(model, "xw") and hasattr(model, "xet"):
        for r in model.r:
            for i in model.i:
                xet_val = max(value(model.xet[r, i]), 1e-12)
                omegaw = ctx.params.elasticities.omegaw.get((r, i), float("inf"))
                for rp in model.rp:
                    if value(model.xw_flag[r, i, rp]) <= 0.0:
                        model.gw_share[r, i, rp].set_value(0.0)
                        continue
                    xw_val = max(value(model.xw[r, i, rp]), 0.0)
                    pe_val = max(value(model.pe[r, i, rp]), 1e-12)
                    pet_val = max(value(model.pet[r, i]), 1e-12)
                    if omegaw == float("inf"):
                        share = (pe_val * xw_val) / max(pet_val * xet_val, 1e-12)
                    else:
                        share = (xw_val / xet_val) * (pet_val / pe_val) ** omegaw
                    model.gw_share[r, i, rp].set_value(max(share, 0.0))
                if hasattr(model, "xet_flag"):
                    model.xet_flag[r, i].set_value(1.0 if xet_val > 1e-7 else 0.0)

    _refresh_cet_share_state(model, ctx)

    if hasattr(model, "xmt") and hasattr(model, "xma"):
        for r in model.r:
            for i in model.i:
                total_xm = sum(
                    value(model.xma[r, i, aa]) / max(value(model.xscale[r, aa]), 1e-12)
                    for aa in model.aa
                )
                model.xmt[r, i].set_value(max(total_xm, 1e-8))

    _refresh_cet_share_state(model, ctx)

    # Enforce benchmark household demand coherence before macro refresh:
    # yc = sum_i pa*hhd * xc and xc = c_share * yc / pa.
    if hasattr(model, "xc") and hasattr(model, "c_share") and hasattr(model, "yc"):
        for r in model.r:
            yc_target = sum(
                float(ctx.params.benchmark.get_private_demand(str(r), str(i))[0] or 0.0)
                for i in model.i
            )
            yc_target = max(yc_target, 1e-8)
            model.yc[r].set_value(yc_target)
            for i in model.i:
                share = max(float(value(model.c_share[r, i]) or 0.0), 0.0)
                pa_hhd = max(
                    float(value(model.pa[r, i, GTAP_HOUSEHOLD_AGENT]) or 1.0), 1e-12
                )
                model.xc[r, i].set_value(max((share * yc_target) / pa_hhd, 0.0))

    if hasattr(model, "xaa") and hasattr(model, "xc"):
        for r in model.r:
            for i in model.i:
                model.xaa[r, i, GTAP_HOUSEHOLD_AGENT].set_value(
                    max(value(model.xc[r, i]), 0.0)
                )
    if hasattr(model, "xaa") and hasattr(model, "xg"):
        for r in model.r:
            for i in model.i:
                model.xaa[r, i, GTAP_GOVERNMENT_AGENT].set_value(
                    max(value(model.xg[r, i]), 0.0)
                )
    if hasattr(model, "xaa") and hasattr(model, "xi"):
        for r in model.r:
            for i in model.i:
                model.xaa[r, i, GTAP_INVESTMENT_AGENT].set_value(
                    max(value(model.xi[r, i]), 0.0)
                )

    _refresh_macro_initial_state(model, ctx)


def _refresh_macro_initial_state(model: ConcreteModel, ctx: ScalingContext) -> None:
    """Refresh macro variables after any xScale-sensitive initialization changes."""
    from pyomo.environ import value

    final_demand_agents = (
        GTAP_HOUSEHOLD_AGENT,
        GTAP_GOVERNMENT_AGENT,
        GTAP_INVESTMENT_AGENT,
        GTAP_MARGIN_AGENT,
    )
    base_pa = {
        (r, i, agent): float(value(model.pa[r, i, agent]))
        for r in model.r
        for i in model.i
        for agent in final_demand_agents
    }
    base_xaa = {
        (r, i, agent): float(value(model.xaa[r, i, agent]))
        for r in model.r
        for i in model.i
        for agent in final_demand_agents
    }
    base_pefob = {
        (r, i, rp): float(value(model.pefob[r, i, rp]))
        for r in model.r
        for i in model.i
        for rp in model.rp
    }
    base_pmcif = {
        (rp, i, r): float(value(model.pmcif[rp, i, r]))
        for rp in model.rp
        for i in model.i
        for r in model.r
    }
    base_xw = {
        (r, i, rp): float(value(model.xw[r, i, rp]))
        for r in model.r
        for i in model.i
        for rp in model.rp
    }
    base_pabs = {r: max(float(value(model.pabs[r])), 1e-8) for r in model.r}
    base_rgdpmp = {r: max(float(value(model.rgdpmp[r])), 1e-8) for r in model.r}

    for r in model.r:
        capital_factors = [
            f for f in model.f if str(f).lower() in ("capital", "cap", "k", "kap")
        ]
        for f in model.f:
            if hasattr(model, "xft") and (r, f) in model.xft and f in ctx.sets.mf:
                model.xft[r, f].set_value(
                    sum(
                        value(model.xf[r, f, a]) / max(value(model.xscale[r, a]), 1e-12)
                        for a in model.a
                    )
                )

        if hasattr(model, "kstock") and r in model.kstock:
            raw_vkb = ctx.params.benchmark.vkb
            benchmark_kstock_val = raw_vkb.get(r)
            if benchmark_kstock_val is None:
                benchmark_kstock_val = raw_vkb.get((r,), 0.0)
            benchmark_kstock = float(benchmark_kstock_val or 0.0)
            if benchmark_kstock > 0.0:
                model.kstock[r].set_value(max(benchmark_kstock, 1e-8))

        if hasattr(model, "ytax") and (r, "ft") in model.ytax:
            # eq_ytax[ft] = sum(fcttx * pf * xf / xscale); fcttx=ftrv/EVFB
            # rtf=VFM/EVFB-1 includes kappaf (factor rent), NOT a tax.
            ft_total = 0.0
            if hasattr(model, "fcttx"):
                for rr, f, a in [(rr, f, a) for (rr, f, a) in model.fcttx if rr == r]:
                    ft_total += (
                        value(model.fcttx[rr, f, a])
                        * value(model.pf[r, f, a])
                        * value(model.xf[r, f, a])
                        / max(value(model.xscale[r, a]), 1e-12)
                    )
            model.ytax[r, "ft"].set_value(ft_total)

        if hasattr(model, "ytax") and (r, "dt") in model.ytax:
            dt_total = 0.0
            for f in model.f:
                for a in model.a:
                    kappa = float(
                        ctx.params.taxes.kappaf_activity.get((r, f, a), 0.0) or 0.0
                    )
                    if kappa == 0.0:
                        kappa = float(ctx.params.taxes.kappaf.get((r, f), 0.0) or 0.0)
                    if kappa == 0.0:
                        continue
                    dt_total += (
                        kappa
                        * value(model.pf[r, f, a])
                        * value(model.xf[r, f, a])
                        / max(value(model.xscale[r, a]), 1e-12)
                    )
            model.ytax[r, "dt"].set_value(dt_total)

        # Recalibrate ytax[pt] (production/output tax) from eq_ytax rule:
        # sum_a sum_i prdtx_rai * p_rai * x (where xflag > 0)
        if hasattr(model, "ytax") and (r, "pt") in model.ytax:
            pt_total = 0.0
            for a in model.a:
                outputs = ctx.sets.activity_commodities.get(str(a), list(ctx.sets.i))
                for i in outputs:
                    if (r, a, i) not in model.xflag or value(
                        model.xflag[r, a, i]
                    ) <= 0.0:
                        continue
                    prdtx = (
                        value(model.prdtx_rai[r, a, i])
                        if (r, a, i) in model.prdtx_rai
                        else 0.0
                    )
                    if prdtx == 0.0:
                        continue
                    p_rai_v = (
                        value(model.p_rai[r, a, i]) if (r, a, i) in model.p_rai else 1.0
                    )
                    x_v = value(model.x[r, a, i]) if (r, a, i) in model.x else 0.0
                    pt_total += prdtx * p_rai_v * x_v
            model.ytax[r, "pt"].set_value(pt_total)

        # Recalibrate ytax[pc/gc/ic] (commodity taxes on private/gov/invest) from eq_ytax rule
        _ctax_agents = {
            "pc": [GTAP_HOUSEHOLD_AGENT],
            "gc": [GTAP_GOVERNMENT_AGENT],
            "ic": [GTAP_INVESTMENT_AGENT],
            "fc": list(model.a),
        }
        for gy, agents in _ctax_agents.items():
            if not (hasattr(model, "ytax") and (r, gy) in model.ytax):
                continue
            ctax_total = 0.0
            for aa in agents:
                for i in model.i:
                    dintx = float(ctx.params.taxes.dintx0.get((r, i, aa), 0.0) or 0.0)
                    mintx = float(ctx.params.taxes.mintx0.get((r, i, aa), 0.0) or 0.0)
                    scale = (
                        value(model.xscale[r, aa])
                        if aa in model.a and (r, aa) in model.xscale
                        else 1.0
                    )
                    if dintx != 0.0 and (r, i, aa) in model.xda:
                        ctax_total += (
                            dintx
                            * value(model.pd[r, i])
                            * value(model.xda[r, i, aa])
                            / max(scale, 1e-12)
                        )
                    if mintx != 0.0 and (r, i, aa) in model.xma:
                        ctax_total += (
                            mintx
                            * value(model.pmt[r, i])
                            * value(model.xma[r, i, aa])
                            / max(scale, 1e-12)
                        )
            model.ytax[r, gy].set_value(ctax_total)

        if hasattr(model, "facty") and r in model.facty:
            gross_factor_income = sum(
                value(model.pf[r, f, a])
                * value(model.xf[r, f, a])
                / max(value(model.xscale[r, a]), 1e-12)
                for f in model.f
                for a in model.a
            )
            model.facty[r].set_value(
                gross_factor_income
                - value(model.fdepr[r]) * value(model.pi[r]) * value(model.kstock[r])
            )

        if hasattr(model, "ytaxTot") and r in model.ytaxTot:
            model.ytaxTot[r].set_value(sum(value(model.ytax[r, gy]) for gy in model.gy))
        if hasattr(model, "ytax_ind") and r in model.ytax_ind:
            model.ytax_ind[r].set_value(
                value(model.ytaxTot[r]) - value(model.ytax[r, "dt"])
            )
        if hasattr(model, "regy") and r in model.regy:
            model.regy[r].set_value(value(model.facty[r]) + value(model.ytax_ind[r]))
        regy_raw = value(model.regy[r])
        regy_val = max(abs(regy_raw), 1e-8)
        if hasattr(model, "ytaxshr"):
            for gy in model.gy:
                model.ytaxshr[r, gy].set_value(value(model.ytax[r, gy]) / regy_val)
        if hasattr(model, "yc") and r in model.yc:
            # GAMS benchmark identity (cal.gms): yc = sum_i pa(r,i,hhd) * xa(r,i,hhd)
            yc_demand = sum(
                value(model.pa[r, i, GTAP_HOUSEHOLD_AGENT])
                * value(model.xaa[r, i, GTAP_HOUSEHOLD_AGENT])
                for i in model.i
                if (r, i, GTAP_HOUSEHOLD_AGENT) in model.pa
                and (r, i, GTAP_HOUSEHOLD_AGENT) in model.xaa
            )
            if yc_demand > 0.0:
                model.yc[r].set_value(yc_demand)
            else:
                model.yc[r].set_value(
                    value(model.betap[r])
                    * (value(model.phi[r]) / max(value(model.phip[r]), 1e-8))
                    * regy_raw
                )
        if hasattr(model, "yg") and r in model.yg:
            # GAMS benchmark identity (cal.gms): yg = sum_i pa(r,i,gov) * xa(r,i,gov)
            yg_demand = sum(
                value(model.pa[r, i, GTAP_GOVERNMENT_AGENT])
                * value(model.xaa[r, i, GTAP_GOVERNMENT_AGENT])
                for i in model.i
                if (r, i, GTAP_GOVERNMENT_AGENT) in model.pa
                and (r, i, GTAP_GOVERNMENT_AGENT) in model.xaa
            )
            if yg_demand > 0.0:
                model.yg[r].set_value(yg_demand)
            else:
                model.yg[r].set_value(
                    value(model.betag[r]) * value(model.phi[r]) * regy_raw
                )
        if hasattr(model, "rsav") and r in model.rsav:
            # GAMS cal.gms fija rsav.l(r) = save(r), sin condicion de signo, asi
            # que se usa el save del benchmark SIEMPRE que exista --incluido cuando
            # es NEGATIVO--. La rama de respaldo (betas*phi*regy_raw) deriva si
            # regy_raw no coincide con el regY de GAMS, y solo vale cuando el SAM no
            # trae `save`.
            #
            # El guard era `> 0.0`, asi que una region DISSAVER caia al respaldo: en
            # gtap7_3x4, EGY (save=-0.0123) se sembraba en -0.01208, un 2.1% fuera de
            # GAMS. En `pure` el solve lo absorbe, pero en `altertax` --donde los
            # impuestos se mueven y regY se recalcula-- arrancar sesgado manda la
            # region entera a otra rama: 15 celdas de EGY ~1% fuera, y el gate NLP de
            # bloques cae de 99.8% a 98.3%.
            save_bench = ctx.params.benchmark.save.get(str(r), None)
            if save_bench is not None and abs(float(save_bench)) > 1e-12:
                model.rsav[r].set_value(float(save_bench))
            else:
                model.rsav[r].set_value(
                    value(model.betas[r]) * value(model.phi[r]) * regy_raw
                )
        # Recalibrate aus consistently with final rsav.
        # GAMS identity (cal.gms:800): aus.l(r) = us.l*pop.l/(rsav.l/psave.l)
        #                                       = pop.l(r) * psave.l(r) / rsav.l(r)
        # (psave in the NUMERATOR — from pop/(rsav/psave)). This ensures eq_us:
        # us = aus*rsav/(psave*pop) gives us=1 at benchmark. rsav may be NEGATIVE
        # for a dissaving region (EGY) → aus negative, us still +1; gate on
        # abs(rsav)>1e-12 (NOT rsav>1e-12) so dissavers get the consistent recal too.
        if hasattr(model, "aus") and r in model.aus:
            rsav_val = (
                value(model.rsav[r])
                if hasattr(model, "rsav") and r in model.rsav
                else 0.0
            )
            pop_val = value(model.pop[r])
            psave_val = (
                value(model.psave[r])
                if hasattr(model, "psave") and r in model.psave
                else 1.0
            )
            if abs(rsav_val) > 1e-12 and pop_val > 1e-12:
                model.aus[r].set_value(pop_val * psave_val / rsav_val)
        # GAMS cal.gms:619 calibrates betaP from BENCHMARK yc/regY (line ~1787).
        # Re-calibrating from post-init perturbed yc/xaa biases betap away from
        # the GAMS calibration (USA: 0.7634 vs GAMS 0.7772, ROW: 0.6091 vs 0.6322).
        # Trust the original calibration; do not recalibrate here.
        if hasattr(model, "chif") and r in model.chif:
            model.chif[r].set_value(value(model.savf[r]) / regy_val)
        if hasattr(model, "yi") and r in model.yi:
            # Compute yi from the income identity so eq_yi = 0 at init.
            # rsav is already initialized to save_param (the GAMS benchmark),
            # so yi_formula ≈ yi_gams within numerical precision (~2-3e-6).
            # This gives a strictly feasible starting point for eq_yi.
            model.yi[r].set_value(
                value(model.pi[r]) * value(model.depr[r]) * value(model.kstock[r])
                + value(model.rsav[r])
                + value(model.savf[r])
            )
        if hasattr(model, "us") and r in model.us:
            model.us[r].set_value(
                value(model.aus[r])
                * value(model.rsav[r])
                / max(value(model.psave[r]) * value(model.pop[r]), 1e-8)
            )

        if hasattr(model, "pmt"):
            for i in model.i:
                esubm = ctx.params.elasticities.esubm.get((r, i), 5.0)
                expo = 1.0 - esubm
                terms = []
                for rp in model.r:
                    amw = float(
                        ctx.params.shares.normalized.import_source_share.get(
                            (r, i, rp), 0.0
                        )
                        or 0.0
                    )
                    if amw <= 0.0:
                        continue
                    bilateral_exports = float(
                        ctx.params.benchmark.vxmd.get((rp, i, r), 0.0) or 0.0
                    )
                    bilateral_imports = float(
                        ctx.params.benchmark.vcif.get((rp, i, r), 0.0) or 0.0
                    )
                    vxsb_qty = float(
                        ctx.params.benchmark.vxsb.get((rp, i, r), 0.0) or 0.0
                    )
                    if (
                        bilateral_exports <= 0.0
                        and bilateral_imports <= 0.0
                        and vxsb_qty <= 0.0
                    ):
                        continue
                    qty = bilateral_exports if bilateral_exports > 0.0 else vxsb_qty
                    if qty > 0.0 and bilateral_imports > 0.0:
                        pmcif = max(bilateral_imports / qty, 1e-8)
                    elif bilateral_imports > 0.0:
                        pmcif = 1.0
                    else:
                        export_tax = float(
                            ctx.params.taxes.rtxs.get((rp, i, r), 0.0) or 0.0
                        )
                        tmarg = sum(
                            ctx.params.benchmark.vtwr.get((rp, i, r, margin), 0.0)
                            for margin in ctx.sets.m
                        )
                        tmarg = (
                            tmarg / max(bilateral_exports, 1e-12)
                            if bilateral_exports > 0.0
                            else 0.0
                        )
                        pmcif = max(1.0 + export_tax + tmarg, 1e-8)
                    imptx = float(ctx.params.taxes.imptx.get((rp, i, r), 0.0) or 0.0)
                    pm = max((1.0 + imptx) * pmcif, 1e-8)
                    terms.append(amw * (pm**expo))
                if terms:
                    rhs = sum(terms)
                    if rhs > 0.0:
                        model.pmt[r, i].set_value(max(rhs ** (1.0 / expo), 1e-8))

        mqabs_tt = 0.0
        mqabs_t0 = 0.0
        mqabs_0t = 0.0
        mqabs_00 = 0.0
        for i in model.i:
            for agent in final_demand_agents:
                pa_t = float(value(model.pa[r, i, agent]))
                xa_t = float(value(model.xaa[r, i, agent]))
                pa_0 = base_pa[(r, i, agent)]
                xa_0 = base_xaa[(r, i, agent)]
                mqabs_tt += pa_t * xa_t
                mqabs_t0 += pa_t * xa_0
                mqabs_0t += pa_0 * xa_t
                mqabs_00 += pa_0 * xa_0

        mqtrade_tt = 0.0
        mqtrade_t0 = 0.0
        mqtrade_0t = 0.0
        mqtrade_00 = 0.0
        for i in model.i:
            for rp in model.rp:
                pexp_t = float(value(model.pefob[r, i, rp]))
                pexp_0 = base_pefob[(r, i, rp)]
                xexp_t = float(value(model.xw[r, i, rp]))
                xexp_0 = base_xw[(r, i, rp)]
                pimp_t = float(value(model.pmcif[rp, i, r]))
                pimp_0 = base_pmcif[(rp, i, r)]
                ximp_t = float(value(model.xw[rp, i, r]))
                ximp_0 = base_xw[(rp, i, r)]

                mqtrade_tt += pexp_t * xexp_t - pimp_t * ximp_t
                mqtrade_t0 += pexp_t * xexp_0 - pimp_t * ximp_0
                mqtrade_0t += pexp_0 * xexp_t - pimp_0 * ximp_t
                mqtrade_00 += pexp_0 * xexp_0 - pimp_0 * ximp_0

        gdp_current = max(mqabs_tt + mqtrade_tt, 1e-8)
        model.gdpmp[r].set_value(gdp_current)

        if mqabs_00 > 1e-12 and mqabs_0t > 1e-12:
            pabs_fisher = base_pabs[r] * math.sqrt(
                (mqabs_t0 / mqabs_00) * (mqabs_tt / mqabs_0t)
            )
            model.pabs[r].set_value(max(pabs_fisher, 1e-8))

        mqgdp_00 = mqabs_00 + mqtrade_00
        mqgdp_t0 = mqabs_t0 + mqtrade_t0
        mqgdp_0t = mqabs_0t + mqtrade_0t
        if ctx.is_counterfactual:
            if mqgdp_00 > 1e-12 and mqgdp_t0 > 1e-12 and mqgdp_0t > 1e-12:
                rgdp_fisher = base_rgdpmp[r] * math.sqrt(
                    (gdp_current / mqgdp_00) * (mqgdp_0t / mqgdp_t0)
                )
                model.rgdpmp[r].set_value(max(rgdp_fisher, 1e-8))
        else:
            # Baseline: rgdpmp = gdpmp (replicates GAMS rgdpmp.l = gdpmp.l assignment).
            model.rgdpmp[r].set_value(max(gdp_current, 1e-8))
        model.pgdpmp[r].set_value(
            max(gdp_current / max(value(model.rgdpmp[r]), 1e-8), 1e-8)
        )
        pi_val = max(value(model.pi[r]), 1e-8)
        model.xiagg[r].set_value(max(value(model.yi[r]) / pi_val, 1e-8))
        model.kapEnd[r].set_value(
            max(
                (1.0 - value(model.depr[r])) * value(model.kstock[r])
                + value(model.xiagg[r]),
                1e-8,
            )
        )

        cap_return = 0.0
        for f in capital_factors:
            for a in model.a:
                kappa = float(ctx.params.taxes.kappaf_activity.get((r, f, a), 0.0))
                cap_return += (
                    (1.0 - kappa)
                    * value(model.pf[r, f, a])
                    * value(model.xf[r, f, a])
                    / max(value(model.xscale[r, a]), 1e-12)
                )
        arent_val = (
            cap_return / max(value(model.kstock[r]), 1e-8) if capital_factors else 0.0
        )
        model.arent[r].set_value(max(arent_val, 1e-8))
        rorc_val = arent_val / pi_val - value(model.fdepr[r])
        model.rorc[r].set_value(rorc_val)
        rore_val = rorc_val * (
            value(model.kstock[r]) / max(value(model.kapEnd[r]), 1e-8)
        ) ** value(model.rorflex[r])
        model.rore[r].set_value(rore_val)

    xigbl_current = sum(
        value(model.xiagg[r]) - value(model.depr[r]) * value(model.kstock[r])
        for r in model.r
    )
    model.xigbl.set_value(max(xigbl_current, 1e-8))
    pigbl_numer = sum(
        value(model.pi[r])
        * (value(model.xiagg[r]) - value(model.depr[r]) * value(model.kstock[r]))
        for r in model.r
    )
    model.pigbl.set_value(max(pigbl_numer / max(value(model.xigbl), 1e-8), 1e-8))
    rorg_numer = sum(
        value(model.rore[r])
        * value(model.pi[r])
        * (value(model.xiagg[r]) - value(model.depr[r]) * value(model.kstock[r]))
        for r in model.r
    )
    model.rorg.set_value(rorg_numer / max(pigbl_numer, 1e-8))
    residual_gap = sum(
        value(model.yi[r])
        - (
            value(model.pi[r]) * value(model.depr[r]) * value(model.kstock[r])
            + value(model.rsav[r])
            + value(model.savf[r])
        )
        for r in model.r
        if str(r) == ctx.residual_region
    )
    model.walras.set_value(residual_gap)


def _refresh_cet_share_state(model: ConcreteModel, ctx: ScalingContext) -> None:
    """Recalibrate top-level CET shares from the current benchmark-consistent state.

    GAMS calibrates `gd` and `ge` after levels/prices are in place:
    gd = (xds/xs) * (ps/pd)^omegax
    ge = (xet/xs) * (ps/pet)^omegax
    """
    from pyomo.environ import value

    if not hasattr(model, "gd_share") or not hasattr(model, "ge_share"):
        return

    for r in model.r:
        for i in model.i:
            xs_val = max(value(model.xs[r, i]), 1e-12)
            xds_val = max(value(model.xds[r, i]), 0.0)
            xet_val = max(value(model.xet[r, i]), 0.0)
            pd_val = max(value(model.pd[r, i]), 1e-12)
            ps_val = max(value(model.ps[r, i]), 1e-12)
            pet_val = max(value(model.pet[r, i]), 1e-12)
            omega = ctx.params.elasticities.omegax.get((r, i), float("inf"))

            if omega == float("inf"):
                gd_val = (pd_val * xds_val) / max(ps_val * xs_val, 1e-12)
                ge_val = (pet_val * xet_val) / max(ps_val * xs_val, 1e-12)
            else:
                gd_val = (xds_val / xs_val) * (ps_val / pd_val) ** omega
                ge_val = (xet_val / xs_val) * (ps_val / pet_val) ** omega

            model.gd_share[r, i].set_value(max(gd_val, 0.0))
            model.ge_share[r, i].set_value(max(ge_val, 0.0))
