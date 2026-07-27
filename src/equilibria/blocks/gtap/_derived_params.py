"""Deterministic GTAP model-parameter derivations, transcribed from the monolith.

The GTAP monolith builds its Pyomo Params in ``GTAPModelEquations._add_parameters``
(``gtap_model_equations.py`` 1620-2789) by deriving them ONCE from the calibrated
``GTAPParameters`` + ``GTAPSets`` (plus the closure switch). This is
model-parameter SETUP, not runtime calibration (that is
``apply_production_scaling`` / ``_recalibrate_*``, which the composer/Task-5 owns).

The block units need the SAME Pyomo Params, keyed and valued identically, so their
``build_expression`` bodies reference ``model.<param>[...]`` exactly as the monolith
does and ``form_diff`` stays clean. Rather than duplicate the derivation in every
block, the recipes live here as pure functions of ``(params, sets)`` returning
``{index_tuple: float}`` dicts (the shape ``create_indexed_param`` consumes).

FIDELITY: each function is a verbatim transcription of the corresponding monolith
block — same benchmark headers, same guards, same fallbacks. Line references point
at the monolith source. Do NOT re-derive economics here; if a monolith recipe
changes, mirror it.

KNOWN CARRY (Blocker C, pre-adjudicated to Task 5): ``gd_share``/``ge_share``/
``gw_share`` are seeded here from ``params.shares`` but the monolith's
``apply_production_scaling`` recomputes them at runtime from post-scaling Var
levels. The composer must run that recompute after block registration. On
gtap7_3x3 all ``omegax=inf`` so gd/ge never enter an active (Leontief) body;
``gw_share`` does enter unit-4 bodies (see that unit's notes).
"""

from __future__ import annotations

import math
from typing import Any


def _f(x: Any) -> float:
    return float(x or 0.0)


def to_array(data: dict, elem_lists: list[list[str]], default: float = 0.0):
    """Fill a numpy array from a {index_tuple: value} dict.

    ``elem_lists`` is the element list per domain in order; the array shape is
    the product of their lengths, filled by index-tuple lookup (matching the
    bridge's itertools.product ordering). Missing cells get ``default`` — the
    same semantics as the monolith's create_indexed_param ``default=``.
    """
    import itertools

    import numpy as np

    shape = tuple(len(e) for e in elem_lists)
    arr = np.full(shape, default, dtype=float)
    pos = [{e: k for k, e in enumerate(el)} for el in elem_lists]
    for key, val in data.items():
        ktup = key if isinstance(key, tuple) else (key,)
        if len(ktup) != len(elem_lists):
            continue
        try:
            idx = tuple(pos[d][ktup[d]] for d in range(len(ktup)))
        except KeyError:
            continue
        arr[idx] = float(val)
    return arr


def _gams_round(value: float) -> int:
    """Monolith _gams_round (1638-1641)."""
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def xscale_data(params: Any, sets: Any) -> dict[tuple[str, str], float]:
    """xScale(r,aa) — monolith 1628-1702.

    Aggregate agents (hhd/gov/inv/tmg) → 1.0. Activity columns rescaled to
    xpScale*10**(-round(log10(xp.l))) with xp = nd + va at purchaser value
    (VA includes the (ftrv-fbep)/evfb subsidy/tax wedge).
    """
    from equilibria.templates.gtap.gtap_parameters import (
        GTAP_GOVERNMENT_AGENT,
        GTAP_HOUSEHOLD_AGENT,
        GTAP_INVESTMENT_AGENT,
        GTAP_MARGIN_AGENT,
    )

    bm = params.benchmark
    out: dict[tuple[str, str], float] = {}
    for r in sets.r:
        for aa in (
            GTAP_HOUSEHOLD_AGENT,
            GTAP_GOVERNMENT_AGENT,
            GTAP_INVESTMENT_AGENT,
            GTAP_MARGIN_AGENT,
        ):
            out[(r, aa)] = 1.0

    for r in sets.r:
        for a in sets.a:
            nd_level = sum(
                _f(bm.vdfp.get((r, i, a), 0.0)) + _f(bm.vmfp.get((r, i, a), 0.0))
                for i in sets.i
            )
            if nd_level <= 0.0:
                nd_level = sum(
                    _f(bm.vdfm.get((r, i, a), 0.0)) + _f(bm.vifm.get((r, i, a), 0.0))
                    for i in sets.i
                )
            va_level = 0.0
            for f in sets.f:
                evfb_val = _f(bm.evfb.get((r, f, a), bm.vfm.get((r, f, a), 0.0)))
                if evfb_val <= 0.0:
                    continue
                fbep_val = _f(bm.fbep.get((r, f, a), 0.0))
                ftrv_val = _f(bm.ftrv.get((r, f, a), 0.0))
                va_level += evfb_val + (ftrv_val - fbep_val)
            xp_level = nd_level + va_level
            if xp_level <= 0.0:
                xp_level = _f(bm.vom.get((r, a), 0.0))
            if xp_level <= 0.0:
                xp_level = sum(_f(bm.makb.get((r, a, i), 0.0)) for i in sets.i)
            if xp_level > 0.0:
                exponent = _gams_round(math.log10(xp_level))
                out[(r, a)] = float(10.0 ** (-exponent))
            else:
                out[(r, a)] = 1.0
    return out


def xflag_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    """xflag(r,a,i) = 1 if |makb|>1e-12 — monolith 1802-1809/1826."""
    bm = params.benchmark
    out: dict[tuple[str, str, str], float] = {}
    for r in sets.r:
        for a in sets.a:
            for i in sets.i:
                val = bm.makb.get((r, a, i), 0.0)
                out[(r, a, i)] = 1.0 if abs(_f(val)) > 1e-12 else 0.0
    return out


def _and_ava_nd_pio(params: Any, sets: Any):
    """adjusted_and_param/ava_param/nd_share/p_io — monolith 1865-1940."""
    bm = params.benchmark
    adjusted_and: dict[tuple[str, str], float] = {}
    adjusted_ava: dict[tuple[str, str], float] = {}
    adjusted_nd_share: dict[tuple[str, str], float] = {}
    adjusted_p_io: dict[tuple[str, str, str], float] = {}
    for r in sets.r:
        for a in sets.a:
            adjusted_total_intermediate = 0.0
            adjusted_values: dict[str, float] = {}
            for i in sets.i:
                domestic_val = _f(bm.vdfp.get((r, i, a), 0.0))
                import_val = _f(bm.vmfp.get((r, i, a), 0.0))
                adjusted_val = max(domestic_val, 0.0) + max(import_val, 0.0)
                adjusted_values[i] = adjusted_val
                adjusted_total_intermediate += adjusted_val

            nd_p = sum(
                _f(bm.vdfp.get((r, i, a), 0.0)) + _f(bm.vmfp.get((r, i, a), 0.0))
                for i in sets.i
            )
            if nd_p <= 0.0:
                nd_p = adjusted_total_intermediate
            va_p = 0.0
            for f in sets.f:
                evfb_val = _f(bm.evfb.get((r, f, a), bm.vfm.get((r, f, a), 0.0)))
                if evfb_val <= 0.0:
                    continue
                fbep_val = _f(bm.fbep.get((r, f, a), 0.0))
                ftrv_val = _f(bm.ftrv.get((r, f, a), 0.0))
                va_p += evfb_val + (ftrv_val - fbep_val)
            xp_model_equiv = nd_p + va_p
            adjusted_and[(r, a)] = (
                nd_p / xp_model_equiv if xp_model_equiv > 0.0 else 0.0
            )
            adjusted_ava[(r, a)] = (
                va_p / xp_model_equiv if xp_model_equiv > 0.0 else 0.0
            )
            adjusted_nd_share[(r, a)] = adjusted_and[(r, a)]
            if adjusted_total_intermediate > 0.0:
                for i, adjusted_val in adjusted_values.items():
                    adjusted_p_io[(r, i, a)] = (
                        adjusted_val / adjusted_total_intermediate
                    )
            else:
                for i in sets.i:
                    adjusted_p_io[(r, i, a)] = 0.0
    return adjusted_and, adjusted_ava, adjusted_nd_share, adjusted_p_io


def and_param_data(params: Any, sets: Any) -> dict[tuple[str, str], float]:
    return _and_ava_nd_pio(params, sets)[0]


def ava_param_data(params: Any, sets: Any) -> dict[tuple[str, str], float]:
    return _and_ava_nd_pio(params, sets)[1]


def nd_share_data(params: Any, sets: Any) -> dict[tuple[str, str], float]:
    return _and_ava_nd_pio(params, sets)[2]


def p_io_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    return _and_ava_nd_pio(params, sets)[3]


def io_param_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    """io_param — read straight from params.calibrated (monolith 1960-1966)."""
    return dict(params.calibrated.io_param)


def af_param_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    """af_param — read straight from params.calibrated (monolith 1975-1981)."""
    return dict(params.calibrated.af_param)


def gx_param_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    """gx_param — read straight from params.calibrated (monolith 1987-1993).

    CARRY: apply_production_scaling recomputes gx per period; the seed is used
    at build (matches the oracle which also seeds from calibrated before scaling
    leaves gx unchanged on the base period)."""
    return dict(params.calibrated.gx_param)


def p_ax_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    """p_ax — read straight from params.shares (monolith 1997-1999)."""
    return dict(params.shares.p_ax)


def lambdaio_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    """lambdaio — params.shifts if present else default 1.0 (monolith 1967-1974)."""
    return dict(params.shifts.lambdaio)


def gd_share_data(params: Any, sets: Any) -> dict[tuple[str, str], float]:
    """gd_share — seed from params.shares.p_gd (monolith 2000-2002). CARRY."""
    return dict(params.shares.p_gd)


def ge_share_data(params: Any, sets: Any) -> dict[tuple[str, str], float]:
    """ge_share — seed from params.shares.p_ge (monolith 2003-2005). CARRY."""
    return dict(params.shares.p_ge)


# ----------------------------------------------------------------------------
# FACTOR unit (monolith 1801-2135, 2280-2401, 5065-5094)
# ----------------------------------------------------------------------------


def xfflag_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    """xfflag(r,f,a) = 1 if |evfb (else vfm)|>1e-12 — monolith 1810-1820."""
    bm = params.benchmark
    out: dict[tuple[str, str, str], float] = {}
    for r in sets.r:
        for f in sets.f:
            for a in sets.a:
                val = bm.evfb.get((r, f, a), bm.vfm.get((r, f, a), 0.0))
                out[(r, f, a)] = 1.0 if abs(_f(val)) > 1e-12 else 0.0
    return out


def xftflag_data(params: Any, sets: Any) -> dict[tuple[str, str], float]:
    """xftflag(r,f) = 1 if any (r,f,a) flow and f in mf|sf — monolith 1810-1825."""
    bm = params.benchmark
    mfsf = set(sets.mf) | set(sets.sf)
    out: dict[tuple[str, str], float] = {}
    for r in sets.r:
        for f in sets.f:
            any_flow = False
            for a in sets.a:
                val = bm.evfb.get((r, f, a), bm.vfm.get((r, f, a), 0.0))
                if abs(_f(val)) > 1e-12:
                    any_flow = True
            out[(r, f)] = 1.0 if (any_flow and f in mfsf) else 0.0
    return out


def _kappaf(params: Any, r: str, f: str, a: str) -> float:
    """kappaf lookup (activity then factor) — monolith 2059-2065 pattern."""
    kappa = _f(params.taxes.kappaf_activity.get((r, f, a), 0.0))
    if kappa == 0.0:
        kappa = _f(params.taxes.kappaf.get((r, f), 0.0))
    return kappa


def gf_share_data(
    params: Any, sets: Any, use_p_gf_directly: bool = False
) -> dict[tuple[str, str, str], float]:
    """gf_share(r,f,a) — monolith 2033-2135. CARRY (Blocker C family).

    ``use_p_gf_directly`` mirrors the monolith's ``self.t0_snapshot is not None``
    switch (2040): for a counterfactual period the composer sets True to trust
    the recalibrated ``p_gf``; for the base build (the gate oracle) it is False
    and mf/sf shares are (re)normalized from the benchmark SAM here.
    """
    bm = params.benchmark
    out: dict[tuple[str, str, str], float] = dict(params.shares.p_gf)
    for r in sets.r:
        for f in sets.mf:
            if use_p_gf_directly:
                pass
            else:
                xf_by_activity: dict[str, float] = {}
                total_xf = 0.0
                for a in sets.a:
                    factor_flow = _f(bm.evfb.get((r, f, a), bm.vfm.get((r, f, a), 0.0)))
                    if factor_flow <= 0.0:
                        continue
                    kappa = _kappaf(params, r, f, a)
                    pf_val = max(1.0 / max(1.0 - kappa, 1e-8), 1e-8)
                    xf_val = max(factor_flow / pf_val, 0.0)
                    if xf_val <= 0.0:
                        continue
                    xf_by_activity[a] = xf_val
                    total_xf += xf_val
                if total_xf <= 0.0:
                    continue
                for a in sets.a:
                    out[(r, f, a)] = xf_by_activity.get(a, 0.0) / total_xf
        for f in sets.sf:
            xf_by_activity_sf: dict[str, float] = {}
            total_xf_sf = 0.0
            for a in sets.a:
                factor_flow = _f(bm.evfb.get((r, f, a), bm.vfm.get((r, f, a), 0.0)))
                if factor_flow <= 0.0:
                    continue
                kappa = _kappaf(params, r, f, a)
                pf_val = max(1.0 / max(1.0 - kappa, 1e-8), 1e-8)
                xf_val = max(factor_flow / pf_val, 0.0)
                if xf_val <= 0.0:
                    continue
                xf_by_activity_sf[a] = xf_val
                total_xf_sf += xf_val
            if total_xf_sf <= 0.0:
                continue
            for a in sets.a:
                out[(r, f, a)] = xf_by_activity_sf.get(a, 0.0) / total_xf_sf
        fnm_factors = [f for f in sets.f if f not in sets.mf and f not in sets.sf]
        for f in fnm_factors:
            for a in sets.a:
                factor_flow = _f(bm.evfb.get((r, f, a), bm.vfm.get((r, f, a), 0.0)))
                if factor_flow <= 0.0:
                    out[(r, f, a)] = 0.0
                    continue
                kappa = _kappaf(params, r, f, a)
                pf_val = max(1.0 / max(1.0 - kappa, 1e-8), 1e-8)
                xf_val = max(factor_flow / pf_val, 0.0)
                out[(r, f, a)] = xf_val
    return out


def _lookup_etrae(params: Any, region: str, factor: str) -> float:
    """etrae lookup by (f,r)/(r,f)/f — monolith 2292-2311."""
    raw = params.elasticities.etrae
    for key in ((factor, region), (region, factor), factor):
        try:
            val = raw.get(key)
        except Exception:
            val = None
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return 0.0


def _aft_etaf(params: Any, sets: Any, closure_name: str = "gtap_standard"):
    """aft(r,f)/etaf(r,f) — monolith 2360-2401.

    ``closure_name`` mirrors ``self.closure.name`` (2394): only ``"altertax"``
    keeps the ``-etrae`` supply elasticity; every other closure (incl. the gate
    oracle's default ``gtap_standard``) sets etaf=0 for all fm/sf. The composer
    passes the real closure name in Task 5.
    """
    bm = params.benchmark
    aft: dict[tuple[str, str], float] = {}
    etaf: dict[tuple[str, str], float] = {}
    for region in sets.r:
        for factor in sets.f:
            if factor in sets.mf or factor in sets.sf:
                benchmark_xft = 0.0
                for activity in sets.a:
                    factor_flow = _f(
                        bm.evfb.get(
                            (region, factor, activity),
                            bm.vfm.get((region, factor, activity), 0.0),
                        )
                    )
                    if factor_flow <= 0.0:
                        continue
                    kappa = _kappaf(params, region, factor, activity)
                    pf_val = max(1.0 / max(1.0 - kappa, 1e-8), 1e-8)
                    benchmark_xft += factor_flow / pf_val
                aft[(region, factor)] = benchmark_xft
                if factor in sets.mf:
                    etaf[(region, factor)] = 0.0
                elif str(closure_name) != "altertax":
                    # gtap-mode: GAMS uses etaf=0 for ALL fm incl sluggish Land
                    # (getData.gms:367-380); etrae belongs only in omegaf (the CET
                    # nest), NOT in the eq_xfteq supply elasticity. Altertax keeps
                    # the -etrae form (byte-identical). Monolith 2394-2401.
                    etaf[(region, factor)] = 0.0
                else:
                    etaf[(region, factor)] = _lookup_etrae(params, region, factor)
    return aft, etaf


def aft_data(
    params: Any, sets: Any, closure_name: str = "gtap_standard"
) -> dict[tuple[str, str], float]:
    return _aft_etaf(params, sets, closure_name)[0]


def etaf_data(
    params: Any, sets: Any, closure_name: str = "gtap_standard"
) -> dict[tuple[str, str], float]:
    return _aft_etaf(params, sets, closure_name)[1]


def fcttx_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    """fcttx(r,f,a) = ftrv/evfb (0 if evfb<=0) — monolith 5065-5069."""
    bm = params.benchmark
    out: dict[tuple[str, str, str], float] = {}
    for r in sets.r:
        for f in sets.f:
            for a in sets.a:
                evfb_val = _f(bm.evfb.get((r, f, a), 0.0))
                if evfb_val <= 0.0:
                    out[(r, f, a)] = 0.0
                else:
                    out[(r, f, a)] = _f(bm.ftrv.get((r, f, a), 0.0)) / evfb_val
    return out


def fctts_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    """fctts(r,f,a) = -fbep/evfb (0 if evfb<=0) — monolith 5071-5075."""
    bm = params.benchmark
    out: dict[tuple[str, str, str], float] = {}
    for r in sets.r:
        for f in sets.f:
            for a in sets.a:
                evfb_val = _f(bm.evfb.get((r, f, a), 0.0))
                if evfb_val <= 0.0:
                    out[(r, f, a)] = 0.0
                else:
                    out[(r, f, a)] = -_f(bm.fbep.get((r, f, a), 0.0)) / evfb_val
    return out


# ----------------------------------------------------------------------------
# ARMINGTON + BILATERAL unit (monolith 2167-2252, 6038-6631, 2993-3145)
# ----------------------------------------------------------------------------

# GTAP aggregate-agent labels (monolith imports these from gtap_parameters).
_HHD, _GOV, _INV, _TMG = "hhd", "gov", "inv", "tmg"


def _two_key(raw_map: Any, region: str, commodity: str) -> float:
    """(region,commodity) then (commodity,region) fallback — monolith _two_key."""
    val = raw_map.get((region, commodity), None)
    if val is None:
        val = raw_map.get((commodity, region), 0.0)
    return float(val or 0.0)


def _vst_value(params: Any, region: str, commodity: str) -> float:
    """VST(region,commodity) with (r,i)/(i,r) tolerance — monolith 97-106."""
    val = params.benchmark.vst.get((region, commodity))
    if val is None:
        val = params.benchmark.vst.get((commodity, region), 0.0)
    return float(val or 0.0)


def tmarg_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    """tmarg(r,i,rp) = max(vcif-vfob,0)/xw_bench — monolith 2168-2188."""
    bm = params.benchmark
    out: dict[tuple[str, str, str], float] = {}
    for r in sets.r:
        for i in sets.i:
            for rp in sets.r:
                xw_bench = _f(bm.vxsb.get((r, i, rp), 0.0))
                vcif = _f(bm.vcif.get((r, i, rp), 0.0))
                vfob = _f(bm.vfob.get((r, i, rp), 0.0))
                out[(r, i, rp)] = (
                    max(vcif - vfob, 0.0) / max(xw_bench, 1e-12)
                    if xw_bench > 0.0
                    else 0.0
                )
    return out


def amgm_data(params: Any, sets: Any) -> dict[tuple[str, str, str, str], float]:
    """amgm(m,r,i,rp) = vtwr(m)/sum_m vtwr — monolith 2190-2193."""
    bm = params.benchmark
    out: dict[tuple[str, str, str, str], float] = {}
    for r in sets.r:
        for i in sets.i:
            for rp in sets.r:
                margin_flow = sum(_f(bm.vtwr.get((r, i, rp, m), 0.0)) for m in sets.m)
                for m in sets.m:
                    flow = _f(bm.vtwr.get((r, i, rp, m), 0.0))
                    out[(m, r, i, rp)] = (
                        flow / margin_flow if margin_flow > 0.0 else 0.0
                    )
    return out


def lambdamg_data(params: Any, sets: Any) -> dict[tuple[str, str, str, str], float]:
    """lambdamg(m,r,i,rp) = 1.0 — monolith 2194."""
    out: dict[tuple[str, str, str, str], float] = {}
    for r in sets.r:
        for i in sets.i:
            for rp in sets.r:
                for m in sets.m:
                    out[(m, r, i, rp)] = 1.0
    return out


def xw_flag_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    """xw_flag(r,i,rp) = 1 if vxsb>0 — monolith 2009-2016."""
    bm = params.benchmark
    out: dict[tuple[str, str, str], float] = {}
    for r in sets.r:
        for i in sets.i:
            for rp in sets.r:
                out[(r, i, rp)] = 1.0 if _f(bm.vxsb.get((r, i, rp), 0.0)) > 0.0 else 0.0
    return out


def xet_flag_data(params: Any, sets: Any) -> dict[tuple[str, str], float]:
    """xet_flag(r,i) = 1.0 for every (r,i) — monolith 2018-2024."""
    return {(r, i): 1.0 for r in sets.r for i in sets.i}


def gw_share_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    """gw_share(r,i,rp) — seed from params.shares.p_gw (monolith 2006-2007). CARRY."""
    return dict(params.shares.p_gw)


def imptx_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    """imptx(r,i,rp) = (vmsb-vcif)/vcif else raw imptx — monolith 5003-5008.

    Mutable in the monolith (5017-5024) and referenced UNWRAPPED in eq_pmeq via
    _imptx_rate_importer (5419: return model.imptx[...]), so it must stay symbolic.
    """
    imptx_p = getattr(params.taxes, "imptx", {}) or {}
    bm = params.benchmark
    vcif_p = getattr(bm, "vcif", {})
    vmsb_p = getattr(bm, "vmsb", {})
    out: dict[tuple[str, str, str], float] = {}
    for r in sets.r:
        for i in sets.i:
            for rp in sets.r:
                vcif = _f(vcif_p.get((r, i, rp), 0.0))
                vmsb = _f(vmsb_p.get((r, i, rp), 0.0))
                if vcif > 1e-12 and (r, i, rp) not in imptx_p:
                    out[(r, i, rp)] = (vmsb - vcif) / vcif
                else:
                    out[(r, i, rp)] = _f(imptx_p.get((r, i, rp), 0.0))
    return out


def _dintx_target(params: Any, sets: Any, r: str, i: str, aa: str) -> float:
    """get_dintx_init body (benchmark branch) — monolith 3021-3058."""
    bm = params.benchmark
    if aa in sets.a:
        numerator = _f(bm.vdfp.get((r, i, aa), 0.0)) - _f(bm.vdfb.get((r, i, aa), 0.0))
        denom = max(_f(bm.vdfb.get((r, i, aa), 0.0)), 0.0)
    elif aa == _HHD:
        numerator = _two_key(bm.vdpp, r, i) - _two_key(bm.vdpb, r, i)
        denom = max(_two_key(bm.vdpb, r, i), 0.0)
    elif aa == _GOV:
        numerator = _two_key(bm.vdgp, r, i) - _two_key(bm.vdgb, r, i)
        denom = max(_two_key(bm.vdgb, r, i), 0.0)
    elif aa == _INV:
        numerator = _two_key(bm.vdip, r, i) - _two_key(bm.vdib, r, i)
        denom = max(_two_key(bm.vdib, r, i), 0.0)
    elif aa == _TMG:
        return 0.0
    else:
        numerator = 0.0
        denom = 0.0
    if denom > 0.0:
        return numerator / denom
    return _f(params.taxes.dintx0.get((r, i, aa), 0.0))


def _mintx_target(params: Any, sets: Any, r: str, i: str, aa: str) -> float:
    """get_mintx_init body (benchmark branch) — monolith 3060-3097."""
    bm = params.benchmark
    if aa in sets.a:
        numerator = _f(bm.vmfp.get((r, i, aa), 0.0)) - _f(bm.vmfb.get((r, i, aa), 0.0))
        denom = max(_f(bm.vmfb.get((r, i, aa), 0.0)), 0.0)
    elif aa == _HHD:
        numerator = _two_key(bm.vmpp, r, i) - _two_key(bm.vmpb, r, i)
        denom = max(_two_key(bm.vmpb, r, i), 0.0)
    elif aa == _GOV:
        numerator = _two_key(bm.vmgp, r, i) - _two_key(bm.vmgb, r, i)
        denom = max(_two_key(bm.vmgb, r, i), 0.0)
    elif aa == _INV:
        numerator = _two_key(bm.vmip, r, i) - _two_key(bm.vmib, r, i)
        denom = max(_two_key(bm.vmib, r, i), 0.0)
    elif aa == _TMG:
        return 0.0
    else:
        numerator = 0.0
        denom = 0.0
    if denom > 0.0:
        return numerator / denom
    return _f(params.taxes.mintx0.get((r, i, aa), 0.0))


def _raw_agent_dom_imp(params: Any, sets: Any, r: str, i: str, aa: str):
    """_raw_agent_domestic_import_eq body — monolith 6086-6120."""
    bm = params.benchmark
    if aa in sets.a:
        raw_domestic = bm.vdfb.get((r, i, aa), 0.0)
        raw_import = bm.vmfb.get((r, i, aa), 0.0)
        if raw_domestic + raw_import <= 0.0:
            raw_domestic = bm.vdfm.get((r, i, aa), 0.0)
            raw_import = bm.vifm.get((r, i, aa), 0.0)
    elif aa == _HHD:
        raw_domestic = bm.vdpb.get((r, i), 0.0)
        raw_import = bm.vmpb.get((r, i), 0.0)
        if raw_domestic + raw_import <= 0.0:
            _, raw_domestic, raw_import = bm.get_private_demand(r, i)
    elif aa == _GOV:
        raw_domestic = bm.vdgb.get((r, i), 0.0)
        raw_import = bm.vmgb.get((r, i), 0.0)
        if raw_domestic + raw_import <= 0.0:
            _, raw_domestic, raw_import = bm.get_government_demand(r, i)
    elif aa == _INV:
        raw_domestic = bm.vdib.get((r, i), 0.0)
        raw_import = bm.vmib.get((r, i), 0.0)
        if raw_domestic + raw_import <= 0.0:
            _, raw_domestic, raw_import = bm.get_investment_demand(r, i)
    elif aa == _TMG:
        raw_domestic = _vst_value(params, str(r), str(i))
        raw_import = 0.0
    else:
        raw_domestic = 0.0
        raw_import = 0.0
    return max(float(raw_domestic or 0.0), 0.0), max(float(raw_import or 0.0), 0.0)


def _agent_list(sets: Any) -> list[str]:
    return list(sets.a) + [_HHD, _GOV, _INV, _TMG]


def armington_shares(
    params: Any, sets: Any
) -> dict[tuple[str, str, str], tuple[float, float]]:
    """(alphad, alpham) per (r,i,aa) — monolith 6122-6229, BENCHMARK-seed variant.

    CARRY (Blocker C / unit-4 pre-adjudicated): the monolith computes these from
    ``t0_arm`` = the model's POST-apply_production_scaling Var levels
    (value(model.xaa/xda/xma/pdp/pmp/pa)). A block cannot read those, so this
    reproduces the SAME algebra from the benchmark SEED (pa=1, pdp=1+dintx0,
    pmp=1+mintx0, xda/xma from the agent-trade cache, xaa = purchaser value).
    On active CES cells the alphad/alpham coefficient therefore DRIFTS from the
    oracle's post-scaling value; the composer re-runs this with the scaled model
    (same family as gd/ge/gw). The equation STRUCTURE and Skip mask are identical.
    """
    esubd = params.elasticities.esubd
    # agent-trade cache (import-scaled domestic/import levels), monolith 6122-6157.
    # get_xda_init = domestic/pd (pd=1); get_xma_init = imported (both from cache).
    trade = _agent_trade_cache(params, sets)
    out: dict[tuple[str, str, str], tuple[float, float]] = {}
    for r in sets.r:
        for i in sets.i:
            for aa in _agent_list(sets):
                dom, imp = trade.get((r, i, aa), (0.0, 0.0))
                xda_bench = max(dom, 0.0)  # /pd_bench=1
                xma_bench = max(imp, 0.0)
                xaa_bench = _xaa_purchaser_value(params, sets, r, i, aa)
                if xaa_bench <= 0.0 or (xda_bench <= 0.0 and xma_bench <= 0.0):
                    out[(r, i, aa)] = (0.0, 0.0)
                    continue
                sigma_m = float(esubd.get((r, i), 2.0))
                pdp_bench = max(1.0 + _dintx_target(params, sets, r, i, aa), 1e-8)
                pmp_bench = max(1.0 + _mintx_target(params, sets, r, i, aa), 1e-8)
                pa_bench = 1.0
                # SURGICAL xaa consistency fix (monolith 6207-6214)
                _xaa_consistent = (
                    pdp_bench * xda_bench + pmp_bench * xma_bench
                ) / pa_bench
                if (
                    _xaa_consistent > 0.0
                    and abs(_xaa_consistent / xaa_bench - 1.0) > 0.01
                ):
                    xaa_bench = _xaa_consistent
                alphad = (
                    (xda_bench / xaa_bench) * (pdp_bench / pa_bench) ** sigma_m
                    if xda_bench > 0.0
                    else 0.0
                )
                alpham = (
                    (xma_bench / xaa_bench) * (pmp_bench / pa_bench) ** sigma_m
                    if xma_bench > 0.0
                    else 0.0
                )
                out[(r, i, aa)] = (alphad, alpham)
    return out


def _agent_trade_cache(
    params: Any, sets: Any
) -> dict[tuple[str, str, str], tuple[float, float]]:
    """domestic/import agent-trade levels with import scaling — monolith 6122-6157."""
    bm = params.benchmark
    out: dict[tuple[str, str, str], tuple[float, float]] = {}
    for r in sets.r:
        for i in sets.i:
            raw_levels: dict[str, tuple[float, float]] = {}
            total_raw_import = 0.0
            for aa in _agent_list(sets):
                dom, imp = _raw_agent_dom_imp(params, sets, r, i, aa)
                raw_levels[aa] = (dom, imp)
                total_raw_import += imp
            target_import_total = sum(_f(bm.vmsb.get((rp, i, r), 0.0)) for rp in sets.r)
            import_scale = (
                (target_import_total / total_raw_import)
                if total_raw_import > 0.0
                else 1.0
            )
            for aa, (dom, imp) in raw_levels.items():
                out[(r, i, aa)] = (dom, imp * import_scale)
    return out


def _xaa_purchaser_value(params: Any, sets: Any, r: str, i: str, aa: str) -> float:
    """get_xaa_purchaser_value_init — monolith 3103-3136."""
    bm = params.benchmark
    if aa in sets.a:
        return max(
            _f(bm.vdfp.get((r, i, aa), 0.0)) + _f(bm.vmfp.get((r, i, aa), 0.0)), 0.0
        )
    if aa == _HHD:
        return max(_two_key(bm.vdpp, r, i) + _two_key(bm.vmpp, r, i), 0.0)
    if aa == _GOV:
        return max(_two_key(bm.vdgp, r, i) + _two_key(bm.vmgp, r, i), 0.0)
    if aa == _INV:
        return max(_two_key(bm.vdip, r, i) + _two_key(bm.vmip, r, i), 0.0)
    if aa == _TMG:
        return max(_vst_value(params, str(r), str(i)), 0.0)
    return 0.0


def prdtx_rai_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    """prdtx_rai(r,a,i) = makb/maks - 1 (else rto) — monolith 2146-2165."""
    bm = params.benchmark
    out: dict[tuple[str, str, str], float] = {}
    for r in sets.r:
        for a in sets.a:
            outputs = sets.activity_commodities.get(a, [])
            if not outputs:
                outputs = list(sets.i)
            for i in outputs:
                makb_val = _f(bm.makb.get((r, a, i), 0.0))
                maks_val = _f(bm.maks.get((r, a, i), 0.0))
                if maks_val > 0 and makb_val > 0:
                    out[(r, a, i)] = makb_val / maks_val - 1.0
                else:
                    out[(r, a, i)] = _f(params.taxes.rto.get((r, a), 0.0))
    return out
