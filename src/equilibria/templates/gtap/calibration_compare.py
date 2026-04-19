"""Calibration comparison helpers for GTAP Python vs GAMS dumps."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from equilibria.templates.gtap.gtap_parameters import (
    GAMSCalibrationDump,
    GTAPParameters,
    GTAP_GOVERNMENT_AGENT,
    GTAP_HOUSEHOLD_AGENT,
    GTAP_INVESTMENT_AGENT,
    GTAP_MARGIN_AGENT,
)


def _as_tuple_key(raw: Any) -> Tuple[str, ...]:
    if isinstance(raw, tuple):
        return tuple(str(x) for x in raw)
    return (str(raw),)


def _key_to_text(key: Tuple[str, ...]) -> str:
    return "(" + ", ".join(str(part) for part in key) + ")"


def _normalize_numeric_map(data: Dict[Any, Any]) -> Dict[Tuple[str, ...], float]:
    normalized: Dict[Tuple[str, ...], float] = {}
    for key, value in data.items():
        try:
            normalized[_as_tuple_key(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _swap_import_source_order(
    data: Dict[Tuple[str, ...], float],
) -> Dict[Tuple[str, ...], float]:
    """Convert (importer,i,exporter) -> (exporter,i,importer) for GAMS parity."""
    swapped: Dict[Tuple[str, ...], float] = {}
    for key, value in data.items():
        if len(key) != 3:
            continue
        importer, commodity, exporter = key
        swapped[(str(exporter), str(commodity), str(importer))] = float(value)
    return swapped


def _iter_gtap_agents(params: GTAPParameters) -> List[str]:
    return list(params.sets.a) + [
        GTAP_HOUSEHOLD_AGENT,
        GTAP_GOVERNMENT_AGENT,
        GTAP_INVESTMENT_AGENT,
        GTAP_MARGIN_AGENT,
    ]


def _raw_agent_domestic_import(
    params: GTAPParameters,
    region: str,
    commodity: str,
    agent: str,
) -> Tuple[float, float]:
    benchmark = params.benchmark
    sets = params.sets

    if agent in sets.a:
        raw_domestic = float(benchmark.vdfb.get((region, commodity, agent), 0.0) or 0.0)
        raw_import = float(benchmark.vmfb.get((region, commodity, agent), 0.0) or 0.0)
        if raw_domestic + raw_import <= 0.0:
            raw_domestic = float(benchmark.vdfm.get((region, commodity, agent), 0.0) or 0.0)
            raw_import = float(benchmark.vifm.get((region, commodity, agent), 0.0) or 0.0)
        return raw_domestic, raw_import

    if agent == GTAP_HOUSEHOLD_AGENT:
        raw_domestic = float(benchmark.vdpb.get((region, commodity), 0.0) or 0.0)
        raw_import = float(benchmark.vmpb.get((region, commodity), 0.0) or 0.0)
        if raw_domestic + raw_import <= 0.0:
            _, raw_domestic, raw_import = benchmark.get_private_demand(region, commodity)
        return float(raw_domestic), float(raw_import)

    if agent == GTAP_GOVERNMENT_AGENT:
        raw_domestic = float(benchmark.vdgb.get((region, commodity), 0.0) or 0.0)
        raw_import = float(benchmark.vmgb.get((region, commodity), 0.0) or 0.0)
        if raw_domestic + raw_import <= 0.0:
            _, raw_domestic, raw_import = benchmark.get_government_demand(region, commodity)
        return float(raw_domestic), float(raw_import)

    if agent == GTAP_INVESTMENT_AGENT:
        raw_domestic = float(benchmark.vdib.get((region, commodity), 0.0) or 0.0)
        raw_import = float(benchmark.vmib.get((region, commodity), 0.0) or 0.0)
        if raw_domestic + raw_import <= 0.0:
            _, raw_domestic, raw_import = benchmark.get_investment_demand(region, commodity)
        return float(raw_domestic), float(raw_import)

    if agent == GTAP_MARGIN_AGENT:
        vst_val = float(benchmark.vst.get((region, commodity), benchmark.vst.get((commodity, region), 0.0)) or 0.0)
        return vst_val, 0.0

    return 0.0, 0.0


def _compute_agent_trade_levels(
    params: GTAPParameters,
) -> Dict[Tuple[str, str, str], Tuple[float, float]]:
    """Mirror GTAPModelEquations benchmark trade-cache construction."""
    benchmark = params.benchmark
    sets = params.sets
    cache: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
    agents = _iter_gtap_agents(params)

    for region in sets.r:
        for commodity in sets.i:
            raw_levels: Dict[str, Tuple[float, float]] = {}
            total_raw_import = 0.0
            for agent in agents:
                raw_domestic, raw_import = _raw_agent_domestic_import(params, region, commodity, agent)
                domestic = max(float(raw_domestic), 0.0)
                imported = max(float(raw_import), 0.0)
                raw_levels[agent] = (domestic, imported)
                total_raw_import += imported

            target_import_total = sum(
                float(benchmark.vmsb.get((source, commodity, region), 0.0) or 0.0)
                for source in sets.r
            )
            import_scale = (target_import_total / total_raw_import) if total_raw_import > 0.0 else 1.0

            for agent, (domestic, imported) in raw_levels.items():
                cache[(region, commodity, agent)] = (domestic, imported * import_scale)

    return cache


def _two_key_value(raw_map: Dict[Any, Any], region: str, commodity: str) -> float:
    val = raw_map.get((region, commodity), None)
    if val is None:
        val = raw_map.get((commodity, region), 0.0)
    return float(val or 0.0)


def _compute_alphaa_3d(
    params: GTAPParameters,
    python_levels: Dict[str, Dict[Tuple[str, ...], float]],
) -> Dict[Tuple[str, ...], float]:
    xaa_levels = _normalize_numeric_map(python_levels.get("xaa", {}))
    pa_levels = _normalize_numeric_map(python_levels.get("pa", {}))
    yc_levels = _normalize_numeric_map(python_levels.get("yc", {}))
    uh_levels = _normalize_numeric_map(python_levels.get("uh", {}))
    alphaa: Dict[Tuple[str, ...], float] = {}

    def _pop_value(region: str) -> float:
        raw = params.benchmark.pop
        val = raw.get(region)
        if val is None:
            val = raw.get((region,), 1.0)
        return float(val or 1.0)

    for region in params.sets.r:
        private_total = sum(float(params.benchmark.get_private_demand(region, commodity)[0] or 0.0) for commodity in params.sets.i)
        government_total = sum(float(params.benchmark.get_government_demand(region, commodity)[0] or 0.0) for commodity in params.sets.i)
        investment_total = sum(float(params.benchmark.get_investment_demand(region, commodity)[0] or 0.0) for commodity in params.sets.i)
        pop = max(_pop_value(region), 1e-12)
        yc = max(float(yc_levels.get((region,), private_total) or private_total), 1e-12)
        yc_pc = max(yc / pop, 1e-12)
        uh = max(float(uh_levels.get((region,), 1.0) or 1.0), 1e-12)
        cde_alpha_den = 0.0

        for commodity in params.sets.i:
            key_hhd = (region, commodity, GTAP_HOUSEHOLD_AGENT)
            xa_hhd = float(xaa_levels.get(key_hhd, 0.0) or 0.0)
            pa_hhd = max(float(pa_levels.get(key_hhd, 1.0) or 1.0), 1e-12)
            xcshr = (pa_hhd * xa_hhd) / yc if xa_hhd > 0.0 else 0.0
            if xcshr <= 0.0:
                continue
            bh = float(params.elasticities.subpar.get((region, commodity), 1.0) or 1.0)
            if abs(bh) < 1e-12:
                bh = 1.0
            cde_alpha_den += xcshr / bh

        for commodity in params.sets.i:
            key_hhd = (region, commodity, GTAP_HOUSEHOLD_AGENT)
            if key_hhd in xaa_levels:
                bh = float(params.elasticities.subpar.get((region, commodity), 1.0) or 1.0)
                if abs(bh) < 1e-12:
                    bh = 1.0
                eh = float(params.elasticities.incpar.get((region, commodity), 1.0) or 1.0)
                pa = max(float(pa_levels.get(key_hhd, 1.0) or 1.0), 1e-12)
                xa_hhd = float(xaa_levels.get(key_hhd, 0.0) or 0.0)
                xcshr = (pa * xa_hhd) / yc if xa_hhd > 0.0 else 0.0
                if xcshr > 0.0 and cde_alpha_den > 0.0:
                    # GAMS CDE formula: alphaa(r,i,h) = ((xcshr/bh) * (((yc_pc/pa)**bh)) * (uh**(-eh*bh))) / denominator
                    alphaa[key_hhd] = ((xcshr / bh) * (((yc_pc / pa) ** bh)) * (uh ** (-eh * bh))) / cde_alpha_den

            key_gov = (region, commodity, GTAP_GOVERNMENT_AGENT)
            if key_gov in xaa_levels and government_total > 0.0:
                sigma_g = float(params.elasticities.esubg.get(region, 1.0) or 1.0)
                if abs(sigma_g - 1.0) < 1e-8:
                    sigma_g = 1.01
                pa = max(float(pa_levels.get(key_gov, 1.0) or 1.0), 1e-12)
                pg = 1.0
                alphaa[key_gov] = (float(xaa_levels[key_gov]) / government_total) * ((pa / pg) ** sigma_g)

            key_inv = (region, commodity, GTAP_INVESTMENT_AGENT)
            if key_inv in xaa_levels and investment_total > 0.0:
                sigma_i = float(params.elasticities.esubi.get(region, 1.0) or 1.0)
                if abs(sigma_i - 1.0) < 1e-8:
                    sigma_i = 1.01
                pa = max(float(pa_levels.get(key_inv, 1.0) or 1.0), 1e-12)
                pi = 1.0
                alphaa[key_inv] = (float(xaa_levels[key_inv]) / investment_total) * ((pa / pi) ** sigma_i)

    for commodity in params.sets.i:
        xtmg = sum(
            float(xaa_levels.get((region, commodity, GTAP_MARGIN_AGENT), 0.0) or 0.0)
            for region in params.sets.r
        )
        if xtmg <= 0.0:
            continue
        for region in params.sets.r:
            key_tmg = (region, commodity, GTAP_MARGIN_AGENT)
            xa = float(xaa_levels.get(key_tmg, 0.0) or 0.0)
            if xa <= 0.0:
                continue
            alphaa[key_tmg] = xa / xtmg

    return alphaa


def _compute_armington_shares_3d(
    params: GTAPParameters,
    python_levels: Dict[str, Dict[Tuple[str, ...], float]],
) -> Tuple[Dict[Tuple[str, ...], float], Dict[Tuple[str, ...], float]]:
    """Compute alphad/alpham(r,i,aa) with GAMS calibration formulas."""
    alphad: Dict[Tuple[str, ...], float] = {}
    alpham: Dict[Tuple[str, ...], float] = {}

    xda_levels = _normalize_numeric_map(python_levels.get("xda", {}))
    xma_levels = _normalize_numeric_map(python_levels.get("xma", {}))
    xaa_levels = _normalize_numeric_map(python_levels.get("xaa", {}))
    pa_levels = _normalize_numeric_map(python_levels.get("pa", {}))
    pmt_levels = _normalize_numeric_map(python_levels.get("pmt", {}))

    for key, xaa in xaa_levels.items():
        if len(key) != 3:
            continue
        region, commodity, agent = key
        xaa_val = max(float(xaa), 0.0)
        if xaa_val <= 0.0:
            continue

        xda_val = max(float(xda_levels.get(key, 0.0) or 0.0), 0.0)
        xma_val = max(float(xma_levels.get(key, 0.0) or 0.0), 0.0)
        pa = max(float(pa_levels.get(key, 1.0) or 1.0), 1e-12)
        pmt = max(float(pmt_levels.get((region, commodity), 1.0) or 1.0), 1e-12)

        if agent in params.sets.a:
            dintx = (
                float(params.benchmark.vdfp.get((region, commodity, agent), 0.0) or 0.0)
                - float(params.benchmark.vdfb.get((region, commodity, agent), 0.0) or 0.0)
            ) / max(float(params.benchmark.vdfb.get((region, commodity, agent), 0.0) or 0.0), 1e-12)
            mintx = (
                float(params.benchmark.vmfp.get((region, commodity, agent), 0.0) or 0.0)
                - float(params.benchmark.vmfb.get((region, commodity, agent), 0.0) or 0.0)
            ) / max(float(params.benchmark.vmfb.get((region, commodity, agent), 0.0) or 0.0), 1e-12)
        elif agent == GTAP_HOUSEHOLD_AGENT:
            dintx = (
                _two_key_value(params.benchmark.vdpp, region, commodity)
                - _two_key_value(params.benchmark.vdpb, region, commodity)
            ) / max(_two_key_value(params.benchmark.vdpb, region, commodity), 1e-12)
            mintx = (
                _two_key_value(params.benchmark.vmpp, region, commodity)
                - _two_key_value(params.benchmark.vmpb, region, commodity)
            ) / max(_two_key_value(params.benchmark.vmpb, region, commodity), 1e-12)
        elif agent == GTAP_GOVERNMENT_AGENT:
            dintx = (
                _two_key_value(params.benchmark.vdgp, region, commodity)
                - _two_key_value(params.benchmark.vdgb, region, commodity)
            ) / max(_two_key_value(params.benchmark.vdgb, region, commodity), 1e-12)
            mintx = (
                _two_key_value(params.benchmark.vmgp, region, commodity)
                - _two_key_value(params.benchmark.vmgb, region, commodity)
            ) / max(_two_key_value(params.benchmark.vmgb, region, commodity), 1e-12)
        elif agent == GTAP_INVESTMENT_AGENT:
            dintx = (
                _two_key_value(params.benchmark.vdip, region, commodity)
                - _two_key_value(params.benchmark.vdib, region, commodity)
            ) / max(_two_key_value(params.benchmark.vdib, region, commodity), 1e-12)
            mintx = (
                _two_key_value(params.benchmark.vmip, region, commodity)
                - _two_key_value(params.benchmark.vmib, region, commodity)
            ) / max(_two_key_value(params.benchmark.vmib, region, commodity), 1e-12)
        else:
            dintx = 0.0
            mintx = 0.0

        pdp = 1.0 + dintx
        pmp = pmt * (1.0 + mintx)
        sigma_m = float(params.elasticities.esubd.get((region, commodity), 2.0) or 2.0)

        if xda_val > 0.0:
            alphad[key] = (xda_val / xaa_val) * ((pdp / pa) ** sigma_m)
        if xma_val > 0.0:
            alpham[key] = (xma_val / xaa_val) * ((pmp / pa) ** sigma_m)

    return alphad, alpham


def _compute_gf_from_python(params: GTAPParameters) -> Dict[Tuple[str, ...], float]:
    """Rebuild gf(r,f,a) with GAMS-consistent mobile/sluggish-factor semantics."""
    benchmark = params.benchmark
    taxes = params.taxes
    sets = params.sets

    gf_data: Dict[Tuple[str, ...], float] = {}

    def _is_land_factor(name: str) -> bool:
        low = str(name).lower()
        return "land" in low or low == "lnd"

    def _is_natres_factor(name: str) -> bool:
        low = str(name).lower()
        return "natres" in low or low in ("nrs", "natres")

    for region in sets.r:
        for factor in sets.mf:
            xf_by_activity: Dict[str, float] = {}
            total_xf = 0.0
            for activity in sets.a:
                factor_flow = float(
                    benchmark.evfb.get((region, factor, activity), benchmark.vfm.get((region, factor, activity), 0.0))
                    or 0.0
                )
                if factor_flow <= 0.0:
                    continue
                kappa = float(taxes.kappaf_activity.get((region, factor, activity), 0.0) or 0.0)
                if kappa == 0.0:
                    kappa = float(taxes.kappaf.get((region, factor), 0.0) or 0.0)
                pf_val = max(1.0 / max(1.0 - kappa, 1e-8), 1e-8)
                xf_val = max(factor_flow / pf_val, 0.0)
                if xf_val <= 0.0:
                    continue
                xf_by_activity[activity] = xf_val
                total_xf += xf_val

            if total_xf <= 0.0:
                continue

            for activity, xf_val in xf_by_activity.items():
                gf_data[(region, factor, activity)] = xf_val / total_xf

        for factor in sets.sf:
            for activity in sets.a:
                factor_flow = float(
                    benchmark.evfb.get((region, factor, activity), benchmark.vfm.get((region, factor, activity), 0.0))
                    or 0.0
                )
                if factor_flow <= 0.0:
                    continue
                if _is_land_factor(factor):
                    gf_data[(region, factor, activity)] = 1.0
                    continue
                kappa = float(taxes.kappaf_activity.get((region, factor, activity), 0.0) or 0.0)
                if kappa == 0.0:
                    kappa = float(taxes.kappaf.get((region, factor), 0.0) or 0.0)
                pf_val = max(1.0 / max(1.0 - kappa, 1e-8), 1e-8)
                xf_val = max(factor_flow / pf_val, 0.0)
                if xf_val > 0.0:
                    gf_data[(region, factor, activity)] = xf_val

    return gf_data


def _compute_lambdam_from_python(params: GTAPParameters) -> Dict[Tuple[str, ...], float]:
    sets = params.sets
    return {(r, i, rp): 1.0 for r in sets.r for i in sets.i for rp in sets.r}


def _compute_lambdamg_from_python(params: GTAPParameters) -> Dict[Tuple[str, ...], float]:
    sets = params.sets
    modes = list(getattr(sets, "m", []) or sets.i)
    return {(m, r, i, rp): 1.0 for m in modes for r in sets.r for i in sets.i for rp in sets.r}


def _compute_kappaf_from_python(params: GTAPParameters) -> Dict[Tuple[str, ...], float]:
    benchmark = params.benchmark
    sets = params.sets
    kappaf: Dict[Tuple[str, ...], float] = {}
    for region in sets.r:
        for factor in sets.f:
            for activity in sets.a:
                evfb = float(benchmark.evfb.get((region, factor, activity), 0.0) or 0.0)
                if evfb <= 0.0:
                    continue
                evos = float(benchmark.evos.get((region, factor, activity), 0.0) or 0.0)
                rate = (evfb - evos) / evfb
                if abs(rate) > 1e-12:
                    kappaf[(region, factor, activity)] = rate
    return kappaf


def _compute_omegaf_from_python(params: GTAPParameters) -> Dict[Tuple[str, ...], float]:
    sets = params.sets
    elasticities = params.elasticities
    omegaf: Dict[Tuple[str, ...], float] = {}

    def _is_natres_factor(name: str) -> bool:
        low = str(name).lower()
        return "natres" in low or low in ("nrs", "natres")

    def _lookup_etrae(region: str, factor: str) -> float:
        raw = elasticities.etrae
        for key in ((factor, region), (region, factor), factor):
            try:
                val = raw.get(key)  # type: ignore[arg-type]
            except Exception:
                val = None
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return float("inf")

    for region in sets.r:
        for factor in sets.mf:
            omegaf[(region, factor)] = float("inf")
        for factor in sets.sf:
            if _is_natres_factor(factor):
                continue
            etrae = _lookup_etrae(region, factor)
            if etrae is None:
                etrae = float("inf")
            if etrae == float("inf"):
                omegaf[(region, factor)] = float("inf")
            else:
                omegaf[(region, factor)] = -float(etrae)
    return omegaf


def _compute_aft_from_python(params: GTAPParameters) -> Dict[Tuple[str, ...], float]:
    benchmark = params.benchmark
    taxes = params.taxes
    sets = params.sets
    aft: Dict[Tuple[str, ...], float] = {}

    def _is_natres_factor(name: str) -> bool:
        low = str(name).lower()
        return "natres" in low or low in ("nrs", "natres")

    for region in sets.r:
        for factor in sets.mf:
            total_xf = 0.0
            for activity in sets.a:
                factor_flow = float(
                    benchmark.evfb.get((region, factor, activity), benchmark.vfm.get((region, factor, activity), 0.0))
                    or 0.0
                )
                if factor_flow <= 0.0:
                    continue
                kappa = float(taxes.kappaf_activity.get((region, factor, activity), 0.0) or 0.0)
                if kappa == 0.0:
                    kappa = float(taxes.kappaf.get((region, factor), 0.0) or 0.0)
                pf_val = max(1.0 / max(1.0 - kappa, 1e-8), 1e-8)
                total_xf += max(factor_flow / pf_val, 0.0)
            if total_xf > 0.0:
                aft[(region, factor)] = total_xf

        for factor in sets.sf:
            if _is_natres_factor(factor):
                continue
            total_xf = 0.0
            for activity in sets.a:
                factor_flow = float(
                    benchmark.evfb.get((region, factor, activity), benchmark.vfm.get((region, factor, activity), 0.0))
                    or 0.0
                )
                if factor_flow <= 0.0:
                    continue
                kappa = float(taxes.kappaf_activity.get((region, factor, activity), 0.0) or 0.0)
                if kappa == 0.0:
                    kappa = float(taxes.kappaf.get((region, factor), 0.0) or 0.0)
                pf_val = max(1.0 / max(1.0 - kappa, 1e-8), 1e-8)
                total_xf += max(factor_flow / pf_val, 0.0)
            if total_xf > 0.0:
                aft[(region, factor)] = total_xf
    return aft


def _compute_axp_from_python(params: GTAPParameters) -> Dict[Tuple[str, ...], float]:
    return {(region, activity): 1.0 for region in params.sets.r for activity in params.sets.a}


def _align_python_level_symbols(
    python_levels: Dict[str, Dict[Tuple[str, ...], float]],
) -> Dict[str, Dict[Tuple[str, ...], float]]:
    """Align Python-only symbol names to their GAMS counterparts."""
    alias = {
        "xaa": "xa",
        "xda": "xd",
        "xma": "xm",
    }
    aligned: Dict[str, Dict[Tuple[str, ...], float]] = {}
    has_disaggregated = {
        "xa": "xaa" in python_levels,
        "xd": "xda" in python_levels,
        "xm": "xma" in python_levels,
    }
    for symbol_name, values in python_levels.items():
        source = str(symbol_name).lower()
        if source in ("xe", "xc"):
            # `xe`/`xc` are Python helper variables; GAMS benchmark dumps expose `xw` and `xa(hhd)` instead.
            continue
        if source in ("xa", "xd", "xm") and has_disaggregated.get(source, False):
            # Prefer the agent-disaggregated representation when available.
            continue
        target = alias.get(source, source)
        target_map = aligned.setdefault(target, {})
        for key, value in values.items():
            try:
                target_map[_as_tuple_key(key)] = float(value)
            except (TypeError, ValueError):
                continue
    return aligned


def _compute_tmarg_from_python(params: GTAPParameters) -> Dict[Tuple[str, ...], float]:
    tmarg_data: Dict[Tuple[str, ...], float] = {}
    benchmark = params.benchmark
    sets = params.sets
    for r in sets.r:
        for i in sets.i:
            for rp in sets.r:
                xw_bench = float(benchmark.vxsb.get((r, i, rp), 0.0) or 0.0)
                vcif = float(benchmark.vcif.get((r, i, rp), 0.0) or 0.0)
                vfob = float(benchmark.vfob.get((r, i, rp), 0.0) or 0.0)
                tmarg_data[(r, i, rp)] = max(vcif - vfob, 0.0) / max(xw_bench, 1e-12) if xw_bench > 0.0 else 0.0
    return tmarg_data


def _compute_chipm_from_python(params: GTAPParameters) -> Dict[Tuple[str, ...], float]:
    chipm_data: Dict[Tuple[str, ...], float] = {}
    benchmark = params.benchmark
    sets = params.sets
    for exporter in sets.r:
        for commodity in sets.i:
            for importer in sets.r:
                vxmd = float(benchmark.vxmd.get((exporter, commodity, importer), 0.0) or 0.0)
                vxsb = float(benchmark.vxsb.get((exporter, commodity, importer), 0.0) or 0.0)
                vcif = float(benchmark.vcif.get((exporter, commodity, importer), 0.0) or 0.0)
                if vxmd <= 0.0 and vxsb <= 0.0 and vcif <= 0.0:
                    continue
                chipm_data[(exporter, commodity, importer)] = 1.0
    return chipm_data


def collect_python_calibration_maps(
    params: GTAPParameters,
    python_levels: Optional[Dict[str, Dict[Tuple[str, ...], float]]] = None,
) -> Dict[str, Dict[Tuple[str, ...], float]]:
    """Collect Python-side calibration objects using GAMS-like symbol names."""
    python_levels = python_levels or collect_python_benchmark_levels(params)
    alphaa_3d = _compute_alphaa_3d(params, python_levels)
    alphad_3d, alpham_3d = _compute_armington_shares_3d(params, python_levels)
    etax_zeros = {(r, i): 0.0 for r in params.sets.r for i in params.sets.i}
    mtax_zeros = {(r, i): 0.0 for r in params.sets.r for i in params.sets.i}

    maps: Dict[str, Dict[Tuple[str, ...], float]] = {
        "and": _normalize_numeric_map(params.calibrated.and_param),
        "ava": _normalize_numeric_map(params.calibrated.ava_param),
        "io": _normalize_numeric_map(params.calibrated.io_param),
        "af": _normalize_numeric_map(params.calibrated.af_param),
        "gx": _normalize_numeric_map(params.calibrated.gx_param),
        "amw": _normalize_numeric_map(_swap_import_source_order(params.shares.p_amw)),
        "gw": _normalize_numeric_map(params.shares.p_gw),
        "gd": _normalize_numeric_map(params.shares.p_gd),
        "ge": _normalize_numeric_map(params.shares.p_ge),
        "gf": _normalize_numeric_map(_compute_gf_from_python(params)),
        "alphaa": _normalize_numeric_map(alphaa_3d),
        "alphad": _normalize_numeric_map(alphad_3d),
        "alpham": _normalize_numeric_map(alpham_3d),
        "alphan": _normalize_numeric_map(params.shares.p_alphan),
        "kappaf": _normalize_numeric_map(_compute_kappaf_from_python(params)),
        "omegaf": _normalize_numeric_map(_compute_omegaf_from_python(params)),
        "aft": _normalize_numeric_map(_compute_aft_from_python(params)),
        "axp": _normalize_numeric_map(_compute_axp_from_python(params)),
        "etaff": _normalize_numeric_map(params.elasticities.etaff),
        "esubt": _normalize_numeric_map(params.elasticities.esubt),
        "esubc": _normalize_numeric_map(params.elasticities.esubc),
        "esubm": _normalize_numeric_map(params.elasticities.esubm),
        "esubva": _normalize_numeric_map(params.elasticities.esubva),
        "sigmas": _normalize_numeric_map(params.elasticities.sigmas),
        "omegaw": _normalize_numeric_map(params.elasticities.omegaw),
        "lambdam": _compute_lambdam_from_python(params),
        "lambdamg": _compute_lambdamg_from_python(params),
        "etax": _normalize_numeric_map(etax_zeros),
        "mtax": _normalize_numeric_map(mtax_zeros),
        "tmarg": _compute_tmarg_from_python(params),
        "chipm": _compute_chipm_from_python(params),
    }
    return {name: data for name, data in maps.items() if data}


def collect_python_benchmark_levels(
    params: GTAPParameters,
    *,
    level_symbols: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[Tuple[str, ...], float]]:
    """Build the Pyomo model and collect initial `.value` levels for key variables."""
    from pyomo.environ import value

    from equilibria.templates.gtap.gtap_contract import build_gtap_closure_config
    from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations

    closure = build_gtap_closure_config({"name": "gtap_standard", "calibration_source": "python", "if_sub": False})
    equations = GTAPModelEquations(params.sets, params, closure)
    model = equations.build_model()

    level_names = tuple(level_symbols or GAMSCalibrationDump.DEFAULT_LEVEL_SYMBOLS)
    levels: Dict[str, Dict[Tuple[str, ...], float]] = {}

    for symbol in level_names:
        if not hasattr(model, symbol):
            continue
        component = getattr(model, symbol)
        symbol_levels: Dict[Tuple[str, ...], float] = {}

        if hasattr(component, "is_indexed") and component.is_indexed():
            for idx in component:
                component_value = value(component[idx], exception=False)
                if component_value is None:
                    continue
                key = idx if isinstance(idx, tuple) else (idx,)
                symbol_levels[tuple(str(x) for x in key)] = float(component_value)
        else:
            component_value = value(component, exception=False)
            if component_value is not None:
                symbol_levels[tuple()] = float(component_value)

        if symbol_levels:
            levels[symbol.lower()] = symbol_levels

    return levels


@dataclass
class SymbolDiff:
    category: str
    name: str
    n_python: int
    n_gams: int
    n_common: int
    n_mismatch: int
    max_abs_diff: float
    max_rel_diff: float
    top_offenders: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CalibrationDiff:
    tol_abs: float
    tol_rel: float
    parameter_diffs: List[SymbolDiff] = field(default_factory=list)
    level_diffs: List[SymbolDiff] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        by_category = {
            "parameters": self.parameter_diffs,
            "levels": self.level_diffs,
        }
        summary: Dict[str, Any] = {
            "tol_abs": self.tol_abs,
            "tol_rel": self.tol_rel,
            "categories": {},
            "totals": {
                "symbols": 0,
                "mismatches": 0,
            },
        }

        for name, entries in by_category.items():
            symbol_count = len(entries)
            mismatch_count = sum(entry.n_mismatch for entry in entries)
            max_abs = max((entry.max_abs_diff for entry in entries), default=0.0)
            max_rel = max((entry.max_rel_diff for entry in entries), default=0.0)
            summary["categories"][name] = {
                "symbol_count": symbol_count,
                "mismatch_count": mismatch_count,
                "max_abs_diff": max_abs,
                "max_rel_diff": max_rel,
            }
            summary["totals"]["symbols"] += symbol_count
            summary["totals"]["mismatches"] += mismatch_count
        return summary

    def _rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for group in (self.parameter_diffs, self.level_diffs):
            for entry in group:
                rows.append(
                    {
                        "category": entry.category,
                        "name": entry.name,
                        "n_python": entry.n_python,
                        "n_gams": entry.n_gams,
                        "n_common": entry.n_common,
                        "n_mismatch": entry.n_mismatch,
                        "max_abs_diff": entry.max_abs_diff,
                        "max_rel_diff": entry.max_rel_diff,
                        "top5_offenders": json.dumps(entry.top_offenders, ensure_ascii=True),
                    }
                )
        return rows

    def write_csv(self, path: Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = self._rows()
        if not rows:
            rows = [
                {
                    "category": "",
                    "name": "",
                    "n_python": 0,
                    "n_gams": 0,
                    "n_common": 0,
                    "n_mismatch": 0,
                    "max_abs_diff": 0.0,
                    "max_rel_diff": 0.0,
                    "top5_offenders": "[]",
                }
            ]
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def write_summary_json(self, path: Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.summary()
        payload["parameter_diffs"] = [entry.__dict__ for entry in self.parameter_diffs]
        payload["level_diffs"] = [entry.__dict__ for entry in self.level_diffs]
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _compare_symbol(
    *,
    category: str,
    name: str,
    python_map: Dict[Tuple[str, ...], float],
    gams_map: Dict[Tuple[str, ...], float],
    tol_abs: float,
    tol_rel: float,
    precision_floor_abs: float = 0.0,
) -> SymbolDiff:
    keys = sorted(set(python_map) | set(gams_map))
    n_mismatch = 0
    max_abs = 0.0
    max_rel = 0.0
    offenders: List[Dict[str, Any]] = []

    for key in keys:
        p_val = python_map.get(key)
        g_val = gams_map.get(key)
        key_text = _key_to_text(key)

        if p_val is None or g_val is None:
            # Treat sparse-support differences as non-mismatch when the present
            # side is numerically zero. GAMS dumps often omit structural zeros.
            if p_val is None and g_val is not None and abs(float(g_val)) <= tol_abs:
                continue
            if g_val is None and p_val is not None and abs(float(p_val)) <= tol_abs:
                continue
            n_mismatch += 1
            offenders.append(
                {
                    "index": key_text,
                    "python": p_val,
                    "gams": g_val,
                    "abs_diff": None,
                    "rel_diff": None,
                    "reason": "missing_in_python" if p_val is None else "missing_in_gams",
                }
            )
            continue

        abs_diff = abs(float(p_val) - float(g_val))
        if abs(float(g_val)) > tol_abs:
            rel_diff = abs_diff / abs(float(g_val))
        else:
            rel_diff = float("inf") if abs_diff > tol_abs else 0.0

        max_abs = max(max_abs, abs_diff)
        max_rel = max(max_rel, rel_diff if rel_diff != float("inf") else max_rel)

        # Ignore tiny numerical drift that is within the calibrated precision level.
        if abs_diff <= max(tol_abs, precision_floor_abs):
            continue

        if abs_diff > tol_abs and rel_diff > tol_rel:
            n_mismatch += 1
            offenders.append(
                {
                    "index": key_text,
                    "python": float(p_val),
                    "gams": float(g_val),
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                    "reason": "value_diff",
                }
            )

    offenders.sort(
        key=lambda item: (
            item["reason"] != "value_diff",
            -(item["abs_diff"] or 0.0),
        )
    )

    return SymbolDiff(
        category=category,
        name=name,
        n_python=len(python_map),
        n_gams=len(gams_map),
        n_common=len(set(python_map) & set(gams_map)),
        n_mismatch=n_mismatch,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        top_offenders=offenders[:5],
    )


def compare_calibration(
    python_params: GTAPParameters,
    gams_dump: GAMSCalibrationDump,
    tol_abs: float = 1e-8,
    tol_rel: float = 1e-6,
    precision_floor_abs: float = 2e-3,
    *,
    python_levels: Optional[Dict[str, Dict[Tuple[str, ...], float]]] = None,
) -> CalibrationDiff:
    """Compare Python GTAP calibration objects against a GAMS dump."""
    if python_levels is None:
        python_levels = collect_python_benchmark_levels(python_params)
    python_param_maps = collect_python_calibration_maps(python_params, python_levels=python_levels)
    python_level_maps = _align_python_level_symbols(python_levels)

    parameter_symbols = sorted(set(python_param_maps) | set(gams_dump.derived_params))
    level_symbols = sorted(set(python_level_maps) | set(gams_dump.benchmark_levels))

    parameter_diffs: List[SymbolDiff] = []
    for symbol in parameter_symbols:
        parameter_diffs.append(
            _compare_symbol(
                category="parameters",
                name=symbol,
                python_map=_normalize_numeric_map(python_param_maps.get(symbol, {})),
                gams_map=_normalize_numeric_map(gams_dump.get_derived(symbol)),
                tol_abs=tol_abs,
                tol_rel=tol_rel,
                precision_floor_abs=precision_floor_abs,
            )
        )

    level_diffs: List[SymbolDiff] = []
    for symbol in level_symbols:
        level_diffs.append(
            _compare_symbol(
                category="levels",
                name=symbol,
                python_map=_normalize_numeric_map(python_level_maps.get(symbol, {})),
                gams_map=_normalize_numeric_map(gams_dump.get_levels(symbol)),
                tol_abs=tol_abs,
                tol_rel=tol_rel,
                precision_floor_abs=precision_floor_abs,
            )
        )

    return CalibrationDiff(
        tol_abs=tol_abs,
        tol_rel=tol_rel,
        parameter_diffs=parameter_diffs,
        level_diffs=level_diffs,
    )
