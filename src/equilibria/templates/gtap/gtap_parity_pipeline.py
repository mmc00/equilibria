"""GTAP GAMS parity helpers.

This module compares the Python GTAP template against GAMS reference outputs.

It supports two data layouts:

1. A single benchmark/result GDX, which is the original interface.
2. A `standard_gtap_7` style bundle where:
   - sets come from `*Sets.gdx`
   - elasticities come from `*Prm.gdx`
   - benchmark and reference values come from `COMP.csv`

The CSV support exists because NEOS/GAMS 52 result GDX files are not yet parsed
reliably by the local pure-Python GDX reader.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values, read_variable_values
from equilibria.templates.gtap import (
    GTAPModelEquations,
    GTAPParameters,
    GTAPSets,
    GTAPSolver,
    build_gtap_contract,
)
from equilibria.templates.gtap.gtap_parameters import GTAPBenchmarkValues
from equilibria.templates.gtap.gtap_solver import SolverResult

logger = logging.getLogger(__name__)

_STANDARD_GTAP_GENERATED_SCALE_CACHE: Dict[Path, Dict[Tuple[str, str, str, str, str], float]] = {}


def _is_standard_gtap_9x10_sets(sets: Optional[GTAPSets]) -> bool:
    """Return whether GTAP sets match the 9x10 agricultural aggregation."""
    if sets is None:
        return False
    activities = {str(activity) for activity in getattr(sets, "a", [])}
    return "a_agricultur" in activities and "a_Crops" not in activities


def _normalize_year_token(value: Any) -> str:
    """Normalize a year token from the CSV or caller input."""
    text = str(value).strip()
    if not text:
        return text
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return text


def _as_float(value: Any) -> float:
    """Convert text values to float, tolerating blanks."""
    text = str(value).strip()
    if not text:
        return 0.0
    if text.upper() in {"UNDF", "NA", "NULL"}:
        return 0.0
    return float(text)


def _row_matches_year(row: Dict[str, str], year: str) -> bool:
    """Return whether a postsim CSV row belongs to the selected year label."""
    return _normalize_year_token(row.get("Year", "")) == year


def _clean_label(value: Any) -> str:
    """Normalize a CSV label cell."""
    return str(value).strip()


def _extract_var_levels(var) -> Dict[Any, float]:
    """Extract Pyomo variable values into a plain dictionary."""
    from pyomo.environ import value

    result: Dict[Any, float] = {}
    for idx in var:
        try:
            result[idx] = float(value(var[idx]))
        except Exception:
            result[idx] = 0.0
    return result


def _extract_xda_unscaled(model) -> Dict[Any, float]:
    """Extract xda levels and remove xscale so values align with GTAP `xd` semantics."""
    from pyomo.environ import value

    if not hasattr(model, "xda"):
        return {}

    result: Dict[Any, float] = {}
    for idx in model.xda:
        try:
            level = float(value(model.xda[idx]))
            if hasattr(model, "xscale") and isinstance(idx, tuple) and len(idx) == 3:
                region = idx[0]
                agent = idx[2]
                scale = float(value(model.xscale[region, agent]))
                if abs(scale) > 1e-12:
                    level /= scale
            result[idx] = level
        except Exception:
            result[idx] = 0.0
    return result


def _extract_xaa_unscaled(model) -> Dict[Any, float]:
    """Extract xaa levels and remove xscale so values align with GTAP `xa` semantics."""
    from pyomo.environ import value

    if not hasattr(model, "xaa"):
        return {}

    result: Dict[Any, float] = {}
    for idx in model.xaa:
        try:
            level = float(value(model.xaa[idx]))
            if hasattr(model, "xscale") and isinstance(idx, tuple) and len(idx) == 3:
                region = idx[0]
                agent = idx[2]
                scale = float(value(model.xscale[region, agent]))
                if abs(scale) > 1e-12:
                    level /= scale
            result[idx] = level
        except Exception:
            result[idx] = 0.0
    return result


def _normalize_singleton_keys(values: Dict[Any, float]) -> Dict[Tuple[str], float]:
    """Normalize 1D variable keys to one-element tuple form.

    This avoids mixed key shapes (e.g. ``"c_Food"`` vs ``("c_Food",)``)
    when comparing snapshots built from different backends.
    """
    normalized: Dict[Tuple[str], float] = {}
    for key, value in values.items():
        if isinstance(key, tuple):
            if len(key) == 1:
                normalized[(str(key[0]),)] = float(value)
            else:
                continue
        else:
            normalized[(str(key),)] = float(value)
    return normalized


def _extract_gtap_factor_prices(model) -> Dict[Any, float]:
    """Extract factor prices using GTAP gross-price semantics when available."""
    from pyomo.environ import value

    if not hasattr(model, "pf"):
        return {}

    pf_values = _extract_var_levels(model.pf)
    if getattr(model, "_equilibria_factor_price_representation", "") != "net_of_direct_tax":
        return pf_values
    if not hasattr(model, "kappaf_activity"):
        return pf_values

    gross_values: Dict[Any, float] = {}
    for idx, net_value in pf_values.items():
        try:
            wedge = float(value(model.kappaf_activity[idx]))
        except Exception:
            wedge = 0.0
        gross_values[idx] = float(net_value) / max(1.0 - wedge, 1e-6)
    return gross_values


def _apply_output_pairs(sets: GTAPSets, pairs: Iterable[Tuple[str, str]]) -> None:
    """Populate GTAP output-structure metadata from explicit make pairs."""
    unique_pairs: List[Tuple[str, str]] = []
    for activity, commodity in pairs:
        if activity not in sets.a or commodity not in sets.i:
            continue
        pair = (activity, commodity)
        if pair not in unique_pairs:
            unique_pairs.append(pair)

    sets.output_pairs = unique_pairs
    sets.activity_commodities = {activity: [] for activity in sets.a}
    sets.commodity_activities = {commodity: [] for commodity in sets.i}
    sets.a_to_i = {}
    sets.i_to_a = {}

    for activity, commodity in unique_pairs:
        sets.activity_commodities.setdefault(activity, [])
        sets.commodity_activities.setdefault(commodity, [])
        if commodity not in sets.activity_commodities[activity]:
            sets.activity_commodities[activity].append(commodity)
        if activity not in sets.commodity_activities[commodity]:
            sets.commodity_activities[commodity].append(activity)

    if not unique_pairs:
        return

    if (
        all(len(outputs) == 1 for outputs in sets.activity_commodities.values())
        and all(len(activities) == 1 for activities in sets.commodity_activities.values())
    ):
        sets.a_to_i = {
            activity: outputs[0]
            for activity, outputs in sets.activity_commodities.items()
            if outputs
        }
        sets.i_to_a = {
            commodity: activities[0]
            for commodity, activities in sets.commodity_activities.items()
            if activities
        }


def _repair_factor_subsets_from_labels(sets: GTAPSets, factors: Sequence[str]) -> None:
    """Rebuild factor subsets from trusted labels when the raw GDX subsets are incomplete."""
    valid_factors = list(dict.fromkeys(str(factor) for factor in factors if factor))
    sets.f = valid_factors

    raw_mf = [factor for factor in sets.mf if factor in valid_factors]
    raw_sf = [factor for factor in sets.sf if factor in valid_factors and factor not in raw_mf]
    assigned = set(raw_mf) | set(raw_sf)

    for factor in valid_factors:
        if factor in assigned:
            continue
        label = factor.lower()
        if "land" in label or "natres" in label or "nrs" in label:
            raw_sf.append(factor)
        else:
            raw_mf.append(factor)

    sets.mf = raw_mf
    sets.sf = raw_sf


def _derive_factor_tax_wedges_from_standard_gtap_csv(
    taxes: GTAPTaxRates,
    csv_path: Path,
    *,
    benchmark_year: str,
    solution_year: str,
) -> None:
    """Derive activity-level factor wedges from benchmark `pf` and aggregate `pft` rows."""
    rows = _load_csv_rows(csv_path)
    factor_years = {
        _normalize_year_token(row.get("Year", ""))
        for row in rows
        if _clean_label(row.get("Variable")) in {"pf", "pft"}
    }

    reference_year = "1" if "1" in factor_years else solution_year
    if reference_year == benchmark_year:
        alternatives = sorted(year for year in factor_years if year and year != benchmark_year)
        if alternatives:
            reference_year = alternatives[0]

    pft_levels: Dict[Tuple[str, str], float] = {}
    for row in rows:
        if _normalize_year_token(row.get("Year", "")) != reference_year:
            continue
        if _clean_label(row.get("Variable")) != "pft":
            continue
        region = _clean_label(row.get("Region"))
        factor = _clean_label(row.get("Sector"))
        value = _as_float(row.get("Value"))
        if region and factor:
            pft_levels[(region, factor)] = value

    for row in rows:
        if _normalize_year_token(row.get("Year", "")) != reference_year:
            continue
        if _clean_label(row.get("Variable")) != "pf":
            continue
        region = _clean_label(row.get("Region"))
        activity = _clean_label(row.get("Sector"))
        factor = _clean_label(row.get("Qualifier"))
        gross_value = _as_float(row.get("Value"))
        aggregate_value = pft_levels.get((region, factor))
        if not region or not factor or not activity:
            continue
        if aggregate_value is None or abs(gross_value) <= 1e-10:
            continue
        taxes.kappaf_activity[(region, factor, activity)] = 1.0 - (aggregate_value / gross_value)


def _safe_ratio(value: float, scale: float) -> float:
    """Return value/scale when the scale is usable, otherwise 0."""
    if abs(scale) <= 1e-10:
        return 0.0
    return float(value) / float(scale)


def _regional_income_scale(benchmark: GTAPBenchmarkValues, sets: GTAPSets, region: str) -> float:
    """Benchmark denominator for regional income variables."""
    factor_income = sum(
        benchmark.vfm.get((region, factor, activity), 0.0)
        for factor in sets.f
        for activity in sets.a
    )
    if factor_income > 1e-10:
        return factor_income

    total_absorption = sum(benchmark.vpm.get((region, commodity), 0.0) for commodity in sets.i)
    total_absorption += sum(benchmark.vgm.get((region, commodity), 0.0) for commodity in sets.i)
    total_absorption += sum(benchmark.vim.get((region, commodity), 0.0) for commodity in sets.i)
    return total_absorption


def _agent_benchmark_total(
    benchmark: GTAPBenchmarkValues,
    sets: GTAPSets,
    region: str,
    commodity: str,
    agent: str,
) -> float:
    """Benchmark total Armington demand for an agent/commodity pair."""
    if agent in sets.a:
        return benchmark.vdfm.get((region, commodity, agent), 0.0) + benchmark.vifm.get((region, commodity, agent), 0.0)
    if agent == "hhd":
        return benchmark.vpm.get((region, commodity), 0.0)
    if agent == "gov":
        return benchmark.vgm.get((region, commodity), 0.0)
    if agent == "inv":
        return benchmark.vim.get((region, commodity), 0.0)
    if agent == "tmg":
        return benchmark.vst.get((region, commodity), 0.0)
    return 0.0


def _agent_domestic_benchmark(
    benchmark: GTAPBenchmarkValues,
    sets: GTAPSets,
    region: str,
    commodity: str,
    agent: str,
) -> float:
    """Benchmark domestic demand for an agent/commodity pair."""
    if agent in sets.a:
        return benchmark.vdfb.get((region, commodity, agent), benchmark.vdfm.get((region, commodity, agent), 0.0))
    if agent == "hhd":
        return benchmark.vdpb.get((region, commodity), 0.0)
    if agent == "gov":
        return benchmark.vdgb.get((region, commodity), 0.0)
    if agent == "inv":
        return benchmark.vdib.get((region, commodity), 0.0)
    if agent == "tmg":
        return benchmark.vst.get((region, commodity), 0.0)
    return 0.0


def _commodity_supply_scale(benchmark: GTAPBenchmarkValues, sets: GTAPSets, region: str, commodity: str) -> float:
    """Benchmark quantity scale for commodity-level quantities in the Python template."""
    source_activities = sets.commodity_activities.get(commodity, [])
    if source_activities:
        total = sum(benchmark.vom.get((region, activity), 0.0) for activity in source_activities)
        if total > 1e-10:
            return total
    return benchmark.vom_i.get((region, commodity), 0.0)


def _domestic_sales_scale(benchmark: GTAPBenchmarkValues, sets: GTAPSets, region: str, commodity: str) -> float:
    """Benchmark denominator for `xds`, preferring supply-side scales when absorption is inconsistent."""
    xs0 = getattr(benchmark, "xs0", {})
    xd0 = getattr(benchmark, "xd0", {})
    supply_scale = xs0.get((region, commodity), 0.0)
    if supply_scale <= 1e-10:
        supply_scale = _commodity_supply_scale(benchmark, sets, region, commodity)

    absorption_scale = xd0.get((region, commodity), 0.0)
    if absorption_scale <= 1e-10:
        _, absorption_scale, _, _, _ = benchmark.get_trade_totals(sets, region, commodity)
    if absorption_scale <= 1e-10:
        return supply_scale
    if supply_scale > 1e-10 and absorption_scale < 0.25 * supply_scale:
        return supply_scale
    return absorption_scale


def _load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """Read a standard_gtap_7 postsim CSV file."""
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolve_standard_gtap_reference_csv(csv_path: Path, sets: Optional[GTAPSets] = None) -> Path:
    """Resolve the preferred standard_gtap reference CSV for the active set domain."""
    candidate = Path(csv_path)
    if candidate.suffix.lower() != ".csv" or candidate.name != "COMP.csv":
        return candidate
    if not _is_standard_gtap_9x10_sets(sets):
        return candidate

    for replacement in (
        candidate.parent / "comp" / "COMP_generated.csv",
        candidate.parent / "comp_neos" / "COMP.csv",
    ):
        if replacement.exists():
            if replacement != candidate:
                logger.info("Using 9x10-compatible GTAP reference CSV %s instead of %s", replacement, candidate)
            return replacement
    return candidate


def _standard_gtap_generated_scale_map(csv_path: Path) -> Dict[Tuple[str, str, str, str, str], float]:
    """Infer per-row scaling for generated comparator CSVs from the NEOS export when available."""
    candidate = Path(csv_path)
    cached = _STANDARD_GTAP_GENERATED_SCALE_CACHE.get(candidate)
    if cached is not None:
        return cached

    scale_map: Dict[Tuple[str, str, str, str, str], float] = {}
    if candidate.name not in {"COMP_generated.csv", "COMP_equilibria.csv"} or candidate.parent.name != "comp":
        _STANDARD_GTAP_GENERATED_SCALE_CACHE[candidate] = scale_map
        return scale_map

    sibling = candidate.parent.parent / "comp_neos" / "COMP.csv"
    if not sibling.exists():
        _STANDARD_GTAP_GENERATED_SCALE_CACHE[candidate] = scale_map
        return scale_map

    generated_values = {
        (
            _clean_label(row.get("Variable")),
            _clean_label(row.get("Region")),
            _clean_label(row.get("Sector")),
            _clean_label(row.get("Qualifier")),
            _normalize_year_token(row.get("Year", "")),
        ): _as_float(row.get("Value"))
        for row in _load_csv_rows(candidate)
    }

    for row in _load_csv_rows(sibling):
        key = (
            _clean_label(row.get("Variable")),
            _clean_label(row.get("Region")),
            _clean_label(row.get("Sector")),
            _clean_label(row.get("Qualifier")),
            _normalize_year_token(row.get("Year", "")),
        )
        generated_value = generated_values.get(key)
        neos_value = _as_float(row.get("Value"))
        if generated_value is None or abs(generated_value) <= 1e-12 or abs(neos_value) <= 1e-12:
            continue

        ratio = neos_value / generated_value
        if ratio <= 0.0:
            continue

        exponent = round(math.log10(ratio))
        snapped = 10.0 ** exponent
        if abs(ratio - snapped) / snapped <= 1e-6:
            scale_map[key] = snapped

    _STANDARD_GTAP_GENERATED_SCALE_CACHE[candidate] = scale_map
    return scale_map


def _standard_gtap_csv_scale_factor(
    csv_path: Path,
    variable: str,
    *,
    region: str = "",
    sector: str = "",
    qualifier: str = "",
    year: str = "",
) -> float:
    """Return the scale factor needed for standard_gtap comparator CSV rows.

    Some generated comparator CSVs serialize quantity and income variables in the
    model's internal `inScale=1e-6` units. Those rows must be rescaled back to
    benchmark magnitudes before parity comparison. Prices and tax rates remain in
    level form and should not be adjusted.
    """
    candidate = Path(csv_path)
    if candidate.name not in {"COMP_generated.csv", "COMP_equilibria.csv"}:
        return 1.0
    if candidate.parent.name != "comp":
        return 1.0

    inferred_scale = _standard_gtap_generated_scale_map(candidate).get(
        (variable, region, sector, qualifier, _normalize_year_token(year))
    )
    if inferred_scale is not None:
        return inferred_scale

    if variable in {"xa", "xd"} and qualifier.startswith("a_"):
        return 1.0e4
    if variable == "xf" and sector.startswith("a_") and qualifier:
        return 1.0e7

    quantity_or_income_vars = {
        "xp", "x", "xs", "xds", "xd", "xw", "xwmg", "xmgm", "xtmg",
        "xaa", "xa", "xe", "xmt", "xet", "xf", "xft", "xc", "xg", "xi",
        "regY", "yc", "yg", "yi",
    }
    return 1.0e6 if variable in quantity_or_income_vars else 1.0


def _resolve_standard_gtap_years(
    csv_path: Path,
    *,
    benchmark_year: str,
    solution_year: str,
) -> tuple[str, str]:
    """Resolve benchmark/solution year tokens against the actual CSV contents."""
    rows = _load_csv_rows(csv_path)
    available_years = [
        _normalize_year_token(row.get("Year", ""))
        for row in rows
        if _normalize_year_token(row.get("Year", ""))
    ]
    available = list(dict.fromkeys(available_years))

    resolved_solution = solution_year
    if resolved_solution not in available:
        if "shock" in available:
            resolved_solution = "shock"
        elif "1" in available:
            resolved_solution = "1"
        elif available:
            resolved_solution = available[0]

    resolved_benchmark = benchmark_year
    if resolved_benchmark not in available:
        if "1" in available:
            resolved_benchmark = "1"
        elif available:
            alternatives = [year for year in available if year != resolved_solution]
            resolved_benchmark = alternatives[0] if alternatives else available[0]

    if resolved_benchmark != benchmark_year or resolved_solution != solution_year:
        logger.info(
            "Resolved standard_gtap CSV years benchmark=%s->%s solution=%s->%s",
            benchmark_year,
            resolved_benchmark,
            solution_year,
            resolved_solution,
        )
    return resolved_benchmark, resolved_solution


def _enrich_sets_from_standard_gtap_csv(
    sets: GTAPSets,
    csv_path: Path,
    *,
    solution_year: str,
    benchmark_year: str,
) -> None:
    """Derive make/output pairs from `x(r,a,i)` rows in a postsim CSV."""
    rows = _load_csv_rows(csv_path)
    regions: List[str] = []
    activities: List[str] = []
    commodities: List[str] = []
    factors: List[str] = []
    margins: List[str] = []
    pairs: List[Tuple[str, str]] = []
    for row in rows:
        variable = _clean_label(row.get("Variable"))
        year = _normalize_year_token(row.get("Year", ""))

        if year == solution_year:
            region = _clean_label(row.get("Region"))
            if region and region != "GBL" and region not in regions:
                regions.append(region)
            if variable == "xp":
                activity = _clean_label(row.get("Sector"))
                if activity and activity not in activities:
                    activities.append(activity)
            elif variable == "xs":
                commodity = _clean_label(row.get("Sector"))
                if commodity and commodity not in commodities:
                    commodities.append(commodity)
            elif variable == "xf":
                factor = _clean_label(row.get("Qualifier"))
                if factor and factor not in factors:
                    factors.append(factor)
            elif variable == "x":
                activity = _clean_label(row.get("Sector"))
                commodity = _clean_label(row.get("Qualifier"))
                if activity and commodity:
                    pairs.append((activity, commodity))

        if year == benchmark_year:
            region = _clean_label(row.get("Region"))
            if region and region != "GBL" and region not in regions:
                regions.append(region)

        if year == benchmark_year and variable == "vst":
            commodity = _clean_label(row.get("Sector"))
            value = _as_float(row.get("Value"))
            if commodity and value > 1e-10 and commodity not in margins:
                margins.append(commodity)

    if regions:
        sets.r = regions
        sets.s = regions.copy()
    if activities:
        sets.a = activities
    if commodities:
        sets.i = commodities
    if factors:
        _repair_factor_subsets_from_labels(sets, factors)
    if margins:
        sets.m = margins

    if pairs:
        _apply_output_pairs(sets, pairs)


def _load_standard_gtap_benchmark_from_csv(
    csv_path: Path,
    sets: GTAPSets,
    *,
    benchmark_year: str,
    solution_year: str,
) -> GTAPBenchmarkValues:
    """Load benchmark values from `COMP.csv` produced by `postsim.gms`."""
    rows = _load_csv_rows(csv_path)
    benchmark = GTAPBenchmarkValues()

    p_solution: Dict[Tuple[str, str, str], float] = {}
    x_solution: Dict[Tuple[str, str, str], float] = {}

    for row in rows:
        variable = _clean_label(row.get("Variable"))
        year = _normalize_year_token(row.get("Year", ""))
        region = _clean_label(row.get("Region"))
        sector = _clean_label(row.get("Sector"))
        qualifier = _clean_label(row.get("Qualifier"))
        value = _as_float(row.get("Value"))

        if year == solution_year:
            if variable == "p" and region and sector and qualifier:
                p_solution[(region, sector, qualifier)] = value
            elif variable == "x" and region and sector and qualifier:
                x_solution[(region, sector, qualifier)] = value
            continue

        if year != benchmark_year:
            continue

        if variable == "evfb" and region and sector and qualifier:
            benchmark.vfm[(region, qualifier, sector)] = value
            benchmark.vfb[(region, sector)] = benchmark.vfb.get((region, sector), 0.0) + value
        elif variable == "vdfp" and region and sector and qualifier:
            benchmark.vdfp[(region, sector, qualifier)] = value
        elif variable == "vdfb" and region and sector and qualifier:
            benchmark.vdfb[(region, sector, qualifier)] = value
        elif variable == "vmfp" and region and sector and qualifier:
            benchmark.vmfp[(region, sector, qualifier)] = value
        elif variable == "vmfb" and region and sector and qualifier:
            benchmark.vmfb[(region, sector, qualifier)] = value
        elif variable == "vdpp" and region and sector:
            benchmark.vdpp[(region, sector)] = value
        elif variable == "vdpb" and region and sector:
            benchmark.vdpb[(region, sector)] = value
        elif variable == "vmpp" and region and sector:
            benchmark.vmpp[(region, sector)] = value
        elif variable == "vmpb" and region and sector:
            benchmark.vmpb[(region, sector)] = value
        elif variable == "vdgp" and region and sector:
            benchmark.vdgp[(region, sector)] = value
        elif variable == "vdgb" and region and sector:
            benchmark.vdgb[(region, sector)] = value
        elif variable == "vmgp" and region and sector:
            benchmark.vmgp[(region, sector)] = value
        elif variable == "vmgb" and region and sector:
            benchmark.vmgb[(region, sector)] = value
        elif variable == "vdip" and region and sector:
            benchmark.vdip[(region, sector)] = value
        elif variable == "vdib" and region and sector:
            benchmark.vdib[(region, sector)] = value
        elif variable == "vmip" and region and sector:
            benchmark.vmip[(region, sector)] = value
        elif variable == "vmib" and region and sector:
            benchmark.vmib[(region, sector)] = value
        elif variable == "vxsb" and region and sector and qualifier:
            benchmark.vxsb[(region, sector, qualifier)] = value
        elif variable == "vfob" and region and sector and qualifier:
            benchmark.vfob[(region, sector, qualifier)] = value
        elif variable == "vcif" and region and sector and qualifier:
            benchmark.vcif[(region, sector, qualifier)] = value
        elif variable == "vmsb" and region and sector and qualifier:
            benchmark.vmsb[(region, sector, qualifier)] = value
        elif variable == "vst" and region and sector and sector in sets.m:
            benchmark.vst[(region, sector)] = value

    for key, quantity in x_solution.items():
        region, activity, commodity = key
        if abs(quantity) <= 1e-10:
            continue
        price = p_solution.get(key, 1.0)
        make_value = quantity * price
        benchmark.makb[(region, activity, commodity)] = make_value
        benchmark.vom[(region, activity)] = benchmark.vom.get((region, activity), 0.0) + make_value

    benchmark._derive_output_totals(sets)
    benchmark._derive_intermediate_totals(sets)
    benchmark._derive_final_demand_totals(sets)
    benchmark._derive_trade_aggregates(sets)
    return benchmark


@dataclass(frozen=True)
class GTAPDataBundle:
    """Split data bundle for `standard_gtap_7` style parity runs."""

    sets_gdx: Path
    elasticities_gdx: Optional[Path] = None
    benchmark_csv: Optional[Path] = None
    benchmark_gdx: Optional[Path] = None


@dataclass(frozen=True)
class GTAPVariableSnapshot:
    """Snapshot of GTAP variable values."""

    xp: Dict[Tuple[str, str], float] = field(default_factory=dict)
    x: Dict[Tuple[str, str, str], float] = field(default_factory=dict)

    xs: Dict[Tuple[str, str], float] = field(default_factory=dict)
    xds: Dict[Tuple[str, str], float] = field(default_factory=dict)
    xd: Dict[Tuple[str, str, str], float] = field(default_factory=dict)

    px: Dict[Tuple[str, str], float] = field(default_factory=dict)
    pp: Dict[Tuple[str, str], float] = field(default_factory=dict)
    ps: Dict[Tuple[str, str], float] = field(default_factory=dict)
    pd: Dict[Tuple[str, str], float] = field(default_factory=dict)
    pa: Dict[Tuple[str, str], float] = field(default_factory=dict)
    paa: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    pdp: Dict[Tuple[str, str, str], float] = field(default_factory=dict)

    pmt: Dict[Tuple[str, str], float] = field(default_factory=dict)
    pmcif: Dict[Tuple[str, str, str], float] = field(default_factory=dict)

    pet: Dict[Tuple[str, str], float] = field(default_factory=dict)
    pe: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    pefob: Dict[Tuple[str, str, str], float] = field(default_factory=dict)

    xe: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    xw: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    xmt: Dict[Tuple[str, str], float] = field(default_factory=dict)
    xet: Dict[Tuple[str, str], float] = field(default_factory=dict)
    xaa: Dict[Tuple[str, str, str], float] = field(default_factory=dict)

    xwmg: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    xmgm: Dict[Tuple[str, str, str, str], float] = field(default_factory=dict)
    pwmg: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    xtmg: Dict[Tuple[str], float] = field(default_factory=dict)
    ptmg: Dict[Tuple[str], float] = field(default_factory=dict)

    xf: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    xft: Dict[Tuple[str, str], float] = field(default_factory=dict)
    pf: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    pft: Dict[Tuple[str, str], float] = field(default_factory=dict)

    xc: Dict[Tuple[str, str], float] = field(default_factory=dict)
    xg: Dict[Tuple[str, str], float] = field(default_factory=dict)
    xi: Dict[Tuple[str, str], float] = field(default_factory=dict)

    regy: Dict[str, float] = field(default_factory=dict)
    yc: Dict[str, float] = field(default_factory=dict)
    yg: Dict[str, float] = field(default_factory=dict)
    yi: Dict[str, float] = field(default_factory=dict)
    ug: Dict[str, float] = field(default_factory=dict)
    arent: Dict[str, float] = field(default_factory=dict)
    facty: Dict[str, float] = field(default_factory=dict)

    pnum: Optional[float] = None
    pabs: Dict[str, float] = field(default_factory=dict)
    walras: Optional[float] = None

    def is_empty(self) -> bool:
        """Return whether the snapshot contains any non-scalar data."""
        for field_name, value in self.__dict__.items():
            if field_name in {"pnum", "walras"}:
                if value is not None:
                    return False
                continue
            if value:
                return False
        return True

    @classmethod
    def from_python_model(cls, model) -> "GTAPVariableSnapshot":
        """Extract a snapshot from a solved Python/Pyomo model."""
        from pyomo.environ import value

        return cls(
            xp=_extract_var_levels(model.xp) if hasattr(model, "xp") else {},
            x=_extract_var_levels(model.x) if hasattr(model, "x") else {},
            xs=_extract_var_levels(model.xs) if hasattr(model, "xs") else {},
            xds=(
                _extract_var_levels(model.xds)
                if hasattr(model, "xds")
                else _extract_var_levels(model.xd) if hasattr(model, "xd") else {}
            ),
            xd=_extract_xda_unscaled(model),
            px=_extract_var_levels(model.px) if hasattr(model, "px") else {},
            pp=_extract_var_levels(model.pp) if hasattr(model, "pp") else {},
            ps=_extract_var_levels(model.ps) if hasattr(model, "ps") else {},
            pd=_extract_var_levels(model.pd) if hasattr(model, "pd") else {},
            pa=_extract_var_levels(model.pa) if hasattr(model, "pa") else {},
            paa=_extract_var_levels(model.paa) if hasattr(model, "paa") else {},
            pdp=_extract_var_levels(model.pdp) if hasattr(model, "pdp") else {},
            pmt=_extract_var_levels(model.pmt) if hasattr(model, "pmt") else {},
            pmcif=_extract_var_levels(model.pmcif) if hasattr(model, "pmcif") else {},
            pet=_extract_var_levels(model.pet) if hasattr(model, "pet") else {},
            pe=_extract_var_levels(model.pe) if hasattr(model, "pe") else {},
            pefob=_extract_var_levels(model.pefob) if hasattr(model, "pefob") else {},
            xe=_extract_var_levels(model.xe) if hasattr(model, "xe") else {},
            xw=_extract_var_levels(model.xw) if hasattr(model, "xw") else {},
            xmt=_extract_var_levels(model.xmt) if hasattr(model, "xmt") else {},
            xet=_extract_var_levels(model.xet) if hasattr(model, "xet") else {},
            xaa=_extract_xaa_unscaled(model),
            xwmg=_extract_var_levels(model.xwmg) if hasattr(model, "xwmg") else {},
            xmgm=_extract_var_levels(model.xmgm) if hasattr(model, "xmgm") else {},
            pwmg=_extract_var_levels(model.pwmg) if hasattr(model, "pwmg") else {},
            xtmg=_extract_var_levels(model.xtmg) if hasattr(model, "xtmg") else {},
            ptmg=_normalize_singleton_keys(_extract_var_levels(model.ptmg)) if hasattr(model, "ptmg") else {},
            xf=_extract_var_levels(model.xf) if hasattr(model, "xf") else {},
            xft=_extract_var_levels(model.xft) if hasattr(model, "xft") else {},
            pf=_extract_gtap_factor_prices(model),
            pft=_extract_var_levels(model.pft) if hasattr(model, "pft") else {},
            xc=_extract_var_levels(model.xc) if hasattr(model, "xc") else {},
            xg=_extract_var_levels(model.xg) if hasattr(model, "xg") else {},
            xi=_extract_var_levels(model.xi) if hasattr(model, "xi") else {},
            regy=_extract_var_levels(model.regy) if hasattr(model, "regy") else {},
            yc=_extract_var_levels(model.yc) if hasattr(model, "yc") else {},
            yg=_extract_var_levels(model.yg) if hasattr(model, "yg") else {},
            yi=_extract_var_levels(model.yi) if hasattr(model, "yi") else {},
            pnum=float(value(model.pnum)) if hasattr(model, "pnum") else None,
            pabs=_extract_var_levels(model.pabs) if hasattr(model, "pabs") else {},
            walras=float(value(model.walras)) if hasattr(model, "walras") else None,
        )

    @classmethod
    def from_gdx(cls, gdx_path: Path, sets: GTAPSets) -> "GTAPVariableSnapshot":
        """Extract a snapshot from a GAMS result GDX."""
        gdx_data = read_gdx(gdx_path)

        def read_level(name: str) -> Dict[Any, float]:
            try:
                values = read_variable_values(gdx_data, name)
                return {key: float(attrs.get("level", 0.0)) for key, attrs in values.items()}
            except Exception:
                try:
                    values = read_parameter_values(gdx_data, name)
                    return {key: float(value) for key, value in values.items()}
                except Exception:
                    return {}

        pnum_values = read_level("pnum")
        walras_values = read_level("walras")

        return cls(
            xp=read_level("xp"),
            x=read_level("x"),
            xs=read_level("xs"),
            xds=read_level("xds") or read_level("xd"),
            xd=read_level("xda"),
            px=read_level("px"),
            pp=read_level("pp"),
            ps=read_level("ps"),
            pd=read_level("pd"),
            pa=read_level("pa"),
            paa=read_level("paa"),
            pdp=read_level("pdp"),
            pmt=read_level("pmt"),
            pmcif=read_level("pmcif"),
            pet=read_level("pet"),
            pe=read_level("pe"),
            pefob=read_level("pefob"),
            xe=read_level("xe"),
            xw=read_level("xw"),
            xmt=read_level("xmt"),
            xet=read_level("xet"),
            xaa=read_level("xaa"),
            xwmg=read_level("xwmg"),
            xmgm=read_level("xmgm"),
            pwmg=read_level("pwmg"),
            xtmg=read_level("xtmg"),
            ptmg=_normalize_singleton_keys(read_level("ptmg")),
            xf=read_level("xf"),
            xft=read_level("xft"),
            pf=read_level("pf"),
            pft=read_level("pft"),
            xc=read_level("xc"),
            xg=read_level("xg"),
            xi=read_level("xi"),
            regy=read_level("regy"),
            yc=read_level("yc"),
            yg=read_level("yg"),
            yi=read_level("yi"),
            pnum=float(next(iter(pnum_values.values()))) if pnum_values else None,
            pabs=read_level("pabs"),
            walras=float(next(iter(walras_values.values()))) if walras_values else None,
        )

    @classmethod
    def from_standard_gtap_csv(
        cls,
        csv_path: Path,
        sets: GTAPSets,
        *,
        solution_year: int | str = 1,
    ) -> "GTAPVariableSnapshot":
        """Extract solution values from a `postsim.gms` CSV export."""
        year = _normalize_year_token(solution_year)
        rows = _load_csv_rows(csv_path)

        snapshot = cls()
        data = snapshot.__dict__.copy()
        ytax_by_region: Dict[str, float] = defaultdict(float)
        if len(sets.m) == 1:
            margin_commodity = sets.m[0]
        else:
            margin_commodity = None

        for row in rows:
            if not _row_matches_year(row, year):
                continue

            variable = _clean_label(row.get("Variable"))
            region = _clean_label(row.get("Region"))
            sector = _clean_label(row.get("Sector"))
            qualifier = _clean_label(row.get("Qualifier"))
            value = _as_float(row.get("Value")) * _standard_gtap_csv_scale_factor(
                csv_path,
                variable,
                region=region,
                sector=sector,
                qualifier=qualifier,
                year=year,
            )

            if variable == "xp" and region and sector:
                data["xp"][(region, sector)] = value
            elif variable == "x" and region and sector and qualifier:
                data["x"][(region, sector, qualifier)] = value
            elif variable == "xs" and region and sector:
                data["xs"][(region, sector)] = value
            elif variable == "xds" and region and sector:
                data["xds"][(region, sector)] = value
            elif variable == "xd" and region and sector and qualifier:
                data["xd"][(region, sector, qualifier)] = value
            elif variable == "px" and region and sector:
                data["px"][(region, sector)] = value
            elif variable == "ps" and region and sector:
                data["ps"][(region, sector)] = value
            elif variable == "pd" and region and sector:
                data["pd"][(region, sector)] = value
            elif variable == "pet" and region and sector:
                data["pet"][(region, sector)] = value
            elif variable == "pe" and region and sector and qualifier:
                data["pe"][(region, sector, qualifier)] = value
            elif variable == "xw" and region and sector and qualifier:
                data["xw"][(region, sector, qualifier)] = value
            elif variable == "xwmg" and region and sector and qualifier:
                data["xwmg"][(region, sector, qualifier)] = value
            elif variable == "pwmg" and region and sector and qualifier:
                data["pwmg"][(region, sector, qualifier)] = value
            elif variable == "xmgm" and region and sector and qualifier and margin_commodity:
                data["xmgm"][(margin_commodity, region, sector, qualifier)] = value
            elif variable == "xtmg" and sector:
                data["xtmg"][(sector,)] = value
            elif variable == "ptmg" and sector:
                data["ptmg"][(sector,)] = value
            elif variable == "pdp" and region and sector and qualifier:
                data["pdp"][(region, sector, qualifier)] = value
            elif variable == "xa" and region and sector and qualifier:
                data["xaa"][(region, sector, qualifier)] = value
            elif variable == "pa" and region and sector and qualifier:
                data["paa"][(region, sector, qualifier)] = value
            elif variable == "xf" and region and sector and qualifier:
                data["xf"][(region, qualifier, sector)] = value
            elif variable == "pf" and region and sector and qualifier:
                data["pf"][(region, qualifier, sector)] = value
            elif variable == "xft" and region and sector:
                data["xft"][(region, sector)] = value
            elif variable == "pft" and region and sector:
                data["pft"][(region, sector)] = value
            elif variable == "xc" and region and sector:
                data["xc"][(region, sector)] = value
            elif variable == "xg" and region and sector:
                data["xg"][(region, sector)] = value
            elif variable == "xi" and region and sector:
                data["xi"][(region, sector)] = value
            elif variable == "regY" and region:
                data["regy"][region] = value
            elif variable == "yc" and region:
                data["yc"][region] = value
            elif variable == "yg" and region:
                data["yg"][region] = value
            elif variable == "yi" and region:
                data["yi"][region] = value
            elif variable == "ug" and region:
                data["ug"][region] = value
            elif variable == "arent" and region:
                data["arent"][region] = value
            elif variable == "pabs" and region:
                data["pabs"][region] = value
            elif variable == "ytax" and region:
                ytax_by_region[region] += value

        for region in set(data["regy"]) | set(ytax_by_region):
            data["facty"][region] = float(data["regy"].get(region, 0.0)) - float(ytax_by_region.get(region, 0.0))

        return cls(**data)


@dataclass(frozen=True)
class GTAPParityComparison:
    """Parity comparison result between Python and GAMS."""

    passed: bool
    tolerance: float
    n_variables_compared: int
    n_mismatches: int
    max_abs_diff: float
    max_rel_diff: float
    mismatches: List[Dict[str, Any]]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class GTAPGAMSReference:
    """Reference data from a GAMS run or postsim CSV export."""

    gdx_path: Path
    sets: GTAPSets
    snapshot: GTAPVariableSnapshot
    modelstat: float
    solvestat: float
    solve_time: float

    @classmethod
    def load(
        cls,
        gdx_path: Path,
        sets: Optional[GTAPSets] = None,
        *,
        solution_year: int | str = 1,
    ) -> "GTAPGAMSReference":
        """Load a reference snapshot from CSV or GDX."""
        gdx_path = Path(gdx_path)

        if gdx_path.suffix.lower() == ".csv":
            if sets is None:
                raise ValueError("CSV references require GTAP sets")
            gdx_path = _resolve_standard_gtap_reference_csv(gdx_path, sets)
            snapshot = GTAPVariableSnapshot.from_standard_gtap_csv(
                gdx_path,
                sets,
                solution_year=solution_year,
            )
            return cls(
                gdx_path=gdx_path,
                sets=sets,
                snapshot=snapshot,
                modelstat=1.0,
                solvestat=1.0,
                solve_time=0.0,
            )

        if sets is None:
            data_gdx = gdx_path.parent / f"{gdx_path.stem.split('_')[0]}.gdx"
            if data_gdx.exists():
                sets = GTAPSets()
                sets.load_from_gdx(data_gdx)
            else:
                raise FileNotFoundError(f"Could not infer GTAP sets for {gdx_path}")

        snapshot = GTAPVariableSnapshot.from_gdx(gdx_path, sets)
        if snapshot.is_empty():
            csv_fallback = gdx_path.with_suffix(".csv")
            if csv_fallback.exists():
                logger.info("Falling back to CSV reference for %s", gdx_path)
                return cls.load(csv_fallback, sets, solution_year=solution_year)

        gdx_data = read_gdx(gdx_path)
        modelstat = 1.0
        solvestat = 1.0
        try:
            modelstat_data = read_parameter_values(gdx_data, "modelstat")
            if modelstat_data:
                modelstat = float(next(iter(modelstat_data.values())))
        except Exception:
            pass
        try:
            solvestat_data = read_parameter_values(gdx_data, "solvestat")
            if solvestat_data:
                solvestat = float(next(iter(solvestat_data.values())))
        except Exception:
            pass

        return cls(
            gdx_path=gdx_path,
            sets=sets,
            snapshot=snapshot,
            modelstat=modelstat,
            solvestat=solvestat,
            solve_time=0.0,
        )


def normalize_gams_snapshot_against_benchmark(
    snapshot: GTAPVariableSnapshot,
    benchmark: GTAPBenchmarkValues,
    sets: GTAPSets,
) -> GTAPVariableSnapshot:
    """Convert GAMS benchmark-level quantities and incomes into benchmark ratios."""
    normalized = GTAPVariableSnapshot(
        px=dict(snapshot.px),
        pp=dict(snapshot.pp),
        ps=dict(snapshot.ps),
        pd=dict(snapshot.pd),
        pa=dict(snapshot.pa),
        paa=dict(snapshot.paa),
        pdp=dict(snapshot.pdp),
        pmt=dict(snapshot.pmt),
        pmcif=dict(snapshot.pmcif),
        pet=dict(snapshot.pet),
        pe=dict(snapshot.pe),
        pefob=dict(snapshot.pefob),
        pwmg=dict(snapshot.pwmg),
        pft=dict(snapshot.pft),
        pf=dict(snapshot.pf),
        pabs=dict(snapshot.pabs),
        ptmg=dict(snapshot.ptmg),
        pnum=snapshot.pnum,
        walras=snapshot.walras,
    )

    for key, value in snapshot.xp.items():
        normalized.xp[key] = _safe_ratio(value, benchmark.vom.get(key, 0.0))

    for (region, activity, commodity), value in snapshot.x.items():
        scale = benchmark.makb.get((region, activity, commodity), 0.0)
        normalized.x[(region, activity, commodity)] = _safe_ratio(value, scale)

    for (region, commodity), value in snapshot.xs.items():
        scale = _commodity_supply_scale(benchmark, sets, region, commodity)
        normalized.xs[(region, commodity)] = _safe_ratio(value, scale)

    for (region, commodity), value in snapshot.xds.items():
        scale = _domestic_sales_scale(benchmark, sets, region, commodity)
        normalized.xds[(region, commodity)] = _safe_ratio(value, scale)

    for (region, commodity, agent), value in snapshot.xd.items():
        scale = _agent_domestic_benchmark(benchmark, sets, region, commodity, agent)
        normalized.xd[(region, commodity, agent)] = _safe_ratio(value, scale)

    for (region, commodity, agent), value in snapshot.xaa.items():
        scale = _agent_benchmark_total(benchmark, sets, region, commodity, agent)
        normalized.xaa[(region, commodity, agent)] = _safe_ratio(value, scale)

    for (region, commodity), value in snapshot.xmt.items():
        xmt0 = getattr(benchmark, "xmt0", {})
        scale = xmt0.get((region, commodity), 0.0)
        if scale <= 1e-10:
            _, _, _, scale, _ = benchmark.get_trade_totals(sets, region, commodity)
        normalized.xmt[(region, commodity)] = _safe_ratio(value, scale)

    for (region, commodity), value in snapshot.xet.items():
        xet0 = getattr(benchmark, "xet0", {})
        scale = xet0.get((region, commodity), 0.0)
        if scale <= 1e-10:
            _, _, scale, _, _ = benchmark.get_trade_totals(sets, region, commodity)
        normalized.xet[(region, commodity)] = _safe_ratio(value, scale)

    for (region, commodity, partner), value in snapshot.xw.items():
        if region == partner:
            scale = benchmark.vxmd.get((region, commodity, partner), 0.0)
            if scale <= 1e-10:
                scale = benchmark.viws.get((partner, commodity, region), 0.0)
        elif hasattr(benchmark, "get_import_flow"):
            scale = benchmark.get_import_flow(region, commodity, partner)
        else:
            scale = benchmark.viws.get((partner, commodity, region), 0.0)
            if scale <= 1e-10:
                scale = benchmark.vcif.get((partner, commodity, region), 0.0)
        normalized.xw[(region, commodity, partner)] = _safe_ratio(value, scale)

    for (region, commodity, partner), value in snapshot.xe.items():
        if hasattr(benchmark, "get_export_flow"):
            scale = benchmark.get_export_flow(region, commodity, partner)
        else:
            scale = benchmark.vxsb.get((region, commodity, partner), 0.0)
            if scale <= 1e-10:
                scale = benchmark.vxmd.get((region, commodity, partner), 0.0)
        normalized.xe[(region, commodity, partner)] = _safe_ratio(value, scale)

    for (region, commodity, partner), value in snapshot.xwmg.items():
        vtwr_route = getattr(benchmark, "vtwr_route", {})
        scale = vtwr_route.get((region, commodity, partner), 0.0)
        if scale <= 1e-10:
            scale = sum(
                benchmark.vtwr.get((margin, commodity, region, partner), 0.0)
                for margin in sets.m
            )
        normalized.xwmg[(region, commodity, partner)] = _safe_ratio(value, scale)

    for (margin, region, commodity, partner), value in snapshot.xmgm.items():
        vtwr_margin = getattr(benchmark, "vtwr_margin", {})
        scale = vtwr_margin.get((margin, region, commodity, partner), 0.0)
        if scale <= 1e-10:
            scale = benchmark.vtwr.get((margin, commodity, region, partner), 0.0)
        normalized.xmgm[(margin, region, commodity, partner)] = _safe_ratio(value, scale)

    for (margin,), value in snapshot.xtmg.items():
        vtwr_margin = getattr(benchmark, "vtwr_margin", {})
        scale = sum(
            vtwr_margin.get((margin, destination, commodity, source), 0.0)
            for destination in sets.r
            for commodity in sets.i
            for source in sets.r
            if source != destination
        )
        if scale <= 1e-10:
            scale = sum(
                benchmark.vtwr.get((margin, commodity, destination, source), 0.0)
                for destination in sets.r
                for commodity in sets.i
                for source in sets.r
                if source != destination
            )
        normalized.xtmg[(margin,)] = _safe_ratio(value, scale)

    for (region, factor, activity), value in snapshot.xf.items():
        scale = benchmark.vfm.get((region, factor, activity), 0.0)
        normalized.xf[(region, factor, activity)] = _safe_ratio(value, scale)

    for (region, factor), value in snapshot.xft.items():
        scale = benchmark.vfb.get((region, factor), 0.0)
        if scale <= 1e-10:
            scale = sum(benchmark.vfm.get((region, factor, activity), 0.0) for activity in sets.a)
        normalized.xft[(region, factor)] = _safe_ratio(value, scale)

    for (region, commodity), value in snapshot.xc.items():
        normalized.xc[(region, commodity)] = _safe_ratio(value, benchmark.vpm.get((region, commodity), 0.0))

    for (region, commodity), value in snapshot.xg.items():
        normalized.xg[(region, commodity)] = _safe_ratio(value, benchmark.vgm.get((region, commodity), 0.0))

    for (region, commodity), value in snapshot.xi.items():
        normalized.xi[(region, commodity)] = _safe_ratio(value, benchmark.vim.get((region, commodity), 0.0))

    for region, value in snapshot.regy.items():
        normalized.regy[region] = _safe_ratio(value, _regional_income_scale(benchmark, sets, region))

    for region, value in snapshot.yc.items():
        scale = sum(benchmark.vpm.get((region, commodity), 0.0) for commodity in sets.i)
        normalized.yc[region] = _safe_ratio(value, scale)

    for region, value in snapshot.yg.items():
        scale = sum(benchmark.vgm.get((region, commodity), 0.0) for commodity in sets.i)
        normalized.yg[region] = _safe_ratio(value, scale)

    for region, value in snapshot.yi.items():
        scale = sum(benchmark.vim.get((region, commodity), 0.0) for commodity in sets.i)
        normalized.yi[region] = _safe_ratio(value, scale)

    return normalized


def compare_variable_groups(
    python: Dict,
    gams: Dict,
    group_name: str,
    tolerance: float = 1e-6,
) -> Tuple[int, int, float, List[Dict[str, Any]]]:
    """Compare one variable group."""
    n_compared = 0
    n_mismatches = 0
    max_diff = 0.0
    mismatches: List[Dict[str, Any]] = []

    all_keys = set(python.keys()) | set(gams.keys())
    for key in all_keys:
        py_val = python.get(key, 0.0)
        gams_val = gams.get(key, 0.0)
        if py_val == 0.0 and gams_val == 0.0:
            continue

        n_compared += 1
        abs_diff = abs(py_val - gams_val)
        rel_diff = abs_diff / max(abs(gams_val), 1e-10)
        max_diff = max(max_diff, abs_diff)

        if abs_diff > tolerance:
            n_mismatches += 1
            mismatches.append(
                {
                    "group": group_name,
                    "key": key,
                    "python": py_val,
                    "gams": gams_val,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                }
            )

    return n_compared, n_mismatches, max_diff, mismatches


def _detect_degenerate_groups(
    py_snapshot: GTAPVariableSnapshot,
    gams_snapshot: GTAPVariableSnapshot,
) -> Dict[str, Any]:
    """Detect blocks that collapsed near zero in the Python solution."""

    def collapse_ratio(py_values: Dict[Any, float], gams_values: Dict[Any, float]) -> tuple[int, int, float]:
        relevant = [key for key, gams_value in gams_values.items() if abs(gams_value) > 1e-8]
        if not relevant:
            return 0, 0, 0.0
        collapsed = sum(1 for key in relevant if abs(py_values.get(key, 0.0)) <= 1e-6)
        return collapsed, len(relevant), collapsed / len(relevant)

    quantity_groups = [
        "xp", "x", "xs", "xds", "xd", "xe", "xw", "xmt", "xet", "xaa",
        "xwmg", "xmgm", "xtmg", "xf", "xft", "xc", "xg", "xi", "regy", "yc", "yg", "yi",
    ]
    price_groups = [
        "px", "pp", "ps", "pd", "pa", "paa", "pdp", "pmt", "pmcif",
        "pet", "pe", "pefob", "pwmg", "ptmg", "pf", "pft", "pabs",
    ]

    collapsed_quantities: Dict[str, float] = {}
    collapsed_prices: Dict[str, float] = {}
    total_quantity_collapsed = 0
    total_quantity_relevant = 0
    total_price_collapsed = 0
    total_price_relevant = 0

    for group_name in quantity_groups:
        py_values = getattr(py_snapshot, group_name)
        gams_values = getattr(gams_snapshot, group_name)
        collapsed, relevant, ratio = collapse_ratio(py_values, gams_values)
        total_quantity_collapsed += collapsed
        total_quantity_relevant += relevant
        if relevant and ratio >= 0.8:
            collapsed_quantities[group_name] = ratio

    for group_name in price_groups:
        py_values = getattr(py_snapshot, group_name)
        gams_values = getattr(gams_snapshot, group_name)
        collapsed, relevant, ratio = collapse_ratio(py_values, gams_values)
        total_price_collapsed += collapsed
        total_price_relevant += relevant
        if relevant and ratio >= 0.8:
            collapsed_prices[group_name] = ratio

    quantity_ratio = (
        total_quantity_collapsed / total_quantity_relevant
        if total_quantity_relevant
        else 0.0
    )
    price_ratio = (
        total_price_collapsed / total_price_relevant
        if total_price_relevant
        else 0.0
    )

    dead_numeraire_suspected = (
        quantity_ratio >= 0.5
        and any(group in collapsed_prices for group in ("pf", "pft"))
    )

    return {
        "collapsed_quantity_groups": collapsed_quantities,
        "collapsed_price_groups": collapsed_prices,
        "collapsed_quantity_ratio": quantity_ratio,
        "collapsed_price_ratio": price_ratio,
        "dead_numeraire_suspected": dead_numeraire_suspected,
    }


def compare_gtap_gams_parity(
    python_model,
    gams_reference: GTAPGAMSReference,
    tolerance: float = 1e-6,
    *,
    benchmark: Optional[GTAPBenchmarkValues] = None,
    normalize_reference: bool = False,
) -> GTAPParityComparison:
    """Compare a Python model or snapshot against a GAMS reference."""
    py_snapshot = (
        python_model
        if isinstance(python_model, GTAPVariableSnapshot)
        else GTAPVariableSnapshot.from_python_model(python_model)
    )
    gams_snapshot = gams_reference.snapshot
    if normalize_reference:
        if benchmark is None:
            raise ValueError("normalize_reference=True requires a benchmark")
        gams_snapshot = normalize_gams_snapshot_against_benchmark(
            gams_snapshot,
            benchmark,
            gams_reference.sets,
        )

    all_mismatches: List[Dict[str, Any]] = []
    n_compared = 0
    n_mismatches = 0
    max_abs_diff = 0.0
    max_rel_diff = 0.0

    variable_groups = [
        ("xp", py_snapshot.xp, gams_snapshot.xp),
        ("x", py_snapshot.x, gams_snapshot.x),
        ("xs", py_snapshot.xs, gams_snapshot.xs),
        ("xds", py_snapshot.xds, gams_snapshot.xds),
        ("xd", py_snapshot.xd, gams_snapshot.xd),
        ("px", py_snapshot.px, gams_snapshot.px),
        ("pp", py_snapshot.pp, gams_snapshot.pp),
        ("ps", py_snapshot.ps, gams_snapshot.ps),
        ("pd", py_snapshot.pd, gams_snapshot.pd),
        ("pa", py_snapshot.pa, gams_snapshot.pa),
        ("paa", py_snapshot.paa, gams_snapshot.paa),
        ("pdp", py_snapshot.pdp, gams_snapshot.pdp),
        ("pmt", py_snapshot.pmt, gams_snapshot.pmt),
        ("pmcif", py_snapshot.pmcif, gams_snapshot.pmcif),
        ("pet", py_snapshot.pet, gams_snapshot.pet),
        ("pe", py_snapshot.pe, gams_snapshot.pe),
        ("pefob", py_snapshot.pefob, gams_snapshot.pefob),
        ("xe", py_snapshot.xe, gams_snapshot.xe),
        ("xw", py_snapshot.xw, gams_snapshot.xw),
        ("xmt", py_snapshot.xmt, gams_snapshot.xmt),
        ("xet", py_snapshot.xet, gams_snapshot.xet),
        ("xaa", py_snapshot.xaa, gams_snapshot.xaa),
        ("xwmg", py_snapshot.xwmg, gams_snapshot.xwmg),
        ("xmgm", py_snapshot.xmgm, gams_snapshot.xmgm),
        ("pwmg", py_snapshot.pwmg, gams_snapshot.pwmg),
        ("xtmg", py_snapshot.xtmg, gams_snapshot.xtmg),
        ("ptmg", py_snapshot.ptmg, gams_snapshot.ptmg),
        ("xf", py_snapshot.xf, gams_snapshot.xf),
        ("xft", py_snapshot.xft, gams_snapshot.xft),
        ("pf", py_snapshot.pf, gams_snapshot.pf),
        ("pft", py_snapshot.pft, gams_snapshot.pft),
        ("xc", py_snapshot.xc, gams_snapshot.xc),
        ("xg", py_snapshot.xg, gams_snapshot.xg),
        ("xi", py_snapshot.xi, gams_snapshot.xi),
        ("regy", py_snapshot.regy, gams_snapshot.regy),
        ("yc", py_snapshot.yc, gams_snapshot.yc),
        ("yg", py_snapshot.yg, gams_snapshot.yg),
        ("yi", py_snapshot.yi, gams_snapshot.yi),
        ("pabs", py_snapshot.pabs, gams_snapshot.pabs),
    ]

    for group_name, py_vals, gams_vals in variable_groups:
        if not gams_vals:
            continue
        comp, mism, max_d, details = compare_variable_groups(
            py_vals,
            gams_vals,
            group_name,
            tolerance,
        )
        n_compared += comp
        n_mismatches += mism
        max_abs_diff = max(max_abs_diff, max_d)
        all_mismatches.extend(details)
        for detail in details:
            max_rel_diff = max(max_rel_diff, detail.get("rel_diff", 0.0))

    if gams_snapshot.pnum is not None:
        n_compared += 1
        abs_diff = abs((py_snapshot.pnum or 0.0) - gams_snapshot.pnum)
        if abs_diff > tolerance:
            n_mismatches += 1
            all_mismatches.append(
                {
                    "group": "pnum",
                    "key": (),
                    "python": py_snapshot.pnum,
                    "gams": gams_snapshot.pnum,
                    "abs_diff": abs_diff,
                    "rel_diff": 0.0,
                }
            )

    if gams_snapshot.walras is not None:
        n_compared += 1
        abs_diff = abs((py_snapshot.walras or 0.0) - gams_snapshot.walras)
        if abs_diff > tolerance:
            n_mismatches += 1
            all_mismatches.append(
                {
                    "group": "walras",
                    "key": (),
                    "python": py_snapshot.walras,
                    "gams": gams_snapshot.walras,
                    "abs_diff": abs_diff,
                    "rel_diff": 0.0,
                }
            )

    all_mismatches.sort(key=lambda item: item["abs_diff"], reverse=True)
    passed = n_mismatches == 0
    summary = {
        "gams_modelstat": gams_reference.modelstat,
        "gams_solvestat": gams_reference.solvestat,
        "n_variables": n_compared,
        "n_mismatches": n_mismatches,
        "mismatch_rate": n_mismatches / max(n_compared, 1) * 100.0,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
    }
    summary.update(_detect_degenerate_groups(py_snapshot, gams_snapshot))

    return GTAPParityComparison(
        passed=passed,
        tolerance=tolerance,
        n_variables_compared=n_compared,
        n_mismatches=n_mismatches,
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        mismatches=all_mismatches[:50],
        summary=summary,
    )


def _build_standard_gtap_params(
    bundle: GTAPDataBundle,
    *,
    benchmark_year: str,
    solution_year: str,
) -> Tuple[GTAPSets, GTAPParameters]:
    """Build GTAP sets and parameters from a split standard_gtap_7 bundle."""
    sets = GTAPSets()
    sets.load_from_gdx(bundle.sets_gdx)

    resolved_benchmark_csv = (
        _resolve_standard_gtap_reference_csv(bundle.benchmark_csv, sets)
        if bundle.benchmark_csv
        else None
    )

    resolved_benchmark_year = benchmark_year
    resolved_solution_year = solution_year
    if resolved_benchmark_csv:
        resolved_benchmark_year, resolved_solution_year = _resolve_standard_gtap_years(
            resolved_benchmark_csv,
            benchmark_year=benchmark_year,
            solution_year=solution_year,
        )

    if resolved_benchmark_csv:
        _enrich_sets_from_standard_gtap_csv(
            sets,
            resolved_benchmark_csv,
            solution_year=resolved_solution_year,
            benchmark_year=resolved_benchmark_year,
        )

    params = GTAPParameters()
    params.sets = sets

    if bundle.elasticities_gdx:
        params.elasticities.load_from_gdx(bundle.elasticities_gdx, sets)

    if resolved_benchmark_csv:
        params.benchmark = _load_standard_gtap_benchmark_from_csv(
            resolved_benchmark_csv,
            sets,
            benchmark_year=resolved_benchmark_year,
            solution_year=resolved_solution_year,
        )
    elif bundle.benchmark_gdx:
        params.benchmark.load_from_gdx(bundle.benchmark_gdx, sets)
    else:
        raise ValueError("Split GTAP bundle requires benchmark_csv or benchmark_gdx")

    if bundle.benchmark_gdx:
        params.taxes.load_from_gdx(bundle.benchmark_gdx, sets)

    params.taxes.derive_agent_consumption_taxes(params.benchmark, sets)
    params.taxes.derive_trade_route_wedges(params.benchmark, sets)
    if resolved_benchmark_csv:
        _derive_factor_tax_wedges_from_standard_gtap_csv(
            params.taxes,
            resolved_benchmark_csv,
            benchmark_year=resolved_benchmark_year,
            solution_year=resolved_solution_year,
        )
    params.shares.calibrate(params.benchmark, params.elasticities, sets)
    params.calibrated.calibrate_from_benchmark(params.benchmark, params.elasticities, sets)
    if resolved_benchmark_csv:
        from equilibria.templates.gtap.gtap_equilibrium import GTAPEquilibriumSnapshot

        snapshot = GTAPEquilibriumSnapshot.from_csv(resolved_benchmark_csv, year=int(resolved_solution_year))
        params.apply_equilibrium_snapshot(snapshot)
    return sets, params


class GTAPParityRunner:
    """Runner for GTAP parity checks."""

    def __init__(
        self,
        gdx_file: Optional[Path] = None,
        gams_results_gdx: Optional[Path] = None,
        *,
        sets_gdx: Optional[Path] = None,
        elasticities_gdx: Optional[Path] = None,
        benchmark_csv: Optional[Path] = None,
        benchmark_gdx: Optional[Path] = None,
        closure: str = "gtap_standard",
        solver: str = "ipopt",
        solver_options: Optional[Dict[str, Any]] = None,
        tolerance: float = 1e-6,
        benchmark_year: int | str = 2011,
        solution_year: int | str = 1,
        normalize_reference: bool = True,
        warm_start_reference: bool = False,
    ):
        """Initialize the parity runner."""
        self.closure = closure
        self.solver = solver
        self.solver_options = dict(solver_options or {})
        self.tolerance = tolerance
        self.solution_year = _normalize_year_token(solution_year)
        self.benchmark_year = _normalize_year_token(benchmark_year)
        self.normalize_reference = normalize_reference
        self.warm_start_reference = warm_start_reference

        self.gdx_file = Path(gdx_file) if gdx_file else None
        self.gams_results_gdx = Path(gams_results_gdx) if gams_results_gdx else None

        self.bundle: Optional[GTAPDataBundle] = None
        if sets_gdx:
            self.bundle = GTAPDataBundle(
                sets_gdx=Path(sets_gdx),
                elasticities_gdx=Path(elasticities_gdx) if elasticities_gdx else None,
                benchmark_csv=Path(benchmark_csv) if benchmark_csv else None,
                benchmark_gdx=Path(benchmark_gdx) if benchmark_gdx else None,
            )

        if self.bundle:
            self.sets, self.params = _build_standard_gtap_params(
                self.bundle,
                benchmark_year=self.benchmark_year,
                solution_year=self.solution_year,
            )
            self.data_label = str(self.bundle.sets_gdx)
        elif self.gdx_file:
            self.sets = GTAPSets()
            self.sets.load_from_gdx(self.gdx_file)
            self.params = GTAPParameters()
            self.params.load_from_gdx(self.gdx_file)
            self.data_label = str(self.gdx_file)
        else:
            raise ValueError("Provide either gdx_file or sets_gdx")

        self.contract = build_gtap_contract({"closure": closure})
        self.equations = GTAPModelEquations(self.sets, self.params, self.contract.closure)
        self.model = self.equations.build_model()

        self.gams_reference: Optional[GTAPGAMSReference] = None
        if self.gams_results_gdx and self.gams_results_gdx.exists():
            self.gams_reference = GTAPGAMSReference.load(
                self.gams_results_gdx,
                self.sets,
                solution_year=self.solution_year,
            )

    def run_python(self) -> SolverResult:
        """Run the Python GTAP model."""
        logger.info("Running Python GTAP model...")
        solver = GTAPSolver(
            self.model,
            self.contract.closure,
            solver_name=self.solver,
            solver_options=self.solver_options,
        )
        if self.gams_reference and self.warm_start_reference:
            reference_hint = self.gams_reference.snapshot
            if self.normalize_reference:
                reference_hint = normalize_gams_snapshot_against_benchmark(
                    reference_hint,
                    self.params.benchmark,
                    self.sets,
                )
            solver.apply_solution_hint(reference_hint)
        result = solver.solve()
        logger.info("Python solve: %s, Walras=%s", result.status.value, result.walras_value)
        return result

    def run_gams(self, gams_script: Optional[Path] = None) -> GTAPGAMSReference:
        """Return the available GAMS reference."""
        if gams_script and Path(gams_script).exists():
            raise NotImplementedError("GAMS execution is not implemented in the parity runner")
        if self.gams_reference:
            return self.gams_reference
        raise ValueError("No GAMS reference available")

    def run_parity_check(self) -> GTAPParityComparison:
        """Run the full parity check."""
        py_result = self.run_python()
        if not py_result.success:
            return GTAPParityComparison(
                passed=False,
                tolerance=self.tolerance,
                n_variables_compared=0,
                n_mismatches=0,
                max_abs_diff=0.0,
                max_rel_diff=0.0,
                mismatches=[],
                summary={"error": "Python solve failed", "message": py_result.message},
            )

        if not self.gams_reference:
            return GTAPParityComparison(
                passed=False,
                tolerance=self.tolerance,
                n_variables_compared=0,
                n_mismatches=0,
                max_abs_diff=0.0,
                max_rel_diff=0.0,
                mismatches=[],
                summary={"error": "No GAMS reference"},
            )

        return compare_gtap_gams_parity(
            self.model,
            self.gams_reference,
            tolerance=self.tolerance,
            benchmark=self.params.benchmark,
            normalize_reference=self.normalize_reference,
        )

    def generate_report(self, comparison: GTAPParityComparison) -> str:
        """Generate a plain-text report."""
        lines = [
            "=" * 70,
            "GTAP Parity Check Report",
            "=" * 70,
            f"Data source: {self.data_label}",
            f"GAMS results: {self.gams_results_gdx}",
            f"Closure: {self.closure}",
            f"Tolerance: {self.tolerance}",
            "",
            f"Status: {'PASSED' if comparison.passed else 'FAILED'}",
            f"Variables compared: {comparison.n_variables_compared}",
            f"Mismatches: {comparison.n_mismatches}",
            f"Max absolute diff: {comparison.max_abs_diff:.2e}",
            f"Max relative diff: {comparison.max_rel_diff:.2e}",
            "",
        ]

        if comparison.summary.get("dead_numeraire_suspected"):
            lines.append(
                "Diagnostic: Python solution shows broad quantity collapse with pf/pft near zero; numeraire linkage is likely incomplete."
            )
            lines.append("")

        if comparison.mismatches:
            lines.extend(["Top Mismatches:", "-" * 70])
            for mismatch in comparison.mismatches[:20]:
                lines.append(
                    f"  {mismatch['group']}{mismatch['key']}: "
                    f"Python={mismatch['python']:.6f} "
                    f"GAMS={mismatch['gams']:.6f} "
                    f"Diff={mismatch['abs_diff']:.6e}"
                )

        lines.append("=" * 70)
        return "\n".join(lines)


def load_gtap_gams_reference(
    gdx_path: Path,
    sets: Optional[GTAPSets] = None,
    *,
    solution_year: int | str = 1,
) -> GTAPGAMSReference:
    """Load a GTAP GAMS reference from CSV or GDX."""
    return GTAPGAMSReference.load(gdx_path, sets, solution_year=solution_year)


def run_gtap_parity_test(
    gdx_file: Optional[Path] = None,
    gams_results_gdx: Optional[Path] = None,
    *,
    sets_gdx: Optional[Path] = None,
    elasticities_gdx: Optional[Path] = None,
    benchmark_csv: Optional[Path] = None,
    benchmark_gdx: Optional[Path] = None,
    closure: str = "gtap_standard",
    solver_options: Optional[Dict[str, Any]] = None,
    tolerance: float = 1e-6,
    benchmark_year: int | str = 2011,
    solution_year: int | str = 1,
    normalize_reference: bool = True,
    output_file: Optional[Path] = None,
) -> GTAPParityComparison:
    """Convenience wrapper for a parity run."""
    runner = GTAPParityRunner(
        gdx_file=gdx_file,
        gams_results_gdx=gams_results_gdx,
        sets_gdx=sets_gdx,
        elasticities_gdx=elasticities_gdx,
        benchmark_csv=benchmark_csv,
        benchmark_gdx=benchmark_gdx,
        closure=closure,
        solver_options=solver_options,
        tolerance=tolerance,
        benchmark_year=benchmark_year,
        solution_year=solution_year,
        normalize_reference=normalize_reference,
    )
    result = runner.run_parity_check()
    report = runner.generate_report(result)
    print(report)
    if output_file:
        Path(output_file).write_text(report, encoding="utf-8")
    return result
