"""GTAP Parameters (Standard GTAP 7)

This module defines all GTAP model parameters following the GTAP Standard 7 implementation.
Reference: /Users/marmol/proyectos2/cge_babel/standard_gtap_7/model.gms

Parameters include:
- Elasticities (substitution, transformation)
- Benchmark SAM values
- Tax rates
- Share parameters (from calibration)
- Technical change parameters
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import math
from statistics import median

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values
from equilibria.babel.gdx.gdxdump import(
    read_parameter_with_gdxdump,
    read_variable_levels_with_gdxdump,
)
from equilibria.templates.gtap.gtap_equilibrium import GTAPEquilibriumSnapshot
from equilibria.templates.gtap.gtap_sets import GTAPSets
from equilibria.templates.gtap.gtap_std7_mapping import (
    get_benchmark_parameter_name,
    get_tax_parameter_name,
    get_elasticity_parameter_name,
    reorder_parameter_keys,
)

GTAP_HOUSEHOLD_AGENT = "hhd"
GTAP_GOVERNMENT_AGENT = "gov"
GTAP_INVESTMENT_AGENT = "inv"
GTAP_MARGIN_AGENT = "tmg"


@dataclass
class GTAPElasticities:
    """Elasticity parameters for GTAP model.
    
    Key elasticities:
    - esubva: CES elasticity between value-added and intermediate demand
    - esubt: CES elasticity between primary factors and intermediates (top production nest)
    - esubd: CES elasticity between domestic and imported goods (top Armington)
    - esubm: CES elasticity across import sources (bottom Armington)
    - etrae: CET elasticity for factor mobility across sectors
    - omegax: CET elasticity between domestic sales and exports
    - omegaw: CET elasticity across export destinations
    """
    
    # Production elasticities
    esubva: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, a)
    esubt: Dict[Tuple[str, str], float] = field(default_factory=dict)   # (r, a)
    esubd: Dict[Tuple[str, str], float] = field(default_factory=dict)   # (r, i)
    
    # Trade elasticities - Armington
    esubm: Dict[Tuple[str, str], float] = field(default_factory=dict)   # (r, i)
    
    # Trade elasticities - CET
    omegax: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, i)
    omegaw: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, i)
    omegas: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, a) - activity transformation
    sigmas: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, i) - commodity aggregation
    etraq: Dict[Tuple[str, str], float] = field(default_factory=dict)   # (r, a) - source for omegas
    esubq: Dict[Tuple[str, str], float] = field(default_factory=dict)   # (r, i) - source for sigmas

    # Factor mobility elasticities
    etrae: Dict[str, float] = field(default_factory=dict)  # f
    omegaf: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, f) CET factor mobility
    etaff: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, f, a) sector-specific supply

    # Demand elasticities
    esubg: Dict[str, float] = field(default_factory=dict)  # r (government)
    esubi: Dict[str, float] = field(default_factory=dict)  # r (investment)
    esubc: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, h) consumption
    incpar: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, i) CDE expansion parameter
    subpar: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, i) CDE substitution parameter

    # Transport margins
    sigmam: Dict[str, float] = field(default_factory=dict)  # m

    # Nested CES elasticities for the ND/VA blocks
    sigmap: Dict[Tuple[str, str], float] = field(default_factory=dict)
    sigmand: Dict[Tuple[str, str], float] = field(default_factory=dict)
    sigmav: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    def load_from_gdx(self, gdx_path: Path, sets: GTAPSets) -> None:
        """Load elasticities from GDX file using GTAP Standard 7 native parameter names."""
        gdx_data = read_gdx(gdx_path)

        # Load production elasticities (GTAP Std 7 already uses ESUBVA, ESUBD uppercase)
        self._load_parameter(gdx_data, gdx_path, "esubva", self.esubva, (sets.r, sets.a))
        self._load_parameter(gdx_data, gdx_path, "esubt", self.esubt, (sets.r, sets.a))
        self._load_parameter(gdx_data, gdx_path, "esubd", self.esubd, (sets.r, sets.i))

        # Load trade elasticities
        self._load_parameter(gdx_data, gdx_path, "esubm", self.esubm, (sets.r, sets.i))
        self._load_parameter(gdx_data, gdx_path, "omegax", self.omegax, (sets.r, sets.i))
        self._load_parameter(gdx_data, gdx_path, "omegaw", self.omegaw, (sets.r, sets.i))
        self._load_parameter(gdx_data, gdx_path, "omegas", self.omegas, (sets.r, sets.a))
        self._load_parameter(gdx_data, gdx_path, "sigmas", self.sigmas, (sets.r, sets.i))
        self._load_parameter(gdx_data, gdx_path, "etraq", self.etraq, (sets.r, sets.a))
        self._load_parameter(gdx_data, gdx_path, "esubq", self.esubq, (sets.r, sets.i))

        # Load factor mobility
        self._load_parameter(gdx_data, gdx_path, "etrae", self.etrae, (sets.f,))
        self._load_parameter(gdx_data, gdx_path, "omegaf", self.omegaf, (sets.r, sets.f))
        self._load_parameter(gdx_data, gdx_path, "etaff", self.etaff, (sets.r, sets.f, sets.a))

        # Load demand elasticities
        self._load_parameter(gdx_data, gdx_path, "esubg", self.esubg, (sets.r,))
        self._load_parameter(gdx_data, gdx_path, "esubi", self.esubi, (sets.r,))
        self._load_parameter(gdx_data, gdx_path, "incpar", self.incpar, (sets.r, sets.i))
        self._load_parameter(gdx_data, gdx_path, "subpar", self.subpar, (sets.r, sets.i))

        # Nested CES elasticities
        self._load_parameter(gdx_data, gdx_path, "sigmap", self.sigmap, (sets.r, sets.a))
        self._load_parameter(gdx_data, gdx_path, "sigmand", self.sigmand, (sets.r, sets.a))
        self._load_parameter(gdx_data, gdx_path, "sigmav", self.sigmav, (sets.r, sets.a))

        # Populate nested CES elasticities from the production elasticities
        self.initialize_nested_elasticities(sets)

    def override_omegas_sigmas_from_gdx(self, gdx_path: Path, sets: GTAPSets) -> None:
        """Override only make-aggregation elasticities from a calibration GDX.

        This is used to mirror GAMS COMP-calibrated values for:
        - omegas(r,a): activity transformation elasticity
        - sigmas(r,i): commodity aggregation elasticity
        """
        gdx_data = read_gdx(gdx_path)

        override_omegas: Dict[Tuple[str, str], float] = {}
        override_sigmas: Dict[Tuple[str, str], float] = {}
        self._load_parameter(gdx_data, gdx_path, "omegas", override_omegas, (sets.r, sets.a))
        self._load_parameter(gdx_data, gdx_path, "sigmas", override_sigmas, (sets.r, sets.i))

        if override_omegas:
            self.omegas.update(override_omegas)
        if override_sigmas:
            self.sigmas.update(override_sigmas)

    def _load_parameter(
        self,
        gdx_data: Dict,
        gdx_path: Path,
        name: str,
        target: Dict,
        index_sets: Tuple[Sequence[str], ...] = (),
    ) -> None:
        """Load a single elasticity parameter using GTAP Standard 7 native names.
        
        Args:
            gdx_data: GDX data dictionary
            gdx_path: Path to GDX file for gdxdump fallback
            name: Internal parameter name (lowercase like 'esubva', 'esubd')
            target: Target dictionary to populate
            index_sets: Tuple of index sets for key validation
        """
        # Convert internal name to GTAP Std 7 name
        gtap_name = get_elasticity_parameter_name(name)
        
        values: Dict[Tuple[str, ...], float] = {}
        try:
            values = read_parameter_values(gdx_data, gtap_name)
        except (KeyError, ValueError):
            pass

        if not values:
            values = read_parameter_with_gdxdump(gdx_path, gtap_name)
        
        # Reorder keys if needed
        if values:
            values = reorder_parameter_keys(gtap_name, values)

        if values:
            normalized_sets = tuple(set(index) for index in index_sets)
            for raw_key, raw_value in values.items():
                if raw_value is None:
                    continue
                try:
                    numeric = float(raw_value)
                except (TypeError, ValueError):
                    continue
                key_tuple = (
                    tuple(raw_key) if isinstance(raw_key, tuple) else (raw_key,)
                    if raw_key else ()
                )
                normalized_key = self._align_key(key_tuple, index_sets, normalized_sets)
                target[normalized_key] = numeric

    def _align_key(
        self,
        key_tuple: Tuple[str, ...],
        index_sets: Tuple[Sequence[str], ...],
        normalized_sets: Tuple[set, ...],
    ) -> Tuple[str, ...]:
        if not index_sets or len(key_tuple) != len(index_sets):
            return key_tuple
        aligned: list[str] = []
        used_indices: set[int] = set()
        for set_idx, desired in enumerate(normalized_sets):
            found = None
            for element_idx, element in enumerate(key_tuple):
                if element_idx in used_indices:
                    continue
                if element in desired:
                    found = element
                    used_indices.add(element_idx)
                    break
            if found is None:
                return key_tuple
            aligned.append(found)
        return tuple(aligned)

    def _broadcast_index_tuples(self, labels: Tuple[str, ...], expected_dim: int, membership_sets: List[set], index_sets: Tuple[List[str], ...]) -> List[Tuple[str, ...]]:
        if expected_dim == 2 and len(labels) == 1:
            label = labels[0]
            if label in membership_sets[1]:
                return [(region, label) for region in index_sets[0]]
            if label in membership_sets[0]:
                return [(label, other) for other in index_sets[1]]
        return []

    def calibrate_from_comp_gdx(self, comp_gdx: Path, sets: GTAPSets) -> None:
        """Estimate sigm* elasticities using the COMP GDX time series."""
        va_series = self._series_from_gdx(comp_gdx, "va", (0, 1))
        nd_series = self._series_from_gdx(comp_gdx, "nd", (0, 1))
        pva_series = self._series_from_gdx(comp_gdx, "pva", (0, 1))
        pnd_series = self._series_from_gdx(comp_gdx, "pnd", (0, 1))

        self._calibrate_sigmap(va_series, nd_series, pnd_series, pva_series)

        xa_raw = read_variable_levels_with_gdxdump(comp_gdx, "xa")
        pa_raw = read_variable_levels_with_gdxdump(comp_gdx, "pa")
        xa_series = self._aggregate_series(xa_raw, (0, 2))
        price_pa = self._weighted_average(pa_raw, xa_raw, (0, 2))
        self._calibrate_sigmand(nd_series, xa_series, pnd_series, price_pa)

        xf_raw = read_variable_levels_with_gdxdump(comp_gdx, "xf")
        pf_raw = read_variable_levels_with_gdxdump(comp_gdx, "pf")
        total_xf = self._aggregate_series(xf_raw, (0, 2))
        share_xf = self._factor_share_series(xf_raw, total_xf)
        price_ratio_pf = self._pf_over_pva(pf_raw, pva_series)
        self._calibrate_sigmav(share_xf, price_ratio_pf)

    def _calibrate_sigmap(self, va_series, nd_series, pnd_series, pva_series) -> None:
        for key, va_data in va_series.items():
            nd_data = nd_series.get(key)
            if not nd_data:
                continue
            price_data = self._price_ratio(pnd_series.get(key, {}), pva_series.get(key, {}))
            share_series = self._ratios(va_data, nd_data)
            sigma = self._estimate_sigma_from_series(share_series, price_data)
            if sigma is not None:
                self.sigmap[key] = sigma

    def _calibrate_sigmand(self, nd_series, xa_series, pnd_series, pa_series) -> None:
        for key, nd_data in nd_series.items():
            xa_data = xa_series.get(key)
            if not xa_data:
                continue
            share_series = self._ratios(nd_data, xa_data)
            price_data = self._price_ratio(pnd_series.get(key, {}), pa_series.get(key, {}))
            sigma = self._estimate_sigma_from_series(share_series, price_data)
            if sigma is not None:
                self.sigmand[key] = sigma

    def _calibrate_sigmav(self, share_xf, price_ratio_pf) -> None:
        factor_sigmas: Dict[Tuple[str, str], list[float]] = defaultdict(list)
        for (r, f, a), share_series in share_xf.items():
            price_series = price_ratio_pf.get((r, f, a))
            if not price_series:
                continue
            sigma = self._estimate_sigma_from_series(share_series, price_series)
            if sigma is not None:
                factor_sigmas[(r, a)].append(sigma)

        for key, values in factor_sigmas.items():
            if values:
                self.sigmav[key] = median(values)

    def _series_from_gdx(self, gdx_path: Path, var_name: str, prefix_indices: Tuple[int, ...]) -> Dict[Tuple[str, ...], Dict[str, float]]:
        raw = read_variable_levels_with_gdxdump(gdx_path, var_name)
        series: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for key, value in raw.items():
            prefix = tuple(key[idx] for idx in prefix_indices)
            time_label = key[-1]
            series[prefix][time_label] += value
        return series

    def _aggregate_series(self, raw_data: dict, prefix_indices: Tuple[int, ...]) -> Dict[Tuple[str, ...], Dict[str, float]]:
        result: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for key, value in raw_data.items():
            prefix = tuple(key[idx] for idx in prefix_indices)
            time_label = key[-1]
            result[prefix][time_label] += value
        return result

    def _weighted_average(self, price_data: dict, weight_data: dict, prefix_indices: Tuple[int, ...]) -> Dict[Tuple[str, ...], Dict[str, float]]:
        numerators: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        denominators: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for key, price in price_data.items():
            prefix = tuple(key[idx] for idx in prefix_indices)
            time_label = key[-1]
            weight = weight_data.get(key)
            if weight is None:
                continue
            numerators[prefix][time_label] += price * weight
            denominators[prefix][time_label] += weight
        averages: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(dict)
        for prefix, times in numerators.items():
            for time_label, num in times.items():
                denom = denominators[prefix].get(time_label, 0.0)
                if denom > 0:
                    averages[prefix][time_label] = num / denom
        return averages

    def _factor_share_series(self, xf_raw: dict, totals: Dict[Tuple[str, ...], Dict[str, float]]) -> Dict[Tuple[str, str, str], Dict[str, float]]:
        shares: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(dict)
        for key, value in xf_raw.items():
            r, f, a, time_label = key
            total = totals.get((r, a), {}).get(time_label, 0.0)
            if total > 0:
                shares[(r, f, a)][time_label] = value / total
        return shares

    def _pf_over_pva(self, pf_raw: dict, pva_series: Dict[Tuple[str, ...], Dict[str, float]]) -> Dict[Tuple[str, str, str], Dict[str, float]]:
        ratios: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(dict)
        for key, pf_value in pf_raw.items():
            r, f, a, time_label = key
            pva_value = pva_series.get((r, a), {}).get(time_label)
            if pva_value and pva_value > 0:
                ratios[(r, f, a)][time_label] = pf_value / pva_value
        return ratios

    def _ratios(self, num_series: Dict[str, float], denom_series: Dict[str, float]) -> Dict[str, float]:
        ratios: Dict[str, float] = {}
        for time_label, num in num_series.items():
            denom = denom_series.get(time_label)
            if denom and denom > 0:
                ratios[time_label] = num / denom
        return ratios

    def _price_ratio(self, numerator: Dict[str, float], denominator: Dict[str, float]) -> Dict[str, float]:
        ratio: Dict[str, float] = {}
        for time_label, num in numerator.items():
            denom = denominator.get(time_label)
            if denom and denom > 0:
                ratio[time_label] = num / denom
        return ratio

    def _estimate_sigma_from_series(self, share_series: Dict[str, float], price_series: Dict[str, float]) -> float | None:
        time_order = {'base': 1, 't0': 1, 'check': 2, 'shock': 3}
        common_times = sorted(
            set(share_series.keys()) & set(price_series.keys()),
            key=lambda label: time_order.get(label, float('inf')),
        )
        sigmas: list[float] = []
        for i in range(len(common_times) - 1):
            t1 = common_times[i]
            t2 = common_times[i + 1]
            share1 = share_series[t1]
            share2 = share_series[t2]
            price1 = price_series[t1]
            price2 = price_series[t2]
            if share1 <= 0 or share2 <= 0 or price1 <= 0 or price2 <= 0 or price1 == price2:
                continue
            try:
                sigmas.append(math.log(share2 / share1) / math.log(price2 / price1))
            except (ValueError, ZeroDivisionError):
                continue
        if not sigmas:
            return None
        return median(sigmas)

    def initialize_nested_elasticities(self, sets: GTAPSets) -> None:
        """Ensure sigmap, sigmand and sigmav cover every (region, activity)."""
        default_value = 1.0
        for r in sets.r:
            for a in sets.a:
                # GAMS cal.gms:
                #   sigmap  <- esubt(a,r) by default
                #   sigmand <- sigmap by default
                #   sigmav  <- esubva(a,r) by default
                sigmap_default = float(self.esubt.get((r, a), 0.0))
                sigmap_value = self.sigmap.get((r, a), sigmap_default)
                self.sigmap[(r, a)] = sigmap_value
                self.sigmand[(r, a)] = self.sigmand.get((r, a), sigmap_value)
                self.sigmav[(r, a)] = self.sigmav.get((r, a), self.esubva.get((r, a), default_value))

        # Populate GTAP equation-level elasticities used in model.gms blocks.
        # If not explicitly present in GDX, map from available calibrated elasticities.
        for r in sets.r:
            for a in sets.a:
                if (r, a) in self.omegas:
                    continue
                # GAMS getData.gms: omegas(r,a) = -etraq(a,r)
                etraq_val = self.etraq.get((r, a))
                if etraq_val is not None:
                    self.omegas[(r, a)] = -float(etraq_val)
                    continue
                self.omegas[(r, a)] = self.sigmap.get((r, a), default_value)
        for r in sets.r:
            for i in sets.i:
                if (r, i) not in self.sigmas:
                    # GAMS getData.gms:
                    # sigmas(r,i) = inf$(esubq(i,r)=0) + (1/esubq(i,r))$(esubq(i,r)<>0)
                    esubq_val = float(self.esubq.get((r, i), 0.0) or 0.0)
                    if abs(esubq_val) <= 1e-12:
                        self.sigmas[(r, i)] = float("inf")
                    else:
                        self.sigmas[(r, i)] = 1.0 / esubq_val
                    # Fallback to legacy mapping only if derivative mapping produced NaN.
                    if math.isnan(self.sigmas[(r, i)]):
                        self.sigmas[(r, i)] = self.esubd.get((r, i), 2.0)
                self.omegax[(r, i)] = self.omegax.get((r, i), float("inf"))
                self.omegaw[(r, i)] = self.omegaw.get((r, i), float("inf"))


@dataclass
class GTAPCalibratedShares:
    """GAMS-style calibrated share parameters.
    
    These parameters are calibrated from benchmark SAM data to ensure
    the benchmark is an exact solution of the CES/CET equations.
    
    Following GAMS cal.gms lines 724-730:
    - and: ND bundle share parameter (calibrated with sigmap)
    - ava: VA bundle share parameter (calibrated with sigmap)
    - io: Input-output coefficients wrt ND (calibrated with sigmand)
    - af: Factor shares wrt VA (calibrated with sigmav)
    - gx: CET share parameter for commodity supply (calibrated with omegas)
    """
    
    # Production shares (GAMS: and, ava)
    and_param: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, a) - ND bundle share
    ava_param: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, a) - VA bundle share
    
    # IO coefficients (GAMS: io)
    io_param: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, a) - IO coefficient
    
    # Factor shares (GAMS: af)
    af_param: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, f, a) - Factor share
    
    # Make shares (GAMS: gx)
    gx_param: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, a, i) - CET output share
    
    def calibrate_from_benchmark(
        self,
        benchmark: "GTAPBenchmarkValues",
        elasticities: GTAPElasticities,
        sets: GTAPSets,
        taxes: "GTAPTaxRates | None" = None,
    ) -> None:
        """Calibrate all share parameters from benchmark data.
        
        This follows GAMS cal.gms calibration logic (lines 724-730).
        All prices are assumed = 1.0 in benchmark (GAMS convention).
        """
        # All benchmark prices are normalized to 1.0 except the pre-tax make
        # price, which uses the output tax wedge when available so the GTAP
        # make equations start from the same tax-adjusted normalization as GAMS.
        px = pnd = pva = pa = 1.0

        def _kappa(r: str, f: str, a: str) -> float:
            if taxes is None:
                return 0.0
            kappa = float(taxes.kappaf_activity.get((r, f, a), 0.0) or 0.0)
            if kappa == 0.0:
                kappa = float(taxes.kappaf.get((r, f), 0.0) or 0.0)
            return kappa

        def _pf_bench(r: str, f: str, a: str) -> float:
            return max(1.0 / max(1.0 - _kappa(r, f, a), 1e-12), 1e-12)

        def _pfa_bench(r: str, f: str, a: str) -> float:
            factor_tax = float(taxes.rtf.get((r, f, a), 0.0) or 0.0) if taxes is not None else 0.0
            return _pf_bench(r, f, a) * max(1.0 + factor_tax, 1e-12)
        
        # Calculate intermediate values needed for calibration
        nd_values = {}  # ND bundle values
        va_values = {}  # VA bundle values
        xp_values = {}  # Production values
        xa_values = {}  # Armington demand values
        xf_values = {}  # Factor demand values
        
        # Calculate ND, VA, XP from benchmark
        for r in sets.r:
            for a in sets.a:
                # Value added (sum of factor payments)
                # Match GAMS initialization:
                # xf = EVFB/pf, pfa = pf*(1+rtf), va = sum(pfa*xf)/pva
                # => va contribution = EVFB*(1+rtf) when pva=1.
                va_val = 0.0
                for f in sets.f:
                    evfb_val = float(benchmark.evfb.get((r, f, a), benchmark.vfm.get((r, f, a), 0.0)) or 0.0)
                    if evfb_val <= 0.0:
                        continue
                    factor_tax = float(taxes.rtf.get((r, f, a), 0.0) or 0.0) if taxes is not None else 0.0
                    va_val += evfb_val * (1.0 + factor_tax)
                if va_val > 0:
                    va_values[(r, a)] = va_val

                # Intermediate demand bundle at purchaser prices (GAMS-consistent):
                # nd = sum_i pa*xa, where pa*xa corresponds to vdfp + vmfp in benchmark values.
                nd_val = sum(
                    benchmark.vdfp.get((r, i, a), 0.0) + benchmark.vmfp.get((r, i, a), 0.0)
                    for i in sets.i
                )
                if nd_val <= 0.0:
                    nd_val = sum(
                        benchmark.vdfm.get((r, i, a), 0.0) + benchmark.vifm.get((r, i, a), 0.0)
                        for i in sets.i
                    )
                if nd_val <= 0.0:
                    nd_val = 0.0
                if nd_val > 0:
                    nd_values[(r, a)] = nd_val

                # Production value at purchaser prices (GAMS xp identity):
                # xp = sum_i(pdp*xd + pmp*xm) + sum_f(pfa*xf) = nd + va when prices are 1 at benchmark.
                xp_cost_val = nd_val + va_val

                # Fallbacks for sparse/missing cost blocks.
                xp_val = xp_cost_val
                if xp_val <= 0.0:
                    xp_val = benchmark.vom.get((r, a), 0.0)
                if xp_val <= 0.0:
                    outputs = sets.activity_commodities.get(a, [sets.a_to_i.get(a, a)])
                    xp_val = sum(benchmark.makb.get((r, a, i), 0.0) for i in outputs)

                if xp_val > 0:
                    xp_values[(r, a)] = xp_val
        
        # Calibrate AND and AVA (GAMS cal.gms lines 724-725)
        for r in sets.r:
            for a in sets.a:
                xp_val = xp_values.get((r, a), 0.0)
                if xp_val <= 0:
                    continue
                
                sigmap = elasticities.sigmap.get((r, a), 1.0)
                
                # and(r,a,t) = (nd.l/xp.l)*(pnd/px)**sigmap
                nd_val = nd_values.get((r, a), 0.0)
                if nd_val > 0:
                    price_ratio = pnd / px  # = 1.0
                    self.and_param[(r, a)] = (nd_val / xp_val) * (price_ratio ** sigmap)
                
                # ava(r,a,t) = (va.l/xp.l)*(pva/px)**sigmap
                va_val = va_values.get((r, a), 0.0)
                if va_val > 0:
                    price_ratio = pva / px  # = 1.0
                    self.ava_param[(r, a)] = (va_val / xp_val) * (price_ratio ** sigmap)
        
        # Calculate XA (Armington demand) for IO calibration
        for r in sets.r:
            for a in sets.a:
                for i in sets.i:
                    # Use purchaser-value intermediate demand for io calibration.
                    # This keeps xa and nd on the same valuation basis in the
                    # GAMS-style io identity.
                    domestic = benchmark.vdfp.get((r, i, a), 0.0)
                    imported = benchmark.vmfp.get((r, i, a), 0.0)
                    xa_val = domestic + imported
                    if xa_val > 0:
                        xa_values[(r, i, a)] = xa_val
        
        # Calibrate IO coefficients (GAMS cal.gms line 726)
        for r in sets.r:
            for a in sets.a:
                nd_val = nd_values.get((r, a), 0.0)
                if nd_val <= 0:
                    continue
                
                sigmand = elasticities.sigmand.get((r, a), 1.0)
                
                for i in sets.i:
                    xa_val = xa_values.get((r, i, a), 0.0)
                    if xa_val <= 0:
                        continue

                    # io(r,i,a,t) = (xa.l/nd.l)*(pa/pnd)**sigmand
                    # With purchaser-valued xa and nd and benchmark-normalized
                    # price indices, the price term is 1 at calibration point.
                    price_ratio = pa / pnd
                    self.io_param[(r, i, a)] = (xa_val / nd_val) * (price_ratio ** sigmand)
        
        # Calculate XF (factor demand) values
        for r in sets.r:
            for f in sets.f:
                for a in sets.a:
                    vfm_val = float(benchmark.evfb.get((r, f, a), benchmark.vfm.get((r, f, a), 0.0)) or 0.0)
                    pf_val = _pf_bench(r, f, a)
                    xf_val = vfm_val / pf_val
                    if xf_val > 0:
                        xf_values[(r, f, a)] = xf_val
        
        # Calibrate AF (factor shares) (GAMS cal.gms line 727)
        for r in sets.r:
            for a in sets.a:
                va_val = va_values.get((r, a), 0.0)
                if va_val <= 0:
                    continue
                
                sigmav = elasticities.sigmav.get((r, a), 1.0)
                
                for f in sets.f:
                    xf_val = xf_values.get((r, f, a), 0.0)
                    if xf_val <= 0:
                        continue
                    
                    # GAMS calibration uses the tax-inclusive factor price term
                    # in xfeq: af = (xf/va) * (M_PFA/pva)**sigmav.
                    # Here M_PFA maps to pfa = pf*(1+rtf), with pf from kappaf.
                    pfa_term = _pfa_bench(r, f, a)
                    price_ratio = pfa_term / max(pva, 1e-12)
                    self.af_param[(r, f, a)] = (xf_val / va_val) * (price_ratio ** sigmav)
        
        # Calibrate GX (make shares) (GAMS cal.gms lines 729-731)
        for r in sets.r:
            for a in sets.a:
                xp_val = xp_values.get((r, a), 0.0)
                if xp_val <= 0:
                    continue
                
                # Get GAMS activity transformation elasticity omegas(r,a)
                omegas = elasticities.omegas.get((r, a), 1.0)
                output_tax = 0.0
                if taxes is not None:
                    output_tax = float(taxes.rto.get((r, a), 0.0) or 0.0)
                p = 1.0 / max(1.0 + output_tax, 1e-12)
                
                outputs = sets.activity_commodities.get(a, [sets.a_to_i.get(a, a)])
                for i in outputs:
                    x_val = benchmark.makb.get((r, a, i), 0.0)
                    if x_val <= 0:
                        continue

                    # Use commodity-specific make-price ratio when available.
                    # This matches GAMS multi-output activity calibration where
                    # MAKB/MAKS wedges can differ by commodity within activity.
                    maks_val = float(benchmark.maks.get((r, a, i), 0.0) or 0.0)
                    p_rai = p
                    if maks_val > 0.0:
                        p_rai = maks_val / max(float(x_val), 1e-12)
                    
                    if omegas != float('inf'):
                        # gx(r,a,i,t) = (x.l/xp.l)*(px/p)**omegas
                        price_ratio = px / max(p_rai, 1e-12)
                        self.gx_param[(r, a, i)] = (x_val / xp_val) * (price_ratio ** omegas)
                    else:
                        # Perfect transformation: gx = value share
                        self.gx_param[(r, a, i)] = (p * x_val) / (px * xp_val)


@dataclass
class GTAPBenchmarkValues:
    """Benchmark SAM values from GTAP data.
    
    These are the base year values used for calibration.
    """
    
    # Production and supply
    vom: Dict[Tuple[str, str], float] = field(default_factory=dict)     # (r, a) - Output at market prices
    vom_i: Dict[Tuple[str, str], float] = field(default_factory=dict)   # (r, i) - Output by commodity
    makb: Dict[Tuple[str, str, str], float] = field(
        default_factory=dict
    )  # (r, a, i) - Make/output pairs (value)
    maks: Dict[Tuple[str, str, str], float] = field(
        default_factory=dict
    )  # (r, a, i) - Make/output pairs (value) (alternative)
    
    # Factor payments
    vfm: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, f, a) - Factor payments
    vfb: Dict[Tuple[str, str], float] = field(default_factory=dict)     # (r, f) - Factor income
    evfb: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, f, a) - Factor payments at basic prices
    evos: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, f, a) - Factor remuneration net of direct tax
    vmfp: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, a) - Imported factors at purchaser prices?
    vmfb: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, a) - Factor payments imported
    
    # Intermediate demand
    vdfm: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, a) - Domestic intermediate
    vifm: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, a) - Imported intermediate
    vdfp: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, a) - Domestic intermediate at purchaser
    vdfb: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, a) - Domestic intermediate at basic
    vmpp: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, a) - ?
    vmpb: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    vdpp: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    vdpb: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    vdgp: Dict[Tuple[str, str], float] = field(default_factory=dict)
    vdgb: Dict[Tuple[str, str], float] = field(default_factory=dict)
    vmgp: Dict[Tuple[str, str], float] = field(default_factory=dict)
    vmgb: Dict[Tuple[str, str], float] = field(default_factory=dict)
    vdip: Dict[Tuple[str, str], float] = field(default_factory=dict)
    vdib: Dict[Tuple[str, str], float] = field(default_factory=dict)
    vmip: Dict[Tuple[str, str], float] = field(default_factory=dict)
    vmib: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    # Final demand - Private consumption
    vpm: Dict[Tuple[str, str], float] = field(default_factory=dict)     # (r, i) - Private consumption
    
    # Final demand - Government
    vgm: Dict[Tuple[str, str], float] = field(default_factory=dict)     # (r, i) - Government consumption
    
    # Final demand - Investment
    vim: Dict[Tuple[str, str], float] = field(default_factory=dict)     # (r, i) - Investment demand
    
    # Trade flows
    vxmd: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, rp) - Exports (fob)
    vswd: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, rp) - Exports (fob, by source)
    vtwr: Dict[Tuple[str, str, str, str], float] = field(default_factory=dict)  # (r, i, rp, m) - Transport margins
    vxsb: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, rp) - Exports at basic prices
    vfob: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    vcif: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    vmsb: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    vst: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    # Trade at market prices
    viws: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, rp) - Imports (cif)
    vims: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, rp) - Imports ( tariff)

    # Savings, depreciation, and population
    save: Dict[str, float] = field(default_factory=dict)  # r - Net saving
    vdep: Dict[str, float] = field(default_factory=dict)  # r - Capital depreciation
    vkb: Dict[str, float] = field(default_factory=dict)   # r - Capital stock
    pop: Dict[str, float] = field(default_factory=dict)   # r - Population
    
    # Income and savings
    yp: Dict[str, float] = field(default_factory=dict)      # r - Private income
    yg: Dict[str, float] = field(default_factory=dict)      # r - Government income

    def _resolve_final_demand_split(
        self,
        total_flow: float,
        domestic_flow: float,
        import_flow: float,
    ) -> tuple[float, float, float]:
        """Return consistent total, domestic and import benchmark final-demand flows.

        Some GTAP extracts expose Armington totals directly in `vpm`/`vgm`/`vim`
        while others appear to store only the domestic component there and leave
        imports in `vmpp`/`vmgp`/`vmip`. When explicit domestic+import flows imply
        a materially larger total than `vpm`/`vgm`/`vim`, trust the explicit split
        and reconstruct the Armington total from its parts.
        """
        total = max(total_flow, 0.0)
        domestic = max(domestic_flow, 0.0)
        imported = max(import_flow, 0.0)

        explicit_total = domestic + imported

        # Some benchmark extracts store total Armington demand while the explicit
        # domestic split is missing. Reconstruct domestic as total minus imports.
        if domestic <= 0.0 and total > 0.0 and imported > 0.0:
            domestic = max(total - imported, 0.0)
            explicit_total = domestic + imported

        if explicit_total > 0.0:
            # Some benchmark extracts label the domestic component as the total.
            # If the explicit domestic/import split materially exceeds that value,
            # treat the split as authoritative.
            if total <= 0.0 or total < explicit_total * (1.0 - 1e-6):
                total = explicit_total

        if domestic <= 0.0 and total > 0.0:
            domestic = max(total - imported, 0.0)

        if imported <= 0.0 and total > 0.0 and domestic > 0.0:
            imported = max(total - domestic, 0.0)

        if total <= 0.0:
            total = domestic + imported

        return total, domestic, imported

    def get_private_demand(self, region: str, commodity: str) -> tuple[float, float, float]:
        return self._resolve_final_demand_split(
            self.vpm.get((region, commodity), 0.0),
            self.vdpp.get((region, commodity), 0.0),
            self.vmpp.get((region, commodity), 0.0),
        )

    def get_government_demand(self, region: str, commodity: str) -> tuple[float, float, float]:
        return self._resolve_final_demand_split(
            self.vgm.get((region, commodity), 0.0),
            self.vdgp.get((region, commodity), 0.0),
            self.vmgp.get((region, commodity), 0.0),
        )

    def get_investment_demand(self, region: str, commodity: str) -> tuple[float, float, float]:
        return self._resolve_final_demand_split(
            self.vim.get((region, commodity), 0.0),
            self.vdip.get((region, commodity), 0.0),
            self.vmip.get((region, commodity), 0.0),
        )
    
    def load_from_gdx(self, gdx_path: Path, sets: GTAPSets) -> None:
        """Load benchmark values from GDX file using GTAP Standard 7 native parameter names."""
        gdx_data = read_gdx(gdx_path)
        
        # Production - using GTAP Std 7 names
        self._load_param(gdx_data, "vom", self.vom, 2, gdx_path)        # → VOSB
        self._load_param(gdx_data, "makb", self.makb, 3, gdx_path)      # → MAKB
        self._load_param(gdx_data, "maks", self.maks, 3, gdx_path)      # → MAKS (if exists)
        
        # Factors - using GTAP Std 7 names  
        self._load_param(gdx_data, "vfm", self.vfm, 3, gdx_path)        # → EVFP
        self._load_param(gdx_data, "evfb", self.evfb, 3, gdx_path)      # → EVFB
        self._load_param(gdx_data, "evos", self.evos, 3, gdx_path)      # → EVOS
        
        # Intermediate demand - using GTAP Std 7 names
        self._load_param(gdx_data, "vdfm", self.vdfm, 3, gdx_path)      # → VDFP
        self._load_param(gdx_data, "vifm", self.vifm, 3, gdx_path)      # → VIFP
        
        # Final demand - using GTAP Std 7 names
        self._load_param(gdx_data, "vpm", self.vpm, 2, gdx_path)        # → VDPP
        self._load_param(gdx_data, "vgm", self.vgm, 2, gdx_path)        # → VDGP
        self._load_param(gdx_data, "vim", self.vim, 2, gdx_path)        # → VDIP
        
        # Trade flows - using GTAP Std 7 names
        self._load_param(gdx_data, "vxmd", self.vxmd, 3, gdx_path)      # → VXMD
        self._load_param(gdx_data, "viws", self.viws, 3, gdx_path)      # → VIWS
        self._load_param(gdx_data, "vims", self.vims, 3, gdx_path)      # → VIMS
        self._load_param(gdx_data, "vtwr", self.vtwr, 4, gdx_path)      # → VTWR

        # Additional SAM tables - GTAP Std 7 already uses uppercase
        additional = [
            ("MAKB", self.makb, 3),
            ("VDFB", self.vdfb, 3),
            ("VDFP", self.vdfp, 3),
            ("VMFB", self.vmfb, 3),
            ("VMFP", self.vmfp, 3),
            ("VDPB", self.vdpb, 2),
            ("VDPP", self.vdpp, 2),
            ("VDGB", self.vdgb, 2),
            ("VDGP", self.vdgp, 2),
            ("VMGB", self.vmgb, 2),
            ("VMGP", self.vmgp, 2),
            ("VDIB", self.vdib, 2),
            ("VDIP", self.vdip, 2),
            ("VMIB", self.vmib, 2),
            ("VMIP", self.vmip, 2),
            ("VMPB", self.vmpb, 2),
            ("VMPP", self.vmpp, 2),
            ("VFOB", self.vfob, 3),
            ("VCIF", self.vcif, 3),
            ("VMSB", self.vmsb, 3),
            ("VST", self.vst, 2),
            ("VXSB", self.vxsb, 3),
            ("SAVE", self.save, 1),
            ("VDEP", self.vdep, 1),
            ("VKB", self.vkb, 1),
            ("POP", self.pop, 1),
        ]
        for name, target, ndim in additional:
            self._load_param(gdx_data, name, target, ndim, gdx_path)

        # Align benchmark values with GAMS inScale usage (cal.gms).
        in_scale = 1e-6

        def _scale_dict(values: Dict) -> None:
            for key, val in list(values.items()):
                values[key] = float(val) * in_scale

        monetary_attrs = [
            "vom",
            "makb",
            "maks",
            "vfm",
            "vfb",
            "evfb",
            "evos",
            "vmfp",
            "vmfb",
            "vdfm",
            "vifm",
            "vdfp",
            "vdfb",
            "vmpp",
            "vmpb",
            "vdpp",
            "vdpb",
            "vdgp",
            "vdgb",
            "vmgp",
            "vmgb",
            "vdip",
            "vdib",
            "vmip",
            "vmib",
            "vpm",
            "vgm",
            "vim",
            "vxmd",
            "vswd",
            "vtwr",
            "vxsb",
            "vfob",
            "vcif",
            "vmsb",
            "vst",
            "viws",
            "vims",
            "save",
            "vdep",
            "vkb",
            "yp",
            "yg",
        ]

        for attr in monetary_attrs:
            values = getattr(self, attr, None)
            if isinstance(values, dict) and values:
                _scale_dict(values)

        # Aggregate factor benchmark values by (region, factor) for xft scaling.
        if self.evfb:
            self.vfb.clear()
            for (region, factor, _activity), value in self.evfb.items():
                key = (region, factor)
                self.vfb[key] = self.vfb.get(key, 0.0) + float(value)

        # If household domestic purchases are missing, reconstruct from totals.
        for key, total in self.vpm.items():
            if total <= 0.0:
                continue
            imported = self.vmpp.get(key, 0.0)
            domestic = self.vdpp.get(key, 0.0)
            if domestic <= 0.0 and imported > 0.0:
                self.vdpp[key] = max(total - imported, 0.0)
            if self.vdpp.get(key, 0.0) > 0.0 and self.vdpb.get(key, 0.0) <= 0.0:
                self.vdpb[key] = self.vdpp[key]
            if imported > 0.0 and self.vmpb.get(key, 0.0) <= 0.0:
                self.vmpb[key] = imported

        self._derive_output_totals(sets)
        self._derive_intermediate_totals(sets)
        self._derive_final_demand_totals(sets)
        self._derive_trade_aggregates(sets)

    def _load_param(self, gdx_data: Dict, name: str, target: Dict, ndim: int, gdx_path: Path) -> None:
        """Helper to load a parameter using GTAP Standard 7 native names.
        
        Args:
            gdx_data: GDX data dictionary
            name: Internal parameter name (e.g., 'vom', 'vfm') or GTAP Std 7 name (e.g., 'VOSB')
            target: Target dictionary to populate
            ndim: Expected number of dimensions
            gdx_path: Path to GDX file for gdxdump fallback
        """
        # Convert internal name to GTAP Std 7 name (if not already uppercase)
        gtap_name = get_benchmark_parameter_name(name) if name.islower() else name
        
        # Special handling for 'vom' - calculate from MAKB
        if name == 'vom' and gtap_name == 'MAKB':
            makb_values = read_parameter_values(gdx_data, 'MAKB')
            if makb_values:
                # MAKB has structure (commodity, activity, region)
                # vom(region, activity) = SUM_commodity MAKB(commodity, activity, region)
                makb_reordered = reorder_parameter_keys('MAKB', makb_values)  # → (r, a, i)
                for (r, a, i), val in makb_reordered.items():
                    key = (r, a)
                    target[key] = target.get(key, 0.0) + val
            return
        
        try:
            values = read_parameter_values(gdx_data, gtap_name)
            if not values:
                # Fallback to gdxdump if pure Python reader fails
                values = read_parameter_with_gdxdump(gdx_path, gtap_name)
            if values:
                # Reorder keys if needed
                values = reorder_parameter_keys(gtap_name, values)
                target.update(values)
        except (KeyError, ValueError):
            values = read_parameter_with_gdxdump(gdx_path, gtap_name)
            if values:
                # Reorder keys if needed
                values = reorder_parameter_keys(gtap_name, values)
                target.update(values)

    def _derive_output_totals(self, sets: GTAPSets) -> None:
        """Derived output totals (e.g., vom_i) from make/output pairs."""
        for (region, activity, commodity), value in self.makb.items():
            if value <= 0:
                continue
            key = (region, commodity)
            self.vom_i[key] = self.vom_i.get(key, 0.0) + value

    def _derive_intermediate_totals(self, sets: GTAPSets) -> None:
        """Placeholder for intermediate demand aggregators (no-op for now)."""
        return

    def _derive_final_demand_totals(self, sets: GTAPSets) -> None:
        """Placeholder for final demand aggregates (no-op for now)."""
        return

    def _derive_trade_aggregates(self, sets: GTAPSets) -> None:
        """Placeholder for trade aggregates (no-op for now)."""
        return

    def get_trade_totals(self, sets: GTAPSets, region: str, commodity: str) -> tuple[float, float, float, float, float]:
        """Return benchmark trade totals for standard aggregates.

        The simplified Python CET block only represents a top-level split between
        domestic sales and aggregate exports. To keep that block benchmark-feasible,
        we anchor domestic sales to absorbed domestic use and aggregate exports to
        the observed bilateral export flows. The top-level supply quantity then
        follows from that same split so the CET nest starts from an internally
        consistent benchmark.
        """
        _, private_domestic, private_import = self.get_private_demand(region, commodity)
        _, government_domestic, government_import = self.get_government_demand(region, commodity)
        _, investment_domestic, investment_import = self.get_investment_demand(region, commodity)

        domestic_use = (
            sum(self.vdfb.get((region, commodity, a), 0.0) for a in sets.a)
            + private_domestic
            + government_domestic
            + investment_domestic
        )
        xmt = sum(
            self.vifm.get((region, commodity, a), 0.0)
            for a in sets.a
        )
        xmt += private_import + government_import + investment_import
        # GAMS calibrates the export CET on VXSB (exports at basic prices),
        # not on VXMD (exports at market/fob prices). VXSB is the quantity
        # that appears in xet.l after the cal.gms benchmark normalization.
        xd = max(domestic_use, 0.0)
        xs = max(sum(self.makb.get((region, activity, commodity), 0.0) for activity in sets.a), 0.0)
        if xs <= 0.0:
            xs = max(self.vom_i.get((region, commodity), 0.0), 0.0)
        if xs <= 0.0:
            xs = max(xd, 0.0)
        # GAMS overwrites the first xet seed with the residual CET identity:
        # xet = (ps * xs - pd * xds) / pet
        # At benchmark prices this collapses to xs - xd, which is the
        # quantity that must be used to calibrate the top-level CET shares.
        xet = max(xs - xd, 0.0)
        xa = xd + xmt
        return xs, xd, xet, xmt, xa


@dataclass
class GTAPTaxRates:
    """Tax rates for GTAP model.
    
    All tax rates are expressed as fractions (e.g., 0.05 = 5% tax).
    """
    
    # Output taxes
    rto: Dict[Tuple[str, str], float] = field(default_factory=dict)     # (r, a) - Output tax rate
    
    # Factor taxes
    rtf: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, f, a) - Factor tax rate
    rtfd: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, f, a) - Factor tax on domestic
    rtfi: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, f, a) - Factor tax on imports
    
    # Consumption taxes
    rtpd: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, a) - Domestic consumption tax
    rtpi: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, a) - Import consumption tax
    rtgd: Dict[Tuple[str, str], float] = field(default_factory=dict)     # (r, i) - Government domestic tax
    rtgi: Dict[Tuple[str, str], float] = field(default_factory=dict)     # (r, i) - Government import tax
    
    # Trade taxes
    rtxs: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, rp) - Export subsidy (negative = tax)
    rtms: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, rp) - Import tariff
    imptx: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, rp) - Import tax rate (ad-valorem)
    
    # Direct taxes
    kappaf: Dict[Tuple[str, str], float] = field(default_factory=dict)   # (r, f) - Income tax rate
    kappaf_activity: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, f, a) - Activity-level wedges
    
    # Agent-specific consumption taxes (GAMS dintx0, mintx0)
    # These are calculated from benchmark data as (purchaser_price - basic_price) / (basic_price * quantity)
    dintx0: Dict[Tuple[str, str, str], float] = field(default_factory=dict)   # (r, i, aa) - Domestic consumption tax by agent
    mintx0: Dict[Tuple[str, str, str], float] = field(default_factory=dict)   # (r, i, aa) - Import consumption tax by agent
    
    def load_from_gdx(self, gdx_path: Path, sets: GTAPSets) -> None:
        """Load tax rates from GDX file."""
        gdx_data = read_gdx(gdx_path)
        
        self._load_param(gdx_data, "rto", self.rto, 2, gdx_path)
        self._load_param(gdx_data, "rtf", self.rtf, 3, gdx_path)
        self._load_param(gdx_data, "rtfd", self.rtfd, 3, gdx_path)
        self._load_param(gdx_data, "rtfi", self.rtfi, 3, gdx_path)
        self._load_param(gdx_data, "rtpd", self.rtpd, 3, gdx_path)
        self._load_param(gdx_data, "rtpi", self.rtpi, 3, gdx_path)
        self._load_param(gdx_data, "rtgd", self.rtgd, 2, gdx_path)
        self._load_param(gdx_data, "rtgi", self.rtgi, 2, gdx_path)
        self._load_param(gdx_data, "rtxs", self.rtxs, 3, gdx_path)
        self._load_param(gdx_data, "rtms", self.rtms, 3, gdx_path)

        # Align tax revenue flows with GAMS inScale usage before rate derivation.
        in_scale = 1e-6

        def _scale_dict(values: Dict) -> None:
            for key, val in list(values.items()):
                values[key] = float(val) * in_scale

        for attr in ("rto", "rtf", "rtfd", "rtpd", "rtpi", "rtgd", "rtgi", "rtxs", "rtms"):
            values = getattr(self, attr, None)
            if isinstance(values, dict) and values:
                _scale_dict(values)
        # Normalize import tax rates to ad-valorem form used in equations.
        if self.rtms:
            self.imptx = {k: float(v) / 1000.0 for k, v in self.rtms.items()}
        
    def _load_param(self, gdx_data: Dict, name: str, target: Dict, ndim: int, gdx_path: Path) -> None:
        """Helper to load a tax parameter using GTAP Standard 7 native names.
        
        Args:
            gdx_data: GDX data dictionary
            name: Internal tax parameter name (e.g., 'rto', 'rtf')
            target: Target dictionary to populate
            ndim: Expected number of dimensions
            gdx_path: Path to GDX file for gdxdump fallback
        """
        # Convert internal name to GTAP Std 7 name
        gtap_name = get_tax_parameter_name(name)
        
        try:
            values = read_parameter_values(gdx_data, gtap_name)
            if values:
                # Reorder keys if needed
                values = reorder_parameter_keys(gtap_name, values)
                target.update(values)
        except (KeyError, ValueError):
            values = read_parameter_with_gdxdump(gdx_path, gtap_name)
            if values:
                # Reorder keys if needed
                values = reorder_parameter_keys(gtap_name, values)
                target.update(values)

    def derive_agent_consumption_taxes(self, benchmark: "GTAPBenchmarkValues", sets: GTAPSets) -> None:
        """Derive agent-level consumption tax rates (GAMS dintx0, mintx0).
        
        These taxes are calculated from benchmark data as:
            dintx = (purchaser_price - basic_price) / basic_price = (vdfp - vdfb) / vdfb
            mintx = (purchaser_price - basic_price) / basic_price = (vmfp - vmfb) / vmfb
            
        For activities (firms):
            dintx(r,i,a) = (vdfp - vdfb) / (pd * xd) where pd=1, xd=vdfb
            mintx(r,i,a) = (vmfp - vmfb) / (pmt * xm) where pmt=1, xm=vmfb
            
        For final demand agents (gov, priv, inv):
            Similar logic using vdgp/vdgb, vdpp/vdpb, vdip/vdib
        """
        # Activity-level taxes (firms)
        for r in sets.r:
            for i in sets.i:
                for a in sets.a:
                    # Domestic tax on intermediate demand
                    vdfp_val = benchmark.vdfp.get((r, i, a), 0.0)
                    vdfb_val = benchmark.vdfb.get((r, i, a), 0.0)
                    if vdfb_val > 0.0:
                        dintx_rate = (vdfp_val - vdfb_val) / vdfb_val
                        if abs(dintx_rate) > 1e-10:
                            self.dintx0[(r, i, a)] = dintx_rate
                    
                    # Import tax on intermediate demand  
                    vmfp_val = benchmark.vmfp.get((r, i, a), 0.0)
                    vmfb_val = benchmark.vmfb.get((r, i, a), 0.0)
                    if vmfb_val > 0.0:
                        mintx_rate = (vmfp_val - vmfb_val) / vmfb_val
                        if abs(mintx_rate) > 1e-10:
                            self.mintx0[(r, i, a)] = mintx_rate
                            
        # Private household consumption taxes
        for r in sets.r:
            for i in sets.i:
                vdpp_val = benchmark.vdpp.get((r, i), 0.0) if hasattr(benchmark, 'vdpp') else 0.0
                vdpb_val = benchmark.vdpb.get((r, i), 0.0) if hasattr(benchmark, 'vdpb') else 0.0
                vmpp_val = benchmark.vmpp.get((r, i), 0.0) if hasattr(benchmark, 'vmpp') else 0.0
                vmpb_val = benchmark.vmpb.get((r, i), 0.0) if hasattr(benchmark, 'vmpb') else 0.0
                
                if vdpb_val > 0.0:
                    dintx_rate = (vdpp_val - vdpb_val) / vdpb_val
                    if abs(dintx_rate) > 1e-10:
                        self.dintx0[(r, i, "priv")] = dintx_rate
                        
                if vmpb_val > 0.0:
                    mintx_rate = (vmpp_val - vmpb_val) / vmpb_val
                    if abs(mintx_rate) > 1e-10:
                        self.mintx0[(r, i, "priv")] = mintx_rate
                        
        # Government consumption taxes
        for r in sets.r:
            for i in sets.i:
                vdgp_val = benchmark.vdgp.get((r, i), 0.0) if hasattr(benchmark, 'vdgp') else 0.0
                vdgb_val = benchmark.vdgb.get((r, i), 0.0) if hasattr(benchmark, 'vdgb') else 0.0
                vmgp_val = benchmark.vmgp.get((r, i), 0.0) if hasattr(benchmark, 'vmgp') else 0.0
                vmgb_val = benchmark.vmgb.get((r, i), 0.0) if hasattr(benchmark, 'vmgb') else 0.0
                
                if vdgb_val > 0.0:
                    dintx_rate = (vdgp_val - vdgb_val) / vdgb_val
                    if abs(dintx_rate) > 1e-10:
                        self.dintx0[(r, i, "gov")] = dintx_rate
                        
                if vmgb_val > 0.0:
                    mintx_rate = (vmgp_val - vmgb_val) / vmgb_val
                    if abs(mintx_rate) > 1e-10:
                        self.mintx0[(r, i, "gov")] = mintx_rate
                        
        # Investment consumption taxes
        for r in sets.r:
            for i in sets.i:
                vdip_val = benchmark.vdip.get((r, i), 0.0) if hasattr(benchmark, 'vdip') else 0.0
                vdib_val = benchmark.vdib.get((r, i), 0.0) if hasattr(benchmark, 'vdib') else 0.0
                vmip_val = benchmark.vmip.get((r, i), 0.0) if hasattr(benchmark, 'vmip') else 0.0
                vmib_val = benchmark.vmib.get((r, i), 0.0) if hasattr(benchmark, 'vmib') else 0.0
                
                if vdib_val > 0.0:
                    dintx_rate = (vdip_val - vdib_val) / vdib_val
                    if abs(dintx_rate) > 1e-10:
                        self.dintx0[(r, i, "inv")] = dintx_rate
                        
                if vmib_val > 0.0:
                    mintx_rate = (vmip_val - vmib_val) / vmib_val
                    if abs(mintx_rate) > 1e-10:
                        self.mintx0[(r, i, "inv")] = mintx_rate

    def derive_trade_route_wedges(self, benchmark: "GTAPBenchmarkValues", sets: GTAPSets) -> None:
        """Derive trade route wedges (stub)."""
        return

    def derive_from_benchmark(self, benchmark: "GTAPBenchmarkValues", sets: GTAPSets) -> None:
        """Derive ad-valorem tax rates from benchmark SAM flows (GAMS-style)."""
        raw_fbep = dict(self.rtf)
        raw_ftrv = dict(self.rtfd)

        rto_rates: Dict[Tuple[str, str], float] = {}
        for r in sets.r:
            for a in sets.a:
                outputs = sets.activity_commodities.get(a, [])
                if not outputs:
                    outputs = list(sets.i)
                weighted = 0.0
                denom = 0.0
                for i in outputs:
                    makb = float(benchmark.makb.get((r, a, i), 0.0))
                    maks = float(benchmark.maks.get((r, a, i), 0.0))
                    if makb <= 0.0 or maks <= 0.0:
                        continue
                    rate = (makb / maks) - 1.0
                    weighted += rate * makb
                    denom += makb
                if denom > 0.0:
                    rto_rates[(r, a)] = weighted / denom
        if rto_rates:
            self.rto = rto_rates

        rtf_rates: Dict[Tuple[str, str, str], float] = {}
        for (r, f, a), vfm in benchmark.vfm.items():
            evfb = float(benchmark.evfb.get((r, f, a), 0.0) or 0.0)
            denom = evfb if evfb > 0.0 else float(vfm)
            if denom <= 0.0:
                continue
            fbep = float(raw_fbep.get((r, f, a), 0.0))
            ftrv = float(raw_ftrv.get((r, f, a), 0.0))
            # GTAP benchmark-consistent wedge (matches GAMS pfa benchmark levels):
            # rtf = (FBEP + FTRV) / EVFB
            rate = (ftrv + fbep) / denom
            if rate != 0.0 or fbep != 0.0 or ftrv != 0.0:
                rtf_rates[(r, f, a)] = rate
        if rtf_rates:
            self.rtf = rtf_rates

        # GAMS cal.gms:
        #   kappaf = (EVFB - EVOS) / EVFB   when EVFB > 0
        kappaf_activity_rates: Dict[Tuple[str, str, str], float] = {}
        kappaf_aggregate_num: Dict[Tuple[str, str], float] = defaultdict(float)
        kappaf_aggregate_den: Dict[Tuple[str, str], float] = defaultdict(float)
        for (r, f, a), evfb in benchmark.evfb.items():
            evfb_val = float(evfb)
            if evfb_val <= 0.0:
                continue
            evos_val = float(benchmark.evos.get((r, f, a), 0.0))
            rate = (evfb_val - evos_val) / evfb_val
            if abs(rate) > 1e-12:
                kappaf_activity_rates[(r, f, a)] = rate
            kappaf_aggregate_num[(r, f)] += (evfb_val - evos_val)
            kappaf_aggregate_den[(r, f)] += evfb_val
        if kappaf_activity_rates:
            self.kappaf_activity = kappaf_activity_rates
        kappaf_rates: Dict[Tuple[str, str], float] = {}
        for key, den in kappaf_aggregate_den.items():
            if den <= 0.0:
                continue
            rate = kappaf_aggregate_num[key] / den
            if abs(rate) > 1e-12:
                kappaf_rates[key] = rate
        if kappaf_rates:
            self.kappaf = kappaf_rates

        rtpd_rates: Dict[Tuple[str, str, str], float] = {}
        rtpi_rates: Dict[Tuple[str, str, str], float] = {}
        for r in sets.r:
            for i in sets.i:
                for a in sets.a:
                    vdfb = float(benchmark.vdfb.get((r, i, a), 0.0))
                    if vdfb > 0.0:
                        vdfp = float(benchmark.vdfp.get((r, i, a), 0.0))
                        rtpd_rates[(r, i, a)] = (vdfp - vdfb) / vdfb
                    vmfb = float(benchmark.vmfb.get((r, i, a), 0.0))
                    if vmfb > 0.0:
                        vmfp = float(benchmark.vmfp.get((r, i, a), 0.0))
                        rtpi_rates[(r, i, a)] = (vmfp - vmfb) / vmfb
        if rtpd_rates:
            self.rtpd = rtpd_rates
        if rtpi_rates:
            self.rtpi = rtpi_rates

        rtgd_rates: Dict[Tuple[str, str], float] = {}
        rtgi_rates: Dict[Tuple[str, str], float] = {}
        for r in sets.r:
            for i in sets.i:
                vdgb = float(benchmark.vdgb.get((r, i), 0.0))
                if vdgb > 0.0:
                    vdgp = float(benchmark.vdgp.get((r, i), 0.0))
                    rtgd_rates[(r, i)] = (vdgp - vdgb) / vdgb
                vmgb = float(benchmark.vmgb.get((r, i), 0.0))
                if vmgb > 0.0:
                    vmgp = float(benchmark.vmgp.get((r, i), 0.0))
                    rtgi_rates[(r, i)] = (vmgp - vmgb) / vmgb
        if rtgd_rates:
            self.rtgd = rtgd_rates
        if rtgi_rates:
            self.rtgi = rtgi_rates

        rtxs_rates: Dict[Tuple[str, str, str], float] = {}
        for (r, i, rp), vxsb in benchmark.vxsb.items():
            if vxsb <= 0.0:
                continue
            vfob = float(benchmark.vfob.get((r, i, rp), 0.0))
            rtxs_rates[(r, i, rp)] = (vfob - vxsb) / float(vxsb)
        if rtxs_rates:
            self.rtxs = rtxs_rates

        imptx_rates: Dict[Tuple[str, str, str], float] = {}
        for (r, i, rp), vcif in benchmark.vcif.items():
            if vcif <= 0.0:
                continue
            vmsb = float(benchmark.vmsb.get((r, i, rp), 0.0))
            imptx_rates[(r, i, rp)] = (vmsb - vcif) / float(vcif)
        if imptx_rates:
            self.imptx = imptx_rates

        # Derive agent-specific consumption taxes (GAMS dintx0, mintx0)
        self.derive_agent_consumption_taxes(benchmark, sets)


@dataclass
class GTAPNormalizedParameters:
    """Snapshot of normalized shares derived from the benchmark."""

    value_added_share: Dict[Tuple[str, str], float] = field(default_factory=dict)
    intermediate_share: Dict[Tuple[str, str], float] = field(default_factory=dict)
    output_share: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    commodity_share: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    factor_supply_share: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    factor_value_share: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    armington_domestic: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    armington_import: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    armington_national: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    import_source_share: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    export_destination_share: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    domestic_supply_share: Dict[Tuple[str, str], float] = field(default_factory=dict)
    export_supply_share: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def update_from_shares(self, share_params: "GTAPShareParameters") -> None:
        self.value_added_share = share_params.p_va.copy()
        self.intermediate_share = share_params.p_nd.copy()
        self.output_share = share_params.p_gx.copy()
        self.commodity_share = share_params.p_ax.copy()
        self.factor_supply_share = share_params.p_gf.copy()
        self.factor_value_share = share_params.p_af.copy()
        self.armington_domestic = share_params.p_alphad.copy()
        self.armington_import = share_params.p_alpham.copy()
        self.armington_national = share_params.p_alphan.copy()
        self.import_source_share = share_params.p_amw.copy()
        self.export_destination_share = share_params.p_gw.copy()
        self.domestic_supply_share = share_params.p_gd.copy()
        self.export_supply_share = share_params.p_ge.copy()


@dataclass
class GTAPShareParameters:
    """Share parameters from calibration.
    
    These are derived from the benchmark SAM and elasticities.
    """
    
    normalized: GTAPNormalizedParameters = field(default_factory=GTAPNormalizedParameters, init=False)

    # Production shares
    p_gx: Dict[Tuple[str, str, str], float] = field(default_factory=dict)     # (r, a, i) - CET share for output allocation
    p_ax: Dict[Tuple[str, str, str], float] = field(default_factory=dict)     # (r, a, i) - CES share for commodity production
    p_va: Dict[Tuple[str, str], float] = field(default_factory=dict)          # (r, a) - Value-added share
    p_nd: Dict[Tuple[str, str], float] = field(default_factory=dict)          # (r, a) - Intermediate-share complement
    p_io: Dict[Tuple[str, str, str], float] = field(default_factory=dict)     # (r, i, a) - Intermediate demand share inside nd
    
    # Armington shares
    p_alphad: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, aa) - Domestic share
    p_alpham: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, aa) - Import share (top nest)
    p_alphan: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, aa) - National share
    
    # Bilateral trade shares
    p_amw: Dict[Tuple[str, str, str], float] = field(default_factory=dict)    # (r, i, rp) - Import share by source
    p_gw: Dict[Tuple[str, str, str], float] = field(default_factory=dict)     # (r, i, rp) - Export share by destination
    p_gd: Dict[Tuple[str, str], float] = field(default_factory=dict)          # (r, i) - Domestic CET share
    p_ge: Dict[Tuple[str, str], float] = field(default_factory=dict)          # (r, i) - Export CET share
    
    # Factor shares
    p_gf: Dict[Tuple[str, str, str], float] = field(default_factory=dict)     # (r, f, a) - Factor supply share
    p_af: Dict[Tuple[str, str, str], float] = field(default_factory=dict)     # (r, f, a) - Factor share inside value added
    
    # Demand shares
    p_axg: Dict[str, float] = field(default_factory=dict)     # r - Government CES shifter
    p_axi: Dict[str, float] = field(default_factory=dict)     # r - Investment CES shifter
    
    def calibrate(self, benchmark: GTAPBenchmarkValues, elasticities: GTAPElasticities, 
                  sets: GTAPSets) -> None:
        """Calibrate all share parameters from benchmark data."""
        self._calibrate_production_shares(benchmark, sets)
        self._calibrate_armington_shares(benchmark, sets)
        self._calibrate_trade_shares(benchmark, elasticities, sets)
        self._calibrate_factor_shares(benchmark, sets)
        self.normalized.update_from_shares(self)

    def _calibrate_production_shares(self, benchmark: GTAPBenchmarkValues, sets: GTAPSets) -> None:
        """Calibrate production share parameters and VA/ND ratios."""
        for r in sets.r:
            for a in sets.a:
                outputs = sets.activity_commodities.get(a, [])
                if not outputs:
                    if a in sets.a_to_i:
                        outputs = [sets.a_to_i[a]]
                if not outputs:
                    continue

                total_output = sum(benchmark.makb.get((r, a, i), 0.0) for i in outputs)
                if total_output <= 0:
                    total_output = benchmark.vom.get((r, a), 0.0)
                if total_output <= 0:
                    continue

                for i in outputs:
                    value = benchmark.makb.get((r, a, i), 0.0)
                    share = value / total_output if value > 0 else 1.0 / len(outputs)
                    self.p_gx[(r, a, i)] = share

                total_va = sum(
                    benchmark.evfb.get((r, f, a), benchmark.vfm.get((r, f, a), 0.0))
                    for f in sets.f
                )
                if total_va > 0:
                    self.p_va[(r, a)] = min(total_va / total_output, 1.0)
                    for f in sets.f:
                        value = benchmark.evfb.get((r, f, a), benchmark.vfm.get((r, f, a), 0.0))
                        self.p_af[(r, f, a)] = value / total_va if value > 0 else 0.0
                else:
                    self.p_va[(r, a)] = 0.0
                    for f in sets.f:
                        self.p_af[(r, f, a)] = 0.0

                self.p_nd[(r, a)] = max(0.0, 1.0 - self.p_va.get((r, a), 0.0))

                total_intermediate = 0.0
                intermediate_by_commodity: Dict[str, float] = {}
                for i in sets.i:
                    value = benchmark.vdfm.get((r, i, a), 0.0) + benchmark.vifm.get((r, i, a), 0.0)
                    if value <= 0.0:
                        continue
                    intermediate_by_commodity[i] = value
                    total_intermediate += value

                if total_intermediate > 0.0:
                    for i, value in intermediate_by_commodity.items():
                        self.p_io[(r, i, a)] = value / total_intermediate
                else:
                    for i in sets.i:
                        self.p_io[(r, i, a)] = 0.0

        for r in sets.r:
            for i in sets.i:
                activities = sets.commodity_activities.get(i, [])
                if not activities and i in sets.i_to_a:
                    activities = [sets.i_to_a[i]]
                if not activities:
                    continue

                total_output = sum(benchmark.makb.get((r, a, i), 0.0) for a in activities)
                if total_output <= 0.0:
                    positive = [a for a in activities if benchmark.vom.get((r, a), 0.0) > 0.0]
                    if len(positive) == 1:
                        self.p_ax[(r, positive[0], i)] = 1.0
                    continue

                for a in activities:
                    value = benchmark.makb.get((r, a, i), 0.0)
                    self.p_ax[(r, a, i)] = value / total_output if value > 0.0 else 0.0
                
    def _calibrate_armington_shares(self, benchmark: GTAPBenchmarkValues, sets: GTAPSets) -> None:
        """Calibrate Armington share parameters."""
        for r in sets.r:
            for i in sets.i:
                _, xd, _, xmt, xa = benchmark.get_trade_totals(sets, r, i)
                if xa <= 0:
                    self.p_alphad[(r, i)] = 1.0
                    self.p_alpham[(r, i)] = 0.0
                    continue
                self.p_alphad[(r, i)] = xd / xa
                self.p_alpham[(r, i)] = xmt / xa

    def _calibrate_trade_shares(
        self,
        benchmark: GTAPBenchmarkValues,
        elasticities: GTAPElasticities,
        sets: GTAPSets,
    ) -> None:
        """Calibrate bilateral trade share parameters."""
        for r in sets.r:
            for i in sets.i:
                xs = sum(benchmark.makb.get((r, a, i), 0.0) for a in sets.a)
                if xs <= 0.0:
                    xs = benchmark.vom_i.get((r, i), 0.0)
                # GAMS calibrates export-side nests over xwFlag support and
                # does not enforce rp != r at equation-domain level.
                xet = sum(float(benchmark.vxsb.get((r, i, rp), 0.0) or 0.0) for rp in sets.r)
                xds = max(xs - xet, 0.0)
                if xs > 0:
                    self.p_gd[(r, i)] = xds / xs
                    self.p_ge[(r, i)] = xet / xs
                else:
                    self.p_gd[(r, i)] = 1.0
                    self.p_ge[(r, i)] = 0.0

        for r in sets.r:
            for i in sets.i:
                # Keep bilateral source support consistent with GAMS xwFlag.
                xmt = sum(float(benchmark.vmsb.get((rp, i, r), 0.0) or 0.0) for rp in sets.r)
                if xmt <= 0:
                    continue
                sigmaw = elasticities.esubm.get((r, i), 5.0)
                for rp in sets.r:
                    # GAMS calibration uses xw.l initialized from VXSB/pe.
                    benchmark_xw = float(benchmark.vxsb.get((rp, i, r), 0.0) or 0.0)
                    if benchmark_xw <= 0.0:
                        continue
                    xw = benchmark_xw
                    vcif = float(benchmark.vcif.get((rp, i, r), 0.0) or 0.0)
                    pmcif = (vcif / xw) if (vcif > 0.0 and xw > 0.0) else 1.0
                    vmsb = float(benchmark.vmsb.get((rp, i, r), 0.0) or 0.0)
                    imptx = ((vmsb - vcif) / max(pmcif * xw, 1e-12)) if vmsb > 0.0 else 0.0
                    pm = max((1.0 + imptx) * pmcif, 1e-8)
                    self.p_amw[(r, i, rp)] = (xw / xmt) * (pm ** sigmaw)

        for r in sets.r:
            for i in sets.i:
                xet = sum(float(benchmark.vxsb.get((r, i, rp), 0.0) or 0.0) for rp in sets.r)
                if xet <= 0.0:
                    continue
                for rp in sets.r:
                    xw = float(benchmark.vxsb.get((r, i, rp), 0.0) or 0.0)
                    if xw > 0.0:
                        self.p_gw[(r, i, rp)] = xw / xet
                            
    def _calibrate_factor_shares(self, benchmark: GTAPBenchmarkValues, sets: GTAPSets) -> None:
        """Calibrate factor share parameters."""
        for r in sets.r:
            for f in sets.f:
                total_payment = sum(
                    benchmark.evfb.get((r, f, a), benchmark.vfm.get((r, f, a), 0.0))
                    for a in sets.a
                )
                if total_payment > 0:
                    for a in sets.a:
                        value = benchmark.evfb.get((r, f, a), benchmark.vfm.get((r, f, a), 0.0))
                        self.p_gf[(r, f, a)] = value / total_payment

    def apply_equilibrium_snapshot(self, snapshot: GTAPEquilibriumSnapshot, sets: GTAPSets) -> None:
        """Update share parameters using an equilibrium CSV snapshot."""
        xp = snapshot.get("xp")
        va = snapshot.get("va")
        x = snapshot.get("x")
        xw = snapshot.get("xw")
        xf = snapshot.get("xf")
        pf = snapshot.get("pf")

        def iter_ra(data):
            for key, value in data.items():
                if len(key) < 2:
                    continue
                yield key[0], key[1], value

        def iter_rai(data):
            for key, value in data.items():
                if len(key) < 3:
                    continue
                yield key[0], key[1], key[2], value

        def iter_raf(data):
            for key, value in data.items():
                if len(key) < 3:
                    continue
                yield key[0], key[1], key[2], value

        for r, a, xp_val in iter_ra(xp):
            if r not in sets.r or a not in sets.a or xp_val <= 0:
                continue
            va_val = va.get((r, a), 0.0)
            share = max(0.0, min(1.0, va_val / xp_val))
            self.p_va[(r, a)] = share
            self.p_nd[(r, a)] = max(0.0, 1.0 - share)

        for r, a, i, value in iter_rai(x):
            if r not in sets.r or a not in sets.a or i not in sets.activity_commodities.get(a, [i]) + [i]:
                continue
            xp_val = xp.get((r, a), 0.0)
            share = value / xp_val if xp_val > 0 else 0.0
            self.p_gx[(r, a, i)] = share

        for r in sets.r:
            for i in sets.i:
                activities = sets.commodity_activities.get(i, [])
                if not activities and i in sets.i_to_a:
                    activities = [sets.i_to_a[i]]
                if not activities:
                    continue

                total = sum(x.get((r, a, i), 0.0) for a in activities)
                if total <= 0.0:
                    positive = [a for a in activities if xp.get((r, a), 0.0) > 0.0]
                    if len(positive) == 1:
                        self.p_ax[(r, positive[0], i)] = 1.0
                    continue

                for a in activities:
                    value = x.get((r, a, i), 0.0)
                    self.p_ax[(r, a, i)] = value / total if value > 0.0 else 0.0

        factor_totals: Dict[Tuple[str, str], float] = defaultdict(float)
        for r, a, f, xf_val in iter_raf(xf):
            if r not in sets.r or a not in sets.a or f not in sets.f:
                continue
            factor_totals[(r, f)] += xf_val

        for r, a, f, xf_val in iter_raf(xf):
            if r not in sets.r or a not in sets.a or f not in sets.f:
                continue
            total = factor_totals.get((r, f), 0.0)
            share_supply = xf_val / total if total > 0 else 0.0
            self.p_gf[(r, f, a)] = share_supply

        activity_payments: Dict[Tuple[str, str], float] = defaultdict(float)
        factor_payments: Dict[Tuple[str, str, str], float] = {}
        for r, a, f, xf_val in iter_raf(xf):
            if r not in sets.r or a not in sets.a or f not in sets.f:
                continue
            key = (r, a, f)
            price = pf.get(key, None)
            if price is None:
                continue
            payment = xf_val * price
            factor_payments[key] = payment
            activity_payments[(r, a)] += payment

        for (r, a, f), payment in factor_payments.items():
            total = activity_payments.get((r, a), 0.0)
            share_value = payment / total if total > 0 else 0.0
            self.p_af[(r, f, a)] = share_value

        import_totals: Dict[Tuple[str, str], float] = defaultdict(float)
        for exporter, commodity, importer, flow in iter_rai(xw):
            if exporter not in sets.r or importer not in sets.r or commodity not in sets.i:
                continue
            import_totals[(importer, commodity)] += flow

        for exporter, commodity, importer, flow in iter_rai(xw):
            if exporter not in sets.r or importer not in sets.r or commodity not in sets.i:
                continue
            total = import_totals.get((importer, commodity), 0.0)
            share = flow / total if total > 0 else 0.0
            self.p_amw[(importer, commodity, exporter)] = share

        self.normalized.update_from_shares(self)


@dataclass
class GTAPShiftParameters:
    """Variables that act as multipliers/technology shifters."""

    axp: Dict[Tuple[str, str], float] = field(default_factory=dict)
    lambdand: Dict[Tuple[str, str], float] = field(default_factory=dict)
    lambdava: Dict[Tuple[str, str], float] = field(default_factory=dict)
    lambdaio: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    lambdaf: Dict[Tuple[str, str, str], float] = field(default_factory=dict)

    def load_from_gdx(self, gdx_path: Path) -> None:
        """Load technology shifters from the GDX."""
        self.axp = self._collapse(read_variable_levels_with_gdxdump(gdx_path, "axp"))
        self.lambdand = self._collapse(read_variable_levels_with_gdxdump(gdx_path, "lambdand"))
        self.lambdava = self._collapse(read_variable_levels_with_gdxdump(gdx_path, "lambdava"))
        self.lambdaio = self._collapse_io(read_variable_levels_with_gdxdump(gdx_path, "lambdaio"))
        self.lambdaf = self._collapse_factor(read_variable_levels_with_gdxdump(gdx_path, "lambdaf"))

    def _collapse(self, raw: Dict[Tuple[str, ...], float]) -> Dict[Tuple[str, str], float]:
        grouped: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
        for key, value in raw.items():
            if len(key) < 2:
                continue
            region, activity = key[0], key[1]
            time = key[2] if len(key) > 2 else ""
            grouped[(region, activity)][time] = value

        result: Dict[Tuple[str, str], float] = {}
        for node, time_map in grouped.items():
            for preferred in ("base", "t0"):
                if preferred in time_map:
                    result[node] = time_map[preferred]
                    break
            else:
                result[node] = next(iter(time_map.values()))
        return result

    def _collapse_factor(self, raw: Dict[Tuple[str, ...], float]) -> Dict[Tuple[str, str, str], float]:
        grouped: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(dict)
        for key, value in raw.items():
            if len(key) < 3:
                continue
            region, factor, activity = key[0], key[1], key[2]
            time = key[3] if len(key) > 3 else ""
            grouped[(region, factor, activity)][time] = value

        result: Dict[Tuple[str, str, str], float] = {}
        for node, time_map in grouped.items():
            for preferred in ("base", "t0"):
                if preferred in time_map:
                    result[node] = time_map[preferred]
                    break
            else:
                result[node] = next(iter(time_map.values()))
        return result

    def _collapse_io(self, raw: Dict[Tuple[str, ...], float]) -> Dict[Tuple[str, str, str], float]:
        grouped: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(dict)
        for key, value in raw.items():
            if len(key) < 3:
                continue
            region, commodity, activity = key[0], key[1], key[2]
            time = key[3] if len(key) > 3 else ""
            grouped[(region, commodity, activity)][time] = value

        result: Dict[Tuple[str, str, str], float] = {}
        for node, time_map in grouped.items():
            for preferred in ("base", "t0"):
                if preferred in time_map:
                    result[node] = time_map[preferred]
                    break
            else:
                result[node] = next(iter(time_map.values()))
        return result


@dataclass
class GAMSCalibrationDump:
    """Container for calibration/benchmark symbols exported from a GAMS dump GDX.

    The dump is used as a parity reference and optional override source. Symbol keys
    are normalized to lowercase and any trailing time dimension
    (`base`, `t0`, `check`, `shock`) is collapsed to a single benchmark slice.
    """

    source_gdx: Path
    derived_params: Dict[str, Dict[Tuple[str, ...], float]] = field(default_factory=dict)
    benchmark_levels: Dict[str, Dict[Tuple[str, ...], float]] = field(default_factory=dict)

    DEFAULT_DERIVED_SYMBOLS: ClassVar[Tuple[str, ...]] = (
        "and",
        "ava",
        "io",
        "af",
        "gx",
        "gf",
        "gw",
        "ge",
        "gd",
        "amw",
        "alphaa",
        "alphad",
        "alpham",
        "alphan",
        "tmarg",
        "chipm",
        "chipd",
        "kappaf",
        "omegaf",
        "etaff",
        "esubt",
        "esubc",
        "esubm",
        "esubva",
        "sigmas",
        "omegaw",
        "axp",
        "aft",
        "aa",
        "lambdam",
        "lambdamg",
        "mtax",
        "etax",
    )

    DEFAULT_LEVEL_SYMBOLS: ClassVar[Tuple[str, ...]] = (
        "xf",
        "xft",
        "xc",
        "xa",
        "xaa",
        "xda",
        "xma",
        "xm",
        "xd",
        "xmt",
        "xw",
        "xe",
        "xet",
        "xp",
        "va",
        "nd",
        "pf",
        "pfa",
        "pfy",
        "pft",
        "pa",
        "pd",
        "pm",
        "pe",
        "pet",
        "px",
        "pva",
        "pnd",
        "ps",
        "pmcif",
        "pefob",
        "pwmg",
        "ptmg",
        "yi",
        "yc",
        "yg",
        "kstock",
        "arent",
        "ytax",
        "etax",
        "mtax",
        "dintx",
        "mintx",
    )

    _TIME_LABELS: ClassVar[Tuple[str, ...]] = ("base", "t0", "check", "shock")

    @classmethod
    def from_gdx(
        cls,
        path: Path,
        *,
        derived_symbols: Optional[Sequence[str]] = None,
        level_symbols: Optional[Sequence[str]] = None,
    ) -> "GAMSCalibrationDump":
        gdx_path = Path(path).expanduser().resolve()
        if not gdx_path.exists():
            raise FileNotFoundError(f"GAMS calibration dump not found: {gdx_path}")

        gdx_data = read_gdx(gdx_path)
        derived: Dict[str, Dict[Tuple[str, ...], float]] = {}
        levels: Dict[str, Dict[Tuple[str, ...], float]] = {}

        for symbol in tuple(derived_symbols or cls.DEFAULT_DERIVED_SYMBOLS):
            values = cls._read_parameter_symbol(gdx_data, gdx_path, symbol)
            if values:
                derived[symbol.lower()] = cls._collapse_time_dimension(values)

        for symbol in tuple(level_symbols or cls.DEFAULT_LEVEL_SYMBOLS):
            values = read_variable_levels_with_gdxdump(gdx_path, symbol)
            if values:
                levels[symbol.lower()] = cls._collapse_time_dimension(values)

        return cls(source_gdx=gdx_path, derived_params=derived, benchmark_levels=levels)

    def get_derived(self, name: str) -> Dict[Tuple[str, ...], float]:
        return self.derived_params.get(str(name).lower(), {})

    def get_levels(self, name: str) -> Dict[Tuple[str, ...], float]:
        return self.benchmark_levels.get(str(name).lower(), {})

    @classmethod
    def _read_parameter_symbol(
        cls,
        gdx_data: Dict[str, Any],
        gdx_path: Path,
        symbol: str,
    ) -> Dict[Tuple[str, ...], float]:
        candidates = [symbol, symbol.upper()]
        parsed: Dict[Tuple[str, ...], float] = {}
        matched_name = symbol

        for candidate in candidates:
            try:
                parsed = read_parameter_values(gdx_data, candidate)
            except (KeyError, ValueError):
                parsed = {}
            if parsed:
                matched_name = candidate
                break

        if not parsed:
            parsed = read_parameter_with_gdxdump(gdx_path, symbol)
            matched_name = symbol.upper()

        if not parsed:
            return {}

        try:
            parsed = reorder_parameter_keys(matched_name.upper(), parsed)
        except Exception:
            # Some dump symbols are not in the GTAP reorder map.
            pass

        normalized: Dict[Tuple[str, ...], float] = {}
        for key, value in parsed.items():
            key_tuple = tuple(key) if isinstance(key, tuple) else (str(key),)
            try:
                normalized[key_tuple] = float(value)
            except (TypeError, ValueError):
                continue
        return normalized

    @classmethod
    def _collapse_time_dimension(
        cls,
        values: Dict[Tuple[str, ...], float],
    ) -> Dict[Tuple[str, ...], float]:
        """Collapse trailing time dimension when present.

        If a key ends with a known time label, values are grouped by the prefix and
        a preferred label is selected in order: base, t0, check, shock.
        """
        grouped: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(dict)

        for raw_key, raw_value in values.items():
            key = tuple(raw_key) if isinstance(raw_key, tuple) else (str(raw_key),)
            if key and str(key[-1]).lower() in cls._TIME_LABELS:
                core = tuple(str(part) for part in key[:-1])
                time_label = str(key[-1]).lower()
            else:
                core = tuple(str(part) for part in key)
                time_label = ""

            grouped[core][time_label] = float(raw_value)

        collapsed: Dict[Tuple[str, ...], float] = {}
        for core_key, by_time in grouped.items():
            selected: Optional[float] = None
            for preferred in cls._TIME_LABELS:
                if preferred in by_time:
                    selected = by_time[preferred]
                    break
            if selected is None:
                selected = next(iter(by_time.values()))
            collapsed[core_key] = float(selected)
        return collapsed


@dataclass
class GTAPParameters:
    """Complete GTAP parameters container.
    
    This combines all parameter types for the GTAP model.
    """
    
    sets: GTAPSets = field(default_factory=GTAPSets)
    elasticities: GTAPElasticities = field(default_factory=GTAPElasticities)
    benchmark: GTAPBenchmarkValues = field(default_factory=GTAPBenchmarkValues)
    taxes: GTAPTaxRates = field(default_factory=GTAPTaxRates)
    shares: GTAPShareParameters = field(default_factory=GTAPShareParameters)
    calibrated: GTAPCalibratedShares = field(default_factory=GTAPCalibratedShares)  # GAMS-style calibrated shares
    shifts: GTAPShiftParameters = field(default_factory=GTAPShiftParameters)
    
    def load_from_gdx(
        self,
        gdx_path: Path,
        elasticity_gdx: Optional[Path] = None,
        elasticity_override_gdx: Optional[Path] = None,
    ) -> None:
        """Load all parameters from GDX file(s).
        
        Args:
            gdx_path: Path to GTAP GDX file with benchmark data (e.g., basedata-9x10.gdx)
            elasticity_gdx: Optional separate GDX file with elasticities (e.g., default-9x10.gdx)
            elasticity_override_gdx: Optional GDX used to override only omegas/sigmas
                after loading the base elasticity set (typically COMP.gdx).
        """
        # First load sets
        self.sets.load_from_gdx(gdx_path)
        
        # Load elasticities from separate file if provided, otherwise from main file
        if elasticity_gdx is None:
            candidate = gdx_path.with_name("default-9x10.gdx")
            elast_file = candidate if candidate.exists() else gdx_path
        else:
            elast_file = elasticity_gdx
        self.elasticities.load_from_gdx(elast_file, self.sets)
        if elasticity_override_gdx is not None:
            self.elasticities.override_omegas_sigmas_from_gdx(elasticity_override_gdx, self.sets)
        self.elasticities.initialize_nested_elasticities(self.sets)  # Ensure coverage
        
        # Load benchmark and taxes from main file
        self.benchmark.load_from_gdx(gdx_path, self.sets)
        self.taxes.load_from_gdx(gdx_path, self.sets)
        self.taxes.derive_from_benchmark(self.benchmark, self.sets)
        self.shifts.load_from_gdx(gdx_path)

        # Calibrate share parameters (simple shares)
        self.shares.calibrate(self.benchmark, self.elasticities, self.sets)
        
        # Calibrate GAMS-style shares (and, ava, io, af, gx) from benchmark
        self.calibrated.calibrate_from_benchmark(self.benchmark, self.elasticities, self.sets, self.taxes)

    def apply_equilibrium_snapshot(self, snapshot: GTAPEquilibriumSnapshot) -> None:
        """Override share parameters using an equilibrium CSV snapshot."""
        self.shares.apply_equilibrium_snapshot(snapshot, self.sets)
        
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate all parameters."""
        errors = []
        
        # Validate sets
        sets_valid, sets_errors = self.sets.validate()
        errors.extend(sets_errors)
        
        # Validate elasticities are within reasonable bounds
        for key, value in self.elasticities.esubva.items():
            if value < 0:
                errors.append(f"esubva{key} = {value} < 0")
                
        for key, value in self.elasticities.esubm.items():
            if value < 0:
                errors.append(f"esubm{key} = {value} < 0")
        
        return len(errors) == 0, errors
    
    def get_info(self) -> Dict[str, Any]:
        """Get summary information."""
        is_valid, errors = self.validate()
        return {
            "sets": self.sets.get_info(),
            "valid": is_valid,
            "errors": errors,
            "n_elasticities": len(self.elasticities.esubva) + len(self.elasticities.esubm),
            "n_benchmark_values": len(self.benchmark.vom) + len(self.benchmark.vfm),
            "n_tax_rates": len(self.taxes.rto) + len(self.taxes.rtms),
            "n_share_params": len(self.shares.p_gx) + len(self.shares.p_alpham),
        }
