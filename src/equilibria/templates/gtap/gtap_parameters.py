"""GTAP Parameters (CGEBox version)

This module defines all GTAP model parameters following the CGEBox implementation.
Reference: /Users/marmol/proyectos2/cge_babel/cgebox/gams/model/model.gms

Parameters include:
- Elasticities (substitution, transformation)
- Benchmark SAM values
- Tax rates
- Share parameters (from calibration)
- Technical change parameters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values
from equilibria.templates.gtap.gtap_sets import GTAPSets


@dataclass
class GTAPElasticities:
    """Elasticity parameters for GTAP model.
    
    Key elasticities:
    - esubva: CES elasticity between value-added and intermediate demand
    - esubd: CES elasticity between domestic and imported goods (top Armington)
    - esubm: CES elasticity across import sources (bottom Armington)
    - etrae: CET elasticity for factor mobility across sectors
    - omegax: CET elasticity between domestic sales and exports
    - omegaw: CET elasticity across export destinations
    """
    
    # Production elasticities
    esubva: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, a)
    esubd: Dict[Tuple[str, str], float] = field(default_factory=dict)   # (r, i)
    
    # Trade elasticities - Armington
    esubm: Dict[Tuple[str, str], float] = field(default_factory=dict)   # (r, i)
    
    # Trade elasticities - CET
    omegax: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, i)
    omegaw: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, i)
    
    # Factor mobility elasticities
    etrae: Dict[str, float] = field(default_factory=dict)  # f
    
    # Demand elasticities
    esubg: Dict[str, float] = field(default_factory=dict)  # r (government)
    esubi: Dict[str, float] = field(default_factory=dict)  # r (investment)
    esubc: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (r, h) consumption
    
    # Transport margins
    sigmam: Dict[str, float] = field(default_factory=dict)  # m
    
    def load_from_gdx(self, gdx_path: Path, sets: GTAPSets) -> None:
        """Load elasticities from GDX file."""
        gdx_data = read_gdx(gdx_path)
        
        # Load production elasticities
        self._load_parameter(gdx_data, "esubva", self.esubva, (sets.r, sets.a))
        self._load_parameter(gdx_data, "esubd", self.esubd, (sets.r, sets.i))
        
        # Load trade elasticities
        self._load_parameter(gdx_data, "esubm", self.esubm, (sets.r, sets.i))
        self._load_parameter(gdx_data, "omegax", self.omegax, (sets.r, sets.i))
        self._load_parameter(gdx_data, "omegaw", self.omegaw, (sets.r, sets.i))
        
        # Load factor mobility
        self._load_parameter(gdx_data, "etrae", self.etrae, (sets.f,))
        
        # Load demand elasticities
        self._load_parameter(gdx_data, "esubg", self.esubg, (sets.r,))
        self._load_parameter(gdx_data, "esubi", self.esubi, (sets.r,))
        
    def _load_parameter(self, gdx_data: Dict, name: str, target: Dict, 
                       index_sets: Tuple[List[str], ...]) -> None:
        """Load a single parameter from GDX data."""
        try:
            values = read_parameter_values(gdx_data, name)
            if values:
                target.update(values)
        except (KeyError, ValueError):
            # Parameter not found, will use defaults
            pass


@dataclass
class GTAPBenchmarkValues:
    """Benchmark SAM values from GTAP data.
    
    These are the base year values used for calibration.
    """
    
    # Production and supply
    vom: Dict[Tuple[str, str], float] = field(default_factory=dict)     # (r, a) - Output at market prices
    vom_i: Dict[Tuple[str, str], float] = field(default_factory=dict)   # (r, i) - Output by commodity
    
    # Factor payments
    vfm: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, f, a) - Factor payments
    vfb: Dict[Tuple[str, str], float] = field(default_factory=dict)     # (r, f) - Factor income
    
    # Intermediate demand
    vdfm: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, a) - Domestic intermediate
    vifm: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, a) - Imported intermediate
    
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
    
    # Trade at market prices
    viws: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, rp) - Imports (cif)
    vims: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, rp) - Imports ( tariff)
    
    # Income and savings
    yp: Dict[str, float] = field(default_factory=dict)      # r - Private income
    yg: Dict[str, float] = field(default_factory=dict)      # r - Government income
    
    def load_from_gdx(self, gdx_path: Path, sets: GTAPSets) -> None:
        """Load benchmark values from GDX file."""
        gdx_data = read_gdx(gdx_path)
        
        # Production
        self._load_param(gdx_data, "vom", self.vom, 2)
        
        # Factors
        self._load_param(gdx_data, "vfm", self.vfm, 3)
        self._load_param(gdx_data, "vfb", self.vfb, 2)
        
        # Intermediate demand
        self._load_param(gdx_data, "vdfm", self.vdfm, 3)
        self._load_param(gdx_data, "vifm", self.vifm, 3)
        
        # Final demand
        self._load_param(gdx_data, "vpm", self.vpm, 2)
        self._load_param(gdx_data, "vgm", self.vgm, 2)
        self._load_param(gdx_data, "vim", self.vim, 2)
        
        # Trade flows
        self._load_param(gdx_data, "vxmd", self.vxmd, 3)
        self._load_param(gdx_data, "viws", self.viws, 3)
        self._load_param(gdx_data, "vims", self.vims, 3)
        self._load_param(gdx_data, "vtwr", self.vtwr, 4)
        
    def _load_param(self, gdx_data: Dict, name: str, target: Dict, ndim: int) -> None:
        """Helper to load a parameter."""
        try:
            values = read_parameter_values(gdx_data, name)
            if values:
                target.update(values)
        except (KeyError, ValueError):
            pass


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
    
    # Direct taxes
    kappaf: Dict[Tuple[str, str], float] = field(default_factory=dict)   # (r, f) - Income tax rate
    
    def load_from_gdx(self, gdx_path: Path, sets: GTAPSets) -> None:
        """Load tax rates from GDX file."""
        gdx_data = read_gdx(gdx_path)
        
        self._load_param(gdx_data, "rto", self.rto, 2)
        self._load_param(gdx_data, "rtf", self.rtf, 3)
        self._load_param(gdx_data, "rtfd", self.rtfd, 3)
        self._load_param(gdx_data, "rtfi", self.rtfi, 3)
        self._load_param(gdx_data, "rtpd", self.rtpd, 3)
        self._load_param(gdx_data, "rtpi", self.rtpi, 3)
        self._load_param(gdx_data, "rtgd", self.rtgd, 2)
        self._load_param(gdx_data, "rtgi", self.rtgi, 2)
        self._load_param(gdx_data, "rtxs", self.rtxs, 3)
        self._load_param(gdx_data, "rtms", self.rtms, 3)
        
    def _load_param(self, gdx_data: Dict, name: str, target: Dict, ndim: int) -> None:
        """Helper to load a parameter."""
        try:
            values = read_parameter_values(gdx_data, name)
            if values:
                target.update(values)
        except (KeyError, ValueError):
            pass


@dataclass
class GTAPShareParameters:
    """Share parameters from calibration.
    
    These are derived from the benchmark SAM and elasticities.
    """
    
    # Production shares
    p_gx: Dict[Tuple[str, str, str], float] = field(default_factory=dict)     # (r, a, i) - CET share for output allocation
    p_ax: Dict[Tuple[str, str, str], float] = field(default_factory=dict)     # (r, a, i) - CES share for commodity production
    
    # Armington shares
    p_alphad: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, aa) - Domestic share
    p_alpham: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, aa) - Import share (top nest)
    p_alphan: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (r, i, aa) - National share
    
    # Bilateral trade shares
    p_amw: Dict[Tuple[str, str, str], float] = field(default_factory=dict)    # (r, i, rp) - Import share by source
    p_gw: Dict[Tuple[str, str, str], float] = field(default_factory=dict)     # (r, i, rp) - Export share by destination
    
    # Factor shares
    p_gf: Dict[Tuple[str, str, str], float] = field(default_factory=dict)     # (r, f, a) - Factor supply share
    
    # Demand shares
    p_axg: Dict[str, float] = field(default_factory=dict)     # r - Government CES shifter
    p_axi: Dict[str, float] = field(default_factory=dict)     # r - Investment CES shifter
    
    def calibrate(self, benchmark: GTAPBenchmarkValues, elasticities: GTAPElasticities, 
                  sets: GTAPSets) -> None:
        """Calibrate all share parameters from benchmark data."""
        self._calibrate_production_shares(benchmark, sets)
        self._calibrate_armington_shares(benchmark, sets)
        self._calibrate_trade_shares(benchmark, sets)
        self._calibrate_factor_shares(benchmark, sets)
        
    def _calibrate_production_shares(self, benchmark: GTAPBenchmarkValues, sets: GTAPSets) -> None:
        """Calibrate production share parameters."""
        # p_gx: CET shares for output allocation
        for r in sets.r:
            for a in sets.a:
                total_output = sum(benchmark.vom.get((r, a), 0.0) for a in sets.a)
                if total_output > 0:
                    for i in sets.i:
                        value = benchmark.vom.get((r, a), 0.0)
                        self.p_gx[(r, a, i)] = value / total_output
                        
    def _calibrate_armington_shares(self, benchmark: GTAPBenchmarkValues, sets: GTAPSets) -> None:
        """Calibrate Armington share parameters."""
        # p_alphad: Domestic share
        # p_alpham: Import share
        # This requires aggregating across agents
        pass
        
    def _calibrate_trade_shares(self, benchmark: GTAPBenchmarkValues, sets: GTAPSets) -> None:
        """Calibrate bilateral trade share parameters."""
        # p_amw: Import share by source
        for r in sets.r:
            for i in sets.i:
                total_imports = sum(benchmark.viws.get((rp, i, r), 0.0) for rp in sets.r if rp != r)
                if total_imports > 0:
                    for rp in sets.r:
                        if rp != r:
                            value = benchmark.viws.get((rp, i, r), 0.0)
                            self.p_amw[(r, i, rp)] = value / total_imports
                            
    def _calibrate_factor_shares(self, benchmark: GTAPBenchmarkValues, sets: GTAPSets) -> None:
        """Calibrate factor share parameters."""
        # p_gf: Factor supply shares
        for r in sets.r:
            for f in sets.f:
                total_payment = sum(benchmark.vfm.get((r, f, a), 0.0) for a in sets.a)
                if total_payment > 0:
                    for a in sets.a:
                        value = benchmark.vfm.get((r, f, a), 0.0)
                        self.p_gf[(r, f, a)] = value / total_payment


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
    
    def load_from_gdx(self, gdx_path: Path) -> None:
        """Load all parameters from GDX file.
        
        Args:
            gdx_path: Path to GTAP GDX file
        """
        # First load sets
        self.sets.load_from_gdx(gdx_path)
        
        # Then load parameters that depend on sets
        self.elasticities.load_from_gdx(gdx_path, self.sets)
        self.benchmark.load_from_gdx(gdx_path, self.sets)
        self.taxes.load_from_gdx(gdx_path, self.sets)
        
        # Calibrate share parameters
        self.shares.calibrate(self.benchmark, self.elasticities, self.sets)
        
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
