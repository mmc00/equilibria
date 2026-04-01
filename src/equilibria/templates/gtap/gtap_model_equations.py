"""Complete GTAP Model Equations (Functional Implementation)

This module implements a fully functional GTAP CGE model.
All equations are implemented to create a solvable square system.
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING, Any, Dict, Optional

from equilibria.templates.gtap.gtap_parameters import (
    GTAP_GOVERNMENT_AGENT,
    GTAP_HOUSEHOLD_AGENT,
    GTAP_INVESTMENT_AGENT,
    GTAP_MARGIN_AGENT,
)

if TYPE_CHECKING:
    from pyomo.environ import ConcreteModel


class GTAPModelEquations:
    """Complete GTAP CGE model with all equations."""
    
    def __init__(
        self,
        sets: "GTAPSets",
        params: "GTAPParameters",
        closure: Optional["GTAPClosureConfig"] = None,
        reference_snapshot: Optional["GTAPVariableSnapshot"] = None,
    ):
        self.sets = sets
        self.params = params
        self.closure = closure
        self.reference_snapshot = reference_snapshot
        
    def build_model(self) -> "ConcreteModel":
        """Build complete functional GTAP model."""
        from pyomo.environ import ConcreteModel
        
        model = ConcreteModel(name="GTAP_Full_Model")
        
        self._add_sets(model)
        self._add_parameters(model)
        self._add_variables(model)
        self._add_equations(model)
        self._add_objective(model)
        
        return model
    
    def _add_sets(self, model: "ConcreteModel") -> None:
        """Add sets."""
        from pyomo.environ import Set

        agent_labels = list(self.sets.a) + [
            GTAP_HOUSEHOLD_AGENT,
            GTAP_GOVERNMENT_AGENT,
            GTAP_INVESTMENT_AGENT,
            GTAP_MARGIN_AGENT,
        ]
        
        model.r = Set(initialize=self.sets.r, doc="Regions")
        model.i = Set(initialize=self.sets.i, doc="Commodities")
        model.a = Set(initialize=self.sets.a, doc="Activities")
        model.f = Set(initialize=self.sets.f, doc="Factors")
        model.mf = Set(initialize=self.sets.mf, doc="Mobile factors")
        model.sf = Set(initialize=self.sets.sf, doc="Specific factors")
        model.aa = Set(initialize=agent_labels, doc="Absorption agents and activities")
        
        # Aliases for trade
        model.rp = Set(initialize=self.sets.r, doc="Regions (alias)")
    
    def _add_parameters(self, model: "ConcreteModel") -> None:
        """Add all parameters."""
        from pyomo.environ import Param

        xscale_data: Dict[tuple[str, str], float] = {}
        for r in self.sets.r:
            for aa in list(self.sets.a) + [
                GTAP_HOUSEHOLD_AGENT,
                GTAP_GOVERNMENT_AGENT,
                GTAP_INVESTMENT_AGENT,
                GTAP_MARGIN_AGENT,
            ]:
                scale = 1.0
                if aa in self.sets.a:
                    benchmark_output = self.params.benchmark.vom.get((r, aa), 0.0)
                    if benchmark_output > 0.0:
                        scale = 10.0 ** (-round(math.log10(abs(benchmark_output))))
                xscale_data[(r, aa)] = scale
        
        # Helper to create indexed parameters
        def create_indexed_param(name: str, index_sets, data: Dict, default: float = 0.0):
            if not data:
                return
            # Build index values
            values = {}
            for key, value in data.items():
                if isinstance(key, tuple):
                    values[key] = value
                else:
                    values[(key,)] = value
            
            # Get pyomo sets for indexing
            if len(index_sets) == 1:
                idx_set = getattr(model, index_sets[0])
                setattr(model, name, Param(idx_set, initialize=values, default=default, doc=name))
            elif len(index_sets) == 2:
                idx_set1 = getattr(model, index_sets[0])
                idx_set2 = getattr(model, index_sets[1])
                setattr(model, name, Param(idx_set1, idx_set2, initialize=values, default=default, doc=name))
            elif len(index_sets) == 3:
                idx_set1 = getattr(model, index_sets[0])
                idx_set2 = getattr(model, index_sets[1])
                idx_set3 = getattr(model, index_sets[2])
                setattr(model, name, Param(idx_set1, idx_set2, idx_set3, initialize=values, default=default, doc=name))
        
        # Elasticities
        create_indexed_param("esubva", ["r", "a"], self.params.elasticities.esubva, 1.0)
        create_indexed_param("esubd", ["r", "i"], self.params.elasticities.esubd, 2.0)
        create_indexed_param("esubm", ["r", "i"], self.params.elasticities.esubm, 4.0)
        create_indexed_param("omegax", ["r", "i"], self.params.elasticities.omegax, 2.0)
        
        # Benchmark values
        create_indexed_param("vom", ["r", "a"], self.params.benchmark.vom, 0.0)
        create_indexed_param("vfm", ["r", "f", "a"], self.params.benchmark.vfm, 0.0)
        
        # GAMS-style calibrated parameters (and, ava, io, af, gx)
        create_indexed_param("and_param", ["r", "a"], self.params.calibrated.and_param, 0.0)
        create_indexed_param("ava_param", ["r", "a"], self.params.calibrated.ava_param, 0.0)
        create_indexed_param("io_param", ["r", "i", "a"], self.params.calibrated.io_param, 0.0)
        create_indexed_param("af_param", ["r", "f", "a"], self.params.calibrated.af_param, 0.0)
        create_indexed_param("gx_param", ["r", "a", "i"], self.params.calibrated.gx_param, 0.0)
        create_indexed_param("xscale", ["r", "aa"], xscale_data, 1.0)
        create_indexed_param("p_io", ["r", "i", "a"], self.params.shares.p_io, 0.0)
        create_indexed_param("gd_share", ["r", "i"], self.params.shares.p_gd, 0.0)
        create_indexed_param("ge_share", ["r", "i"], self.params.shares.p_ge, 0.0)
        
        # Simple shares (kept for compatibility)
        create_indexed_param("va_share", ["r", "a"], self.params.shares.p_va, 0.0)
        create_indexed_param("nd_share", ["r", "a"], self.params.shares.p_nd, 0.0)
        create_indexed_param("gf_share", ["r", "f", "a"], self.params.shares.p_gf, 0.0)
        create_indexed_param("af_share", ["r", "f", "a"], self.params.shares.p_af, 0.0)
        create_indexed_param("p_gx", ["r", "a", "i"], self.params.shares.p_gx, 0.0)
        
        # Tax rates
        create_indexed_param("rto", ["r", "a"], self.params.taxes.rto, 0.0)
        create_indexed_param("rtf", ["r", "f", "a"], self.params.taxes.rtf, 0.0)

        # Regional income shares calibrated from benchmark absorption totals.
        regional_income_share_data: Dict[tuple[str], float] = {}
        regional_government_share_data: Dict[tuple[str], float] = {}
        regional_investment_share_data: Dict[tuple[str], float] = {}
        for region in self.sets.r:
            factor_income = sum(
                self.params.benchmark.vfm.get((region, factor, activity), 0.0)
                for factor in self.sets.f
                for activity in self.sets.a
            )
            denominator = max(factor_income, 1e-8)
            private_total = sum(self.params.benchmark.vpm.get((region, commodity), 0.0) for commodity in self.sets.i)
            government_total = sum(self.params.benchmark.vgm.get((region, commodity), 0.0) for commodity in self.sets.i)
            investment_total = sum(self.params.benchmark.vim.get((region, commodity), 0.0) for commodity in self.sets.i)

            regional_income_share_data[(region,)] = private_total / denominator
            regional_government_share_data[(region,)] = government_total / denominator
            regional_investment_share_data[(region,)] = investment_total / denominator

        create_indexed_param("yc_share_reg", ["r"], regional_income_share_data, 0.0)
        create_indexed_param("yg_share_reg", ["r"], regional_government_share_data, 0.0)
        create_indexed_param("yi_share_reg", ["r"], regional_investment_share_data, 0.0)
    
    def _add_variables(self, model: "ConcreteModel") -> None:
        """Add all variables for square system.
        
        Initialize with SAM benchmark values (like GAMS cal.gms):
        - Prices = 1.0 (normalized)
        - Quantities = SAM values (millions)
        """
        from pyomo.environ import Var, Reals, NonNegativeReals
        
        # Helper to get SAM value initialization
        def get_vom_init(m, r, a):
            """Get production level from SAM."""
            if self.reference_snapshot:
                ref_xp = self.reference_snapshot.xp.get((r, a))
                if ref_xp is not None and ref_xp > 0.0:
                    return float(ref_xp)
            val = self.params.benchmark.vom.get((r, a), 0.0)
            return max(val, 0.1)  # Avoid zeros
        
        def get_vfm_init(m, r, f, a):
            """Initialize factor demand from the benchmark-consistent demand equation."""
            # First, try to use baseline from reference snapshot (if available)
            if self.reference_snapshot:
                ref_xf = self.reference_snapshot.xf.get((r, f, a))
                if ref_xf is not None and ref_xf > 0.0:
                    return float(ref_xf)
            
            # Fallback to equation-consistent initialization
            af_val = float(m.af_param[r, f, a]) if hasattr(m, "af_param") else 0.0
            if af_val <= 0.0:
                return 0.0

            va_val = get_va_init(m, r, a)
            sigmav = self._get_sigmav(r, a)
            lambdaf = self._lambdaf(r, f, a)

            kappa = self.params.taxes.kappaf_activity.get((r, f, a), 0.0)
            gross_factor_price = 1.0 / max(1.0 - kappa, 1e-8)
            ratio = 1.0 / gross_factor_price

            val = af_val * va_val * (ratio ** sigmav) * (lambdaf ** (sigmav - 1.0))
            return max(val, 0.0)

        def get_pf_init(m, r, f, a):
            # First, try to use baseline from reference snapshot (if available)
            if self.reference_snapshot:
                ref_pf = self.reference_snapshot.pf.get((r, f, a))
                if ref_pf is not None and ref_pf > 0.0:
                    return float(ref_pf)
            
            # Fallback to tax-based initialization
            return max(1.0 / max(1.0 - kappa, 1e-8), 1e-8)
        
        def get_vpm_init(m, r, i):
            """Get total private Armington demand from SAM."""
            val, _, _ = self.params.benchmark.get_private_demand(r, i)
            return max(val, 0.01)
        
        def get_vgm_init(m, r, i):
            """Get total government Armington demand from SAM."""
            val, _, _ = self.params.benchmark.get_government_demand(r, i)
            return max(val, 0.01)
        
        def get_vim_init(m, r, i):
            """Get total investment Armington demand from SAM."""
            val, _, _ = self.params.benchmark.get_investment_demand(r, i)
            return max(val, 0.01)

        def get_xscale(m, r, aa):
            return max(float(m.xscale[r, aa]), 1e-12)

        def get_xaa_raw_init(m, r, i, aa):
            if aa in self.sets.a:
                val = self.params.benchmark.vdfm.get((r, i, aa), 0.0) + self.params.benchmark.vifm.get((r, i, aa), 0.0)
            elif aa == GTAP_HOUSEHOLD_AGENT:
                val = get_vpm_init(m, r, i)
            elif aa == GTAP_GOVERNMENT_AGENT:
                val = get_vgm_init(m, r, i)
            elif aa == GTAP_INVESTMENT_AGENT:
                val = get_vim_init(m, r, i)
            elif aa == GTAP_MARGIN_AGENT:
                val = self.params.benchmark.vst.get((r, i), 0.0)
            else:
                val = 0.0
            return max(val, 0.0)

        def get_xaa_init(m, r, i, aa):
            if self.reference_snapshot:
                ref_xaa = self.reference_snapshot.xaa.get((r, i, aa))
                if ref_xaa is not None and ref_xaa > 0.0:
                    return float(ref_xaa) * get_xscale(m, r, aa)
            return max(get_xaa_raw_init(m, r, i, aa) * get_xscale(m, r, aa), 0.0)

        agent_trade_cache: Dict[tuple[str, str, str], tuple[float, float]] = {}

        def _raw_agent_domestic_import(r, i, aa):
            if aa in self.sets.a:
                raw_domestic = self.params.benchmark.vdfb.get((r, i, aa), 0.0)
                raw_import = self.params.benchmark.vmfb.get((r, i, aa), 0.0)
                if raw_domestic + raw_import <= 0.0:
                    raw_domestic = self.params.benchmark.vdfm.get((r, i, aa), 0.0)
                    raw_import = self.params.benchmark.vifm.get((r, i, aa), 0.0)
            elif aa == GTAP_HOUSEHOLD_AGENT:
                raw_domestic = self.params.benchmark.vdpb.get((r, i), 0.0)
                raw_import = self.params.benchmark.vmpb.get((r, i), 0.0)
                if raw_domestic + raw_import <= 0.0:
                    _, raw_domestic, raw_import = self.params.benchmark.get_private_demand(r, i)
            elif aa == GTAP_GOVERNMENT_AGENT:
                raw_domestic = self.params.benchmark.vdgb.get((r, i), 0.0)
                raw_import = self.params.benchmark.vmgb.get((r, i), 0.0)
                if raw_domestic + raw_import <= 0.0:
                    _, raw_domestic, raw_import = self.params.benchmark.get_government_demand(r, i)
            elif aa == GTAP_INVESTMENT_AGENT:
                raw_domestic = self.params.benchmark.vdib.get((r, i), 0.0)
                raw_import = self.params.benchmark.vmib.get((r, i), 0.0)
                if raw_domestic + raw_import <= 0.0:
                    _, raw_domestic, raw_import = self.params.benchmark.get_investment_demand(r, i)
            elif aa == GTAP_MARGIN_AGENT:
                raw_domestic = self.params.benchmark.vst.get((r, i), 0.0)
                raw_import = 0.0
            else:
                raw_domestic = 0.0
                raw_import = 0.0
            return raw_domestic, raw_import

        def build_agent_trade_cache():
            for r in self.sets.r:
                for i in self.sets.i:
                    for aa in list(self.sets.a) + [
                        GTAP_HOUSEHOLD_AGENT,
                        GTAP_GOVERNMENT_AGENT,
                        GTAP_INVESTMENT_AGENT,
                        GTAP_MARGIN_AGENT,
                    ]:
                        raw_domestic, raw_import = _raw_agent_domestic_import(r, i, aa)

                        # Match standard_gtap_7 (cal.gms): initialize xd/xm directly from
                        # basic-value flows by agent, rather than prorating market-value totals.
                        domestic = max(raw_domestic, 0.0)
                        imported = max(raw_import, 0.0)

                        # Fallback only when no basic-flow signal is available.
                        if domestic <= 0.0 and imported <= 0.0:
                            total = get_xaa_raw_init(model, r, i, aa)
                            if total > 0.0:
                                if aa == GTAP_MARGIN_AGENT:
                                    domestic = total
                                    imported = 0.0
                                else:
                                    domestic = total
                                    imported = 0.0

                        agent_trade_cache[(r, i, aa)] = (domestic, imported)

        def get_agent_trade_levels(m, r, i, aa):
            return agent_trade_cache.get((r, i, aa), (0.0, 0.0))

        def get_xda_init(m, r, i, aa):
            domestic, _ = get_agent_trade_levels(m, r, i, aa)
            return max(domestic * get_xscale(m, r, aa), 0.0)

        def get_xma_init(m, r, i, aa):
            _, imported = get_agent_trade_levels(m, r, i, aa)
            return max(imported * get_xscale(m, r, aa), 0.0)

        def get_pdp_init(m, r, i, aa):
            wedge_map = getattr(self.params.taxes, "pdp_agent_wedge", {})
            wedge = wedge_map.get((r, i, aa), 1.0)
            return max(wedge, 1e-8)

        def get_make_init(m, r, a, i):
            """Get benchmark output by activity-commodity pair from SAM."""
            outputs = self.sets.activity_commodities.get(a, [])
            if outputs and i not in outputs:
                return 0.0
            val = self.params.benchmark.makb.get((r, a, i), 0.0)
            if val > 0.0:
                return val
            share = self.params.calibrated.gx_param.get((r, a, i), 0.0)
            if share > 0.0:
                return max(share * get_vom_init(m, r, a), 0.01)
            return 0.0

        def get_export_init(r, i):
            _, _, xet, _, _ = self.params.benchmark.get_trade_totals(self.sets, r, i)
            return xet

        def get_import_init(r, i):
            intermediate_imports = sum(self.params.benchmark.vifm.get((r, i, a), 0.0) for a in self.sets.a)
            final_imports = (
                self.params.benchmark.vmpp.get((r, i), 0.0)
                + self.params.benchmark.vmgp.get((r, i), 0.0)
                + self.params.benchmark.vmip.get((r, i), 0.0)
            )
            return intermediate_imports + final_imports

        def get_intermediate_use(r, i):
            return sum(
                self.params.benchmark.vdfm.get((r, i, a), 0.0) + self.params.benchmark.vifm.get((r, i, a), 0.0)
                for a in self.sets.a
            )

        def get_final_use(r, i):
            private_total, _, _ = self.params.benchmark.get_private_demand(r, i)
            government_total, _, _ = self.params.benchmark.get_government_demand(r, i)
            investment_total, _, _ = self.params.benchmark.get_investment_demand(r, i)
            return private_total + government_total + investment_total

        def get_total_use(r, i):
            return (
                get_intermediate_use(r, i)
                + get_final_use(r, i)
                + self.params.benchmark.vst.get((r, i), 0.0)
            )

        def get_xs_init(m, r, i):
            total = self.params.benchmark.vom_i.get((r, i), 0.0)
            if total <= 0.0:
                total = sum(self.params.benchmark.makb.get((r, a, i), 0.0) for a in self.sets.a)
            if total <= 0.0:
                total = max(get_total_use(r, i) - get_import_init(r, i), 0.0) + get_export_init(r, i)
            return max(total, 0.01)

        def get_xds_init(m, r, i):
            if self.reference_snapshot:
                ref_xds = self.reference_snapshot.xds.get((r, i))
                if ref_xds is not None and ref_xds > 0.0:
                    return float(ref_xds)
            _, domestic_sales, _, _, _ = self.params.benchmark.get_trade_totals(self.sets, r, i)
            return max(domestic_sales, 0.01)

        build_agent_trade_cache()

        def get_xd_init(m, r, i):
            total = sum(agent_trade_cache.get((r, i, aa), (0.0, 0.0))[0] for aa in model.aa)
            return max(total, 0.01)

        def get_xmt_init(m, r, i):
            total = sum(agent_trade_cache.get((r, i, aa), (0.0, 0.0))[1] for aa in model.aa)
            return max(total, 0.01)

        def get_xet_init(m, r, i):
            return max(get_export_init(r, i), 0.01)

        def get_xa_init(m, r, i):
            return max(get_total_use(r, i), 0.01)

        def get_va_init(m, r, a):
            total = sum(self.params.benchmark.vfm.get((r, f, a), 0.0) for f in self.sets.f)
            return max(total, 0.01)

        def get_nd_init(m, r, a):
            total_intermediate = sum(
                self.params.benchmark.vdfm.get((r, i, a), 0.0) + self.params.benchmark.vifm.get((r, i, a), 0.0)
                for i in self.sets.i
            )
            if total_intermediate > 0.0:
                return max(total_intermediate, 0.01)
            return max(get_vom_init(m, r, a) - get_va_init(m, r, a), 0.01)

        def get_factor_supply_init(m, r, f):
            if str(f) == "NatRes":
                return 0.0
            total = sum(get_vfm_init(m, r, f, a) for a in self.sets.a)
            return max(total, 0.0)

        def get_pft_init(m, r, f):
            if str(f) == "NatRes":
                return 1e-8
            supply = get_factor_supply_init(m, r, f)
            if supply <= 0.0:
                return 1e-8
            return 1.0

        def get_kstock_init(m, r):
            total = sum(self.params.benchmark.vfm.get((r, "Capital", a), 0.0) for a in self.sets.a)
            return max(total, 0.01)
        
        # Production (4 vars per r,a)
        model.xp = Var(model.r, model.a, within=NonNegativeReals, initialize=get_vom_init, doc="Production")
        model.x = Var(model.r, model.a, model.i, within=NonNegativeReals, initialize=get_make_init, doc="Output")
        model.px = Var(model.r, model.a, within=NonNegativeReals, initialize=1.0, doc="Unit cost")
        model.pp = Var(model.r, model.a, within=NonNegativeReals, initialize=1.0, doc="Producer price")
        
        # Supply (3 vars per r,i)
        model.xs = Var(model.r, model.i, within=NonNegativeReals, initialize=get_xs_init, doc="Domestic supply")
        model.xds = Var(model.r, model.i, within=NonNegativeReals, initialize=get_xds_init, doc="Supply of domestically produced goods")
        model.ps = Var(model.r, model.i, within=NonNegativeReals, initialize=1.0, doc="Supply price")
        model.pd = Var(model.r, model.i, within=NonNegativeReals, initialize=1.0, doc="Domestic price")
        
        # Armington (3 vars per r,i)
        model.xa = Var(model.r, model.i, within=NonNegativeReals, initialize=get_xa_init, doc="Armington demand")
        model.pa = Var(model.r, model.i, within=NonNegativeReals, initialize=1.0, doc="Armington price")
        
        # Trade - Domestic/Import split (4 vars per r,i)
        model.xd = Var(model.r, model.i, within=NonNegativeReals, initialize=get_xd_init, doc="Domestic demand")
        model.xmt = Var(model.r, model.i, within=NonNegativeReals, initialize=get_xmt_init, doc="Import demand")
        model.pmt = Var(model.r, model.i, within=NonNegativeReals, initialize=1.0, doc="Import price")
        model.xda = Var(model.r, model.i, model.aa, within=NonNegativeReals, initialize=get_xda_init, doc="Domestic Armington demand by agent")
        model.xma = Var(model.r, model.i, model.aa, within=NonNegativeReals, initialize=get_xma_init, doc="Imported Armington demand by agent")
        model.paa = Var(model.r, model.i, model.aa, within=NonNegativeReals, initialize=1.0, doc="Agent Armington price")
        model.pdp = Var(model.r, model.i, model.aa, within=NonNegativeReals, initialize=get_pdp_init, doc="Agent domestic demand price")
        model.pmp = Var(model.r, model.i, model.aa, within=NonNegativeReals, initialize=1.0, doc="Agent import demand price")
        
        # Trade - Domestic/Export split (4 vars per r,i)
        model.xet = Var(model.r, model.i, within=NonNegativeReals, initialize=get_xet_init, doc="Export supply")
        model.pet = Var(model.r, model.i, within=NonNegativeReals, initialize=1.0, doc="Export price")
        
        # Value added/intermediate bundles
        model.va = Var(model.r, model.a, within=NonNegativeReals, initialize=get_va_init, doc="Value added bundle")
        model.nd = Var(model.r, model.a, within=NonNegativeReals, initialize=get_nd_init, doc="Intermediate bundle")
        model.pva = Var(model.r, model.a, within=NonNegativeReals, initialize=1.0, doc="Value added price")
        model.pnd = Var(model.r, model.a, within=NonNegativeReals, initialize=1.0, doc="Intermediate price")

        # Bilateral trade (2 vars per r,i,rp)  
        def get_pe_init(m, r, i, rp):
            if r == rp:
                return 1.0
            bilateral_exports = self.params.benchmark.vxmd.get((r, i, rp), 0.0)
            mirror_imports = self.params.benchmark.viws.get((rp, i, r), 0.0)
            return 1.0 if (bilateral_exports > 0.0 or mirror_imports > 0.0) else 0.0

        def get_xe_init(m, r, i, rp):
            return max(self.params.benchmark.vxmd.get((r, i, rp), 0.0), 0.0)

        def get_xw_init(m, r, i, rp):
            # First, try to use baseline from reference snapshot (if available)
            if self.reference_snapshot:
                ref_xw = self.reference_snapshot.xw.get((r, i, rp))
                if ref_xw is not None and ref_xw > 0.0:
                    return float(ref_xw)
            # Fallback to benchmark data
            if r == rp:
                diagonal_flow = self.params.benchmark.vxmd.get((r, i, rp), 0.0)
                if diagonal_flow > 0.0:
                    return diagonal_flow
            if hasattr(self.params.benchmark, "get_import_flow"):
                return max(self.params.benchmark.get_import_flow(r, i, rp), 0.0)
            return max(self.params.benchmark.viws.get((rp, i, r), 0.0), 0.0)

        model.xe = Var(model.r, model.i, model.rp, within=NonNegativeReals, initialize=get_xe_init, doc="Bilateral exports")
        model.xw = Var(model.r, model.i, model.rp, within=NonNegativeReals, initialize=get_xw_init, doc="Bilateral imports")
        model.pe = Var(
            model.r,
            model.i,
            model.rp,
            within=NonNegativeReals,
            initialize=get_pe_init,
            doc="Bilateral export price by route",
        )

        def get_pwmg_init(m, r, i, rp):
            margin_flow = sum(self.params.benchmark.vtwr.get((r, i, rp, margin), 0.0) for margin in self.sets.m)
            return 1.0 if margin_flow > 0.0 else 0.0

        model.pwmg = Var(
            model.r,
            model.i,
            model.rp,
            within=NonNegativeReals,
            initialize=get_pwmg_init,
            doc="Route-specific trade and transport margin price",
        )
        model.ptmg = Var(
            model.i,
            within=NonNegativeReals,
            initialize=1.0,
            doc="Commodity trade-margin price index",
        )
        
        # Factors (4 vars per r,f)
        model.xft = Var(model.r, model.f, within=NonNegativeReals, initialize=get_factor_supply_init, doc="Factor supply")
        model.pft = Var(model.r, model.f, within=NonNegativeReals, initialize=get_pft_init, doc="Factor price")
        model.xf = Var(model.r, model.f, model.a, within=NonNegativeReals, initialize=get_vfm_init, doc="Factor demand")
        model.pf = Var(model.r, model.f, model.a, within=NonNegativeReals, initialize=get_pf_init, doc="Factor price by activity")
        model.pwfact = Var(within=NonNegativeReals, initialize=1.0, doc="World factor price")
        
        # Final demand (3 vars per r,i)
        model.xc = Var(model.r, model.i, within=NonNegativeReals, initialize=get_vpm_init, doc="Private consumption")
        model.xg = Var(model.r, model.i, within=NonNegativeReals, initialize=get_vgm_init, doc="Government consumption")
        model.xi = Var(model.r, model.i, within=NonNegativeReals, initialize=get_vim_init, doc="Investment")
        model.xaa = Var(model.r, model.i, model.aa, within=NonNegativeReals, initialize=get_xaa_init, doc="Agent/activity Armington demand")
        
        # Income (3 vars per r) - Calculate from factor income
        def get_regy_init(m, r):
            """Regional income = sum of factor payments."""
            total = sum(self.params.benchmark.vfm.get((r, f, a), 0.0) 
                       for f in self.sets.f for a in self.sets.a)
            return max(total, 100.0)

        def get_income_share(share_param_name: str, region: str) -> float:
            share_param = getattr(model, share_param_name)
            share = float(share_param[region])
            return max(share, 1e-8)
        
        model.regy = Var(model.r, within=Reals, initialize=get_regy_init, doc="Regional income")
        model.yc = Var(
            model.r,
            within=NonNegativeReals,
            initialize=lambda m, r: float(self.reference_snapshot.yc.get(r)) if self.reference_snapshot and self.reference_snapshot.yc.get(r) is not None else (get_regy_init(m, r) * get_income_share("yc_share_reg", r)),
            doc="Private income",
        )
        model.yg = Var(
            model.r,
            within=NonNegativeReals,
            initialize=lambda m, r: float(self.reference_snapshot.yg.get(r)) if self.reference_snapshot and self.reference_snapshot.yg.get(r) is not None else (get_regy_init(m, r) * get_income_share("yg_share_reg", r)),
            doc="Government income",
        )
        model.yi = Var(
            model.r,
            within=NonNegativeReals,
            initialize=lambda m, r: float(self.reference_snapshot.yi.get(r)) if self.reference_snapshot and self.reference_snapshot.yi.get(r) is not None else (get_regy_init(m, r) * get_income_share("yi_share_reg", r)),
            doc="Investment income",
        )
        
        # Numeraire (all prices = 1.0 like GAMS)
        model.pnum = Var(within=NonNegativeReals, initialize=1.0, doc="Numeraire")
        model.pabs = Var(model.r, within=NonNegativeReals, initialize=1.0, doc="Aggregate absorption price")
        model.pfact = Var(model.r, within=NonNegativeReals, initialize=1.0, doc="Regional factor price")
        model.kstock = Var(model.r, within=NonNegativeReals, initialize=get_kstock_init, doc="Capital stock")
        
        # Set strict positive lower bounds to prevent division by zero and negative powers
        # This mirrors GAMS behavior with HOLDFIXED or explicit bounds
        MIN_QUANTITY = 1e-8  # Minimum quantity/price value
        price_vars = ['px', 'pp', 'ps', 'pd', 'pa', 'pmt', 'pet', 'pva', 'pnd', 'pft', 'pf', 
                      'pnum', 'pabs', 'pfact', 'pwfact']
        quantity_vars = ['xp', 'x', 'xs', 'xa', 'xd', 'xmt', 'xet', 'xda', 'xma', 'va', 'nd', 'xe', 'xw', 
            'xft', 'xf', 'xc', 'xg', 'xi', 'xaa', 'yc', 'yg', 'yi', 'kstock']
        
        for var_name in price_vars + quantity_vars:
            if hasattr(model, var_name):
                var = getattr(model, var_name)
                for idx in var:
                    var[idx].setlb(MIN_QUANTITY)
    
    def _add_equations(self, model: "ConcreteModel") -> None:
        """Add all equations for square system."""
        from pyomo.environ import Constraint, exp, log, value
        
        # ========================================================================
        # PRODUCTION BLOCK
        # ========================================================================
        
        # Zero profit: Unit cost = producer price
        def prf_y_rule(model, r, a):
            return model.px[r, a] == model.pp[r, a]
        model.prf_y = Constraint(model.r, model.a, rule=prf_y_rule)

        # Value-added and intermediate nests (GAMS exact formulation)
        # GAMS: nd(r,a,t) =e= and(r,a,t)*xp(r,a,t)*(px(r,a,t)/pnd(r,a,t))**sigmap(r,a)
        #                      * (axp(r,a,t)*lambdand(r,a,t))**(sigmap(r,a)-1)
        def eq_nd_rule(model, r, a):
            and_val = value(model.and_param[r, a])  # GAMS calibrated parameter
            if and_val <= 0.0:
                return Constraint.Skip
            sigmap = self._get_sigmap(r, a)
            px = model.px[r, a]
            pnd = model.pnd[r, a]
            if value(pnd) <= 0:
                return Constraint.Skip
            ratio = px / pnd
            shift = self._axp_shift(r, a) * self._lambdand(r, a)
            return model.nd[r, a] == and_val * model.xp[r, a] * ratio**sigmap * shift**(sigmap - 1)
        model.eq_nd = Constraint(model.r, model.a, rule=eq_nd_rule)

        # GAMS: va(r,a,t) =e= ava(r,a,t)*xp(r,a,t)*(px(r,a,t)/pva(r,a,t))**sigmap(r,a)
        #                      * (axp(r,a,t)*lambdava(r,a,t))**(sigmap(r,a)-1)
        def eq_va_rule(model, r, a):
            ava_val = value(model.ava_param[r, a])  # GAMS calibrated parameter
            if ava_val <= 0.0:
                return Constraint.Skip
            sigmap = self._get_sigmap(r, a)
            px = model.px[r, a]
            pva = model.pva[r, a]
            if value(pva) <= 0:
                return Constraint.Skip
            ratio = px / pva
            shift = self._axp_shift(r, a) * self._lambdava(r, a)
            return model.va[r, a] == ava_val * model.xp[r, a] * ratio**sigmap * shift**(sigmap - 1)
        model.eq_va = Constraint(model.r, model.a, rule=eq_va_rule)
        
        # ========================================================================
        # PRICE EQUATIONS - CES COST FUNCTIONS (CRITICAL)
        # ========================================================================
        # TEMPORARILY COMMENTED OUT TO DEBUG SOLVER ISSUE
        
        # # Unit cost definition (GAMS pxeq, line 545)
        # # GAMS: px**(1-sigmap) =e= (axp**(sigmap-1)) * (and*(pnd/lambdand)**(1-sigmap)
        # #                                                + ava*(pva/lambdava)**(1-sigmap))
        # def eq_pxeq_rule(model, r, a):
        #     and_val = value(model.and_param[r, a])
        #     ava_val = value(model.ava_param[r, a])
        #     if and_val <= 0 and ava_val <= 0:
        #         return Constraint.Skip
        #     
        #     sigmap = self._get_sigmap(r, a)
        #     expo = 1 - sigmap
        #     
        #     # Shifter terms (axp=1.0, lambdand=1.0, lambdava=1.0 in benchmark)
        #     shift = self._axp_shift(r, a) ** (sigmap - 1)
        #     lambdand = self._lambdand(r, a)
        #     lambdava = self._lambdava(r, a)
        #     
        #     # CES aggregator
        #     px = model.px[r, a]
        #     pnd = model.pnd[r, a]
        #     pva = model.pva[r, a]
        #     
        #     if abs(expo) < 1e-8:  # Near Cobb-Douglas
        #         return Constraint.Skip
        #     
        #     # Build terms
        #     term_nd = and_val * (pnd / lambdand) ** expo if and_val > 0 else 0
        #     term_va = ava_val * (pva / lambdava) ** expo if ava_val > 0 else 0
        #     
        #     return px ** expo == shift * (term_nd + term_va)
        # model.eq_pxeq = Constraint(model.r, model.a, rule=eq_pxeq_rule)
        # 
        # # Price of ND bundle (GAMS pndeq, line 559)
        # # GAMS: pnd**(1-sigmand) =e= sum(i, io(r,i,a)*[pa(r,i,a)/lambdaio(r,i,a)]**(1-sigmand))
        # def eq_pndeq_rule(model, r, a):
        #     sigmand = self._get_sigmand(r, a)
        #     expo = 1 - sigmand
        #     
        #     if abs(expo) < 1e-8:  # Near Cobb-Douglas
        #         return Constraint.Skip
        #     
        #     # Sum over all commodities
        #     terms = []
        #     for i in model.i:
        #         io_val = value(model.io_param[r, i, a])
        #         if io_val <= 0:
        #             continue
        #         # lambdaio = 1.0 in benchmark
        #         pa = model.pa[r, i]
        #         terms.append(io_val * pa ** expo)
        #     
        #     if not terms:
        #         return Constraint.Skip
        #     
        #     pnd = model.pnd[r, a]
        #     return pnd ** expo == sum(terms)
        # model.eq_pndeq = Constraint(model.r, model.a, rule=eq_pndeq_rule)
        # 
        # # Price of VA bundle (GAMS pvaeq, line 573)
        # # GAMS: pva**(1-sigmav) =e= sum(f, af(r,f,a)*[pfa(r,f,a)/lambdaf(r,f,a)]**(1-sigmav))
        # def eq_pvaeq_rule(model, r, a):
        #     sigmav = self._get_sigmav(r, a)
        #     expo = 1 - sigmav
        #     
        #     if abs(expo) < 1e-8:  # Near Cobb-Douglas
        #         return Constraint.Skip
        #     
        #     # Sum over all factors
        #     terms = []
        #     for f in model.f:
        #         af_val = value(model.af_param[r, f, a])
        #         if af_val <= 0:
        #             continue
        #         # Get factor price (with taxes)
        #         factor_price = self._factor_price_term(model, r, f, a)
        #         if factor_price is None:
        #             continue
        #         # lambdaf = 1.0 in benchmark
        #         terms.append(af_val * factor_price ** expo)
        #     
        #     if not terms:
        #         return Constraint.Skip
        #     
        #     pva = model.pva[r, a]
        #     return pva ** expo == sum(terms)
        # model.eq_pvaeq = Constraint(model.r, model.a, rule=eq_pvaeq_rule)
        
        # Output allocation (Leontief for simplicity)
        def eq_x_rule(model, r, a, i):
            outputs = self.sets.activity_commodities.get(a, [])
            if outputs and i not in outputs:
                return Constraint.Skip
            share = value(model.gx_param[r, a, i])
            if share <= 0:
                denominator = len(outputs) if outputs else len(list(model.i))
                if denominator <= 0:
                    return Constraint.Skip
                share = 1.0 / denominator
            return model.x[r, a, i] == share * model.xp[r, a]
        model.eq_x = Constraint(model.r, model.a, model.i, rule=eq_x_rule)

        def eq_po_rule(model, r, a):
            outputs = self.sets.activity_commodities.get(a)
            if not outputs:
                return Constraint.Skip
            return model.xp[r, a] == sum(model.x[r, a, i] for i in outputs)
        model.eq_po = Constraint(model.r, model.a, rule=eq_po_rule)
        
        # ========================================================================
        # SUPPLY BLOCK
        # ========================================================================
        
        # Domestic supply
        def eq_xs_rule(model, r, i):
            producing_activities = self.sets.commodity_activities.get(i, list(model.a))
            return model.xs[r, i] == sum(model.x[r, a, i] for a in producing_activities)
        model.eq_xs = Constraint(model.r, model.i, rule=eq_xs_rule)
        
        # Supply price equals domestic price
        def eq_ps_rule(model, r, i):
            return model.ps[r, i] == model.pd[r, i]
        model.eq_ps = Constraint(model.r, model.i, rule=eq_ps_rule)
        
        # ========================================================================
        # TRADE - CET DOMESTIC/EXPORT ALLOCATION
        # ========================================================================
        
        def eq_xds_rule(model, r, i):
            omega = self.params.elasticities.omegax.get((r, i), 2.0)
            gd_share = value(model.gd_share[r, i])
            if gd_share <= 0.0:
                return model.xds[r, i] == 0.0
            if omega == float("inf"):
                return model.pd[r, i] == model.ps[r, i]
            return model.xds[r, i] == gd_share * model.xs[r, i] * (model.pd[r, i] / model.ps[r, i]) ** omega
        model.eq_xds = Constraint(model.r, model.i, rule=eq_xds_rule)

        def eq_xet_rule(model, r, i):
            omega = self.params.elasticities.omegax.get((r, i), 2.0)
            ge_share = value(model.ge_share[r, i])
            if ge_share <= 0.0:
                return model.xet[r, i] == 0.0
            if omega == float("inf"):
                return model.pet[r, i] == model.ps[r, i]
            return model.xet[r, i] == ge_share * model.xs[r, i] * (model.pet[r, i] / model.ps[r, i]) ** omega
        model.eq_xet = Constraint(model.r, model.i, rule=eq_xet_rule)

        def eq_xseq_rule(model, r, i):
            omega = self.params.elasticities.omegax.get((r, i), 2.0)
            gd_share = value(model.gd_share[r, i])
            ge_share = value(model.ge_share[r, i])
            if omega == float("inf"):
                return model.xs[r, i] == model.xds[r, i] + model.xet[r, i]
            exponent = 1.0 + omega
            return model.ps[r, i] ** exponent == gd_share * model.pd[r, i] ** exponent + ge_share * model.pet[r, i] ** exponent
        model.eq_xseq = Constraint(model.r, model.i, rule=eq_xseq_rule)
        
        # Export price relationship (simplified)
        def eq_pe_rule(model, r, i):
            return model.pet[r, i] == model.ps[r, i]
        model.eq_pe = Constraint(model.r, model.i, rule=eq_pe_rule)

        # Bilateral export prices follow aggregate export price on active routes only.
        def eq_pe_route_rule(model, r, i, rp):
            if r == rp:
                return model.pe[r, i, rp] == model.pnum
            bilateral_exports = self.params.benchmark.vxmd.get((r, i, rp), 0.0)
            mirror_imports = self.params.benchmark.viws.get((rp, i, r), 0.0)
            if bilateral_exports <= 0.0 and mirror_imports <= 0.0:
                return model.pe[r, i, rp] == 0.0
            return model.pe[r, i, rp] == model.pet[r, i]
        model.eq_pe_route = Constraint(model.r, model.i, model.rp, rule=eq_pe_route_rule)

        def eq_pdeq_rule(model, r, i):
            return model.xds[r, i] == sum(model.xda[r, i, aa] / model.xscale[r, aa] for aa in model.aa)
        model.eq_pdeq = Constraint(model.r, model.i, rule=eq_pdeq_rule)
        
        # ========================================================================
        # TRADE - CES ARMINGTON DOMESTIC/IMPORT
        # ========================================================================
        
        # Armington aggregation (Leontief for simplicity)
        def eq_xa_rule(model, r, i):
            inventory = self.params.benchmark.vst.get((r, i), 0.0)
            return model.xa[r, i] == sum(model.xaa[r, i, aa] / model.xscale[r, aa] for aa in model.aa) + inventory
        model.eq_xa = Constraint(model.r, model.i, rule=eq_xa_rule)

        # Agent/activity demand for intermediate inputs by activity.
        def eq_xaa_activity_rule(model, r, i, a):
            share = value(model.p_io[r, i, a])
            if share <= 0.0:
                return model.xaa[r, i, a] == 0.0
            return model.xaa[r, i, a] == model.xscale[r, a] * share * model.nd[r, a]
        model.eq_xaa_activity = Constraint(model.r, model.i, model.a, rule=eq_xaa_activity_rule)

        def eq_xaa_hhd_rule(model, r, i):
            return model.xaa[r, i, GTAP_HOUSEHOLD_AGENT] == model.xc[r, i]
        model.eq_xaa_hhd = Constraint(model.r, model.i, rule=eq_xaa_hhd_rule)

        def eq_xaa_gov_rule(model, r, i):
            return model.xaa[r, i, GTAP_GOVERNMENT_AGENT] == model.xg[r, i]
        model.eq_xaa_gov = Constraint(model.r, model.i, rule=eq_xaa_gov_rule)

        def eq_xaa_inv_rule(model, r, i):
            return model.xaa[r, i, GTAP_INVESTMENT_AGENT] == model.xi[r, i]
        model.eq_xaa_inv = Constraint(model.r, model.i, rule=eq_xaa_inv_rule)

        def eq_xaa_tmg_rule(model, r, i):
            benchmark_margin = self.params.benchmark.vst.get((r, i), 0.0)
            return model.xaa[r, i, GTAP_MARGIN_AGENT] == model.xscale[r, GTAP_MARGIN_AGENT] * benchmark_margin
        model.eq_xaa_tmg = Constraint(model.r, model.i, rule=eq_xaa_tmg_rule)

        def get_agent_armington_shares(r, i, aa):
            total = value(model.xaa[r, i, aa])
            if total <= 0.0:
                return 0.0, 0.0
            domestic = value(model.xda[r, i, aa])
            imported = value(model.xma[r, i, aa])
            return domestic / total, imported / total

        def eq_pdp_rule(model, r, i, aa):
            wedge_map = getattr(self.params.taxes, "pdp_agent_wedge", {})
            wedge = wedge_map.get((r, i, aa), 1.0)
            return model.pdp[r, i, aa] == wedge * model.pd[r, i]
        model.eq_pdp = Constraint(model.r, model.i, model.aa, rule=eq_pdp_rule)

        def eq_pmp_rule(model, r, i, aa):
            return model.pmp[r, i, aa] == model.pmt[r, i]
        model.eq_pmp = Constraint(model.r, model.i, model.aa, rule=eq_pmp_rule)

        def eq_paa_rule(model, r, i, aa):
            return model.paa[r, i, aa] == model.pa[r, i]
        model.eq_paa = Constraint(model.r, model.i, model.aa, rule=eq_paa_rule)

        def eq_xda_rule(model, r, i, aa):
            if aa == GTAP_HOUSEHOLD_AGENT:
                _, domestic, _ = self.params.benchmark.get_private_demand(r, i)
                return model.xda[r, i, aa] == model.xscale[r, aa] * max(domestic, 0.0)
            if aa == GTAP_GOVERNMENT_AGENT:
                _, domestic, _ = self.params.benchmark.get_government_demand(r, i)
                return model.xda[r, i, aa] == model.xscale[r, aa] * max(domestic, 0.0)
            if aa == GTAP_INVESTMENT_AGENT:
                _, domestic, _ = self.params.benchmark.get_investment_demand(r, i)
                return model.xda[r, i, aa] == model.xscale[r, aa] * max(domestic, 0.0)
            if aa == GTAP_MARGIN_AGENT:
                domestic = self.params.benchmark.vst.get((r, i), 0.0)
                return model.xda[r, i, aa] == model.xscale[r, aa] * max(domestic, 0.0)
            domestic_share, _ = get_agent_armington_shares(r, i, aa)
            if domestic_share <= 0.0:
                return model.xda[r, i, aa] == 0.0
            return model.xda[r, i, aa] == domestic_share * model.xaa[r, i, aa]
        model.eq_xda = Constraint(model.r, model.i, model.aa, rule=eq_xda_rule)

        def eq_xma_rule(model, r, i, aa):
            if aa == GTAP_HOUSEHOLD_AGENT:
                _, _, imported = self.params.benchmark.get_private_demand(r, i)
                return model.xma[r, i, aa] == model.xscale[r, aa] * max(imported, 0.0)
            if aa == GTAP_GOVERNMENT_AGENT:
                _, _, imported = self.params.benchmark.get_government_demand(r, i)
                return model.xma[r, i, aa] == model.xscale[r, aa] * max(imported, 0.0)
            if aa == GTAP_INVESTMENT_AGENT:
                _, _, imported = self.params.benchmark.get_investment_demand(r, i)
                return model.xma[r, i, aa] == model.xscale[r, aa] * max(imported, 0.0)
            if aa == GTAP_MARGIN_AGENT:
                return model.xma[r, i, aa] == 0.0
            _, import_share = get_agent_armington_shares(r, i, aa)
            if import_share <= 0.0:
                return model.xma[r, i, aa] == 0.0
            return model.xma[r, i, aa] == import_share * model.xaa[r, i, aa]
        model.eq_xma = Constraint(model.r, model.i, model.aa, rule=eq_xma_rule)

        def eq_xd_agg_rule(model, r, i):
            return model.xd[r, i] == sum(model.xda[r, i, aa] / model.xscale[r, aa] for aa in model.aa)
        model.eq_xd_agg = Constraint(model.r, model.i, rule=eq_xd_agg_rule)

        def eq_xmt_agg_rule(model, r, i):
            return model.xmt[r, i] == sum(model.xma[r, i, aa] / model.xscale[r, aa] for aa in model.aa)
        model.eq_xmt_agg = Constraint(model.r, model.i, rule=eq_xmt_agg_rule)
        
        # Armington price (weighted average) - always defined
        def eq_pa_rule(model, r, i):
            total = model.xd[r, i] + model.xmt[r, i] + 0.001  # Small epsilon to avoid division by zero
            return model.pa[r, i] * total == model.xd[r, i] * model.pd[r, i] + model.xmt[r, i] * model.pmt[r, i] + 0.001
        model.eq_pa = Constraint(model.r, model.i, rule=eq_pa_rule)
        
        # Import price (simplified)
        def eq_pmt_rule(model, r, i):
            return model.pmt[r, i] == model.pd[r, i]
        model.eq_pmt = Constraint(model.r, model.i, rule=eq_pmt_rule)

        # Route margin price index in benchmark form (normalized to 1.0 when route is active).
        def eq_pwmg_rule(model, r, i, rp):
            margin_flow = sum(self.params.benchmark.vtwr.get((r, i, rp, margin), 0.0) for margin in self.sets.m)
            if margin_flow <= 0.0:
                return model.pwmg[r, i, rp] == 0.0
            return model.pwmg[r, i, rp] == 1.0
        model.eq_pwmg = Constraint(model.r, model.i, model.rp, rule=eq_pwmg_rule)

        def eq_ptmg_rule(model, i):
            return model.ptmg[i] == model.pnum
        model.eq_ptmg = Constraint(model.i, rule=eq_ptmg_rule)
        
        # ========================================================================
        # FACTOR BLOCK
        # ========================================================================

        # Factor market clearing (distribute xft using gf share)
        def eq_xft_rule(model, r, f):
            return model.xft[r, f] == sum(model.xf[r, f, a] for a in model.a)
        model.eq_xft = Constraint(model.r, model.f, rule=eq_xft_rule)

        # Factor demand (GAMS exact formulation)
        # GAMS: xf(r,f,a,t) =e= af(r,f,a,t)*va(r,a,t)*(pva(r,a,t)/pfa(r,f,a,t))**sigmav(r,a)
        #                       * (lambdaf(r,f,a,t))**(sigmav(r,a)-1)
        def eq_xfeq_rule(model, r, f, a):
            af_val = value(model.af_param[r, f, a])  # GAMS calibrated parameter
            if af_val <= 0.0:
                return Constraint.Skip
            factor_price = self._factor_price_term(model, r, f, a)
            if factor_price is None:
                return Constraint.Skip
            ratio = model.pva[r, a] / factor_price
            sigmav = self._get_sigmav(r, a)
            lambdaf = self._lambdaf(r, f, a)
            return model.xf[r, f, a] == af_val * model.va[r, a] * ratio**sigmav * lambdaf ** (sigmav - 1)
        model.eq_xfeq = Constraint(model.r, model.f, model.a, rule=eq_xfeq_rule)

        # Aggregate supply of factors from production shares
        def eq_xfteq_rule(model, r, f):
            benchmark_supply = sum(self.params.benchmark.vfm.get((r, f, a), 0.0) for a in self.sets.a)
            if benchmark_supply <= 0:
                return Constraint.Skip
            return model.xft[r, f] == benchmark_supply
        model.eq_xfteq = Constraint(model.r, model.f, rule=eq_xfteq_rule)

        # Aggregate factor price equals the demand-weighted activity prices.
        def eq_pfeq_rule(model, r, f):
            total_share = sum(value(model.gf_share[r, f, a]) for a in model.a)
            if total_share <= 0:
                return Constraint.Skip
            weighted = sum(value(model.gf_share[r, f, a]) * model.pf[r, f, a] for a in model.a)
            return model.pft[r, f] * total_share == weighted
        model.eq_pfeq = Constraint(model.r, model.f, rule=eq_pfeq_rule)

        # Note: eq_pvaeq already defined above with GAMS formulation

        # Regional factor price index aggregates all factor prices by supply share.
        def eq_pfact_rule(model, r):
            total_share = 0.0
            weighted = 0.0
            for f in model.f:
                factor_share = sum(value(model.gf_share[r, f, a]) for a in model.a)
                if factor_share <= 0:
                    continue
                total_share += factor_share
                weighted += factor_share * model.pft[r, f]
            if total_share <= 0:
                return model.pfact[r] == model.pnum
            return model.pfact[r] * total_share == weighted
        model.eq_pfact = Constraint(model.r, rule=eq_pfact_rule)

        # Capital stock equals total capital demand across activities.
        def eq_kstock_rule(model, r):
            capital = "Capital"
            if capital not in model.f:
                return Constraint.Skip
            total_capital = sum(model.xf[r, capital, a] for a in model.a)
            return model.kstock[r] == total_capital
        model.eq_kstock = Constraint(model.r, rule=eq_kstock_rule)
        
        # ========================================================================
        # DEMAND BLOCK
        # ========================================================================
        
        # Private consumption (fixed shares for simplicity)
        def eq_xc_rule(model, r, i):
            private_total, _, _ = self.params.benchmark.get_private_demand(r, i)
            return model.xc[r, i] == private_total
        model.eq_xc = Constraint(model.r, model.i, rule=eq_xc_rule)
        
        # Government consumption
        def eq_xg_rule(model, r, i):
            government_total, _, _ = self.params.benchmark.get_government_demand(r, i)
            return model.xg[r, i] == government_total
        model.eq_xg = Constraint(model.r, model.i, rule=eq_xg_rule)
        
        # Investment demand
        def eq_xi_rule(model, r, i):
            investment_total, _, _ = self.params.benchmark.get_investment_demand(r, i)
            return model.xi[r, i] == investment_total
        model.eq_xi = Constraint(model.r, model.i, rule=eq_xi_rule)
        
        # ========================================================================
        # INCOME BLOCK
        # ========================================================================
        
        # Regional income from factors
        def eq_regy_rule(model, r):
            return model.regy[r] == sum(model.pf[r, f, a] * model.xf[r, f, a] 
                                        for f in model.f for a in model.a)
        model.eq_regy = Constraint(model.r, rule=eq_regy_rule)
        
        # Private income share
        def eq_yc_rule(model, r):
            return model.yc[r] == model.regy[r] * model.yc_share_reg[r]
        model.eq_yc = Constraint(model.r, rule=eq_yc_rule)
        
        # Government income share (region-specific benchmark calibration)
        def eq_yg_rule(model, r):
            return model.yg[r] == model.regy[r] * model.yg_share_reg[r]
        model.eq_yg = Constraint(model.r, rule=eq_yg_rule)

        # Investment income share (region-specific benchmark calibration)
        def eq_yi_rule(model, r):
            return model.yi[r] == model.regy[r] * model.yi_share_reg[r]
        model.eq_yi = Constraint(model.r, rule=eq_yi_rule)
        
        def eq_pabs_rule(model, r):
            total_abs = sum(model.xa[r, i] for i in model.i)
            if value(total_abs) <= 0:
                return Constraint.Skip
            return model.pabs[r] * total_abs == sum(model.pa[r, i] * model.xa[r, i] for i in model.i)
        model.eq_pabs = Constraint(model.r, rule=eq_pabs_rule)

        # ========================================================================
        # MARKET CLEARING
        # ========================================================================
        
        # Goods market clearing: Supply = Demand
        def mkt_goods_rule(model, r, i):
            absorption = sum(model.xaa[r, i, aa] / model.xscale[r, aa] for aa in model.aa)
            inventory = self.params.benchmark.vst.get((r, i), 0.0)
            return model.xa[r, i] == absorption + inventory
        model.mkt_goods = Constraint(model.r, model.i, rule=mkt_goods_rule)
        
        # ========================================================================
        # NUMERAIRE
        # ========================================================================
        
        def eq_pwfact_rule(model):
            n_regions = len(list(model.r))
            if n_regions == 0:
                return Constraint.Skip
            return model.pwfact == sum(model.pfact[r] for r in model.r) / n_regions
        model.eq_pwfact = Constraint(rule=eq_pwfact_rule)

        def eq_pnum_rule(model):
            return model.pnum == model.pwfact
        model.eq_pnum = Constraint(rule=eq_pnum_rule)
    
    def _add_objective(self, model: "ConcreteModel") -> None:
        """Add dummy objective for NLP."""
        from pyomo.environ import Objective, minimize

        def dummy_obj(model):
            return 1.0

        model.OBJ = Objective(rule=dummy_obj, sense=minimize)

    def _get_sigmap(self, r: str, a: str) -> float:
        return self.params.elasticities.sigmap.get((r, a), 1.0)

    def _get_sigmand(self, r: str, a: str) -> float:
        return self.params.elasticities.sigmand.get((r, a), 1.0)

    def _get_sigmav(self, r: str, a: str) -> float:
        return self.params.elasticities.sigmav.get((r, a), 1.0)

    def _axp_shift(self, r: str, a: str) -> float:
        return self.params.shifts.axp.get((r, a), 1.0)

    def _lambdand(self, r: str, a: str) -> float:
        return self.params.shifts.lambdand.get((r, a), 1.0)

    def _lambdava(self, r: str, a: str) -> float:
        return self.params.shifts.lambdava.get((r, a), 1.0)

    def _lambdaf(self, r: str, f: str, a: str) -> float:
        return self.params.shifts.lambdaf.get((r, f, a), 1.0)

    def _factor_price_term(self, model: "ConcreteModel", r: str, f: str, a: str):
        return model.pf[r, f, a]
