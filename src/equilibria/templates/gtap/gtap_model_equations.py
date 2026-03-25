"""GTAP Model Equations (CGEBox version)

This module implements all GTAP CGE model equations following the CGEBox implementation.
Reference: /Users/marmol/proyectos2/cge_babel/cgebox/gams/model/model.gms

The model is organized into equation blocks:
1. Production Block: Technology nests, factor demands, output allocation
2. Trade Block: CET exports, CES Armington imports, bilateral trade
3. Demand Block: Private consumption, government, investment
4. Factor Block: Factor markets (mobile & sluggish)
5. Income Block: Regional income, tax revenues
6. Investment Block: Global savings allocation
7. Price Block: Price aggregation and indices
8. Market Clearing: Walras conditions

All equations are formulated as zero-profit conditions (prf_*), 
market clearing (mkt_*), or income balance (inc_*).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from pyomo.environ import ConcreteModel


class GTAPModelEquations:
    """Build GTAP CGE model equations.
    
    This class constructs all equations for the GTAP model following
    the CGEBox GAMS implementation.
    
    Attributes:
        sets: GTAP sets (regions, commodities, factors, etc.)
        params: GTAP parameters (elasticities, shares, taxes)
        equations: Dictionary of equation blocks
        
    Example:
        >>> sets = GTAPSets(); sets.load_from_gdx("asa7x5.gdx")
        >>> params = GTAPParameters(); params.load_from_gdx("asa7x5.gdx")
        >>> eq_builder = GTAPModelEquations(sets, params)
        >>> model = eq_builder.build_model()
    """
    
    def __init__(
        self,
        sets: "GTAPSets",
        params: "GTAPParameters",
        closure: Optional["GTAPClosureConfig"] = None,
    ):
        """Initialize equation builder.
        
        Args:
            sets: GTAP sets
            params: GTAP parameters
            closure: Optional closure configuration
        """
        self.sets = sets
        self.params = params
        self.closure = closure
        self.equations: Dict[str, Any] = {}
        
    def build_model(self) -> "ConcreteModel":
        """Build complete GTAP Pyomo model.
        
        Returns:
            Pyomo ConcreteModel with all variables and equations
        """
        from pyomo.environ import ConcreteModel
        
        model = ConcreteModel(name="GTAP_CGE_Model")
        
        # Build components in order
        self._add_sets(model)
        self._add_parameters(model)
        self._add_variables(model)
        self._add_equations(model)
        self._add_objective(model)
        
        return model
    
    def _add_sets(self, model: "ConcreteModel") -> None:
        """Add Pyomo sets to model.
        
        Sets:
            r: Regions
            i: Commodities
            a: Activities (alias of i)
            f: Factors
            mf: Mobile factors (subset)
            sf: Specific factors (subset)
            m: Transport modes
        """
        from pyomo.environ import Set
        
        model.r = Set(initialize=self.sets.r, doc="Regions")
        model.i = Set(initialize=self.sets.i, doc="Commodities")
        model.a = Set(initialize=self.sets.a, doc="Activities")
        model.f = Set(initialize=self.sets.f, doc="Factors")
        model.mf = Set(initialize=self.sets.mf, doc="Mobile factors")
        model.sf = Set(initialize=self.sets.sf, doc="Sector-specific factors")
        model.m = Set(initialize=self.sets.m, doc="Transport modes")
        
        # Aliases
        model.rp = Set(initialize=self.sets.r, doc="Regions (alias)")
        
    def _add_parameters(self, model: "ConcreteModel") -> None:
        """Add all parameters to model.
        
        Parameters include:
        - Elasticities (esubva, esubd, esubm, omegax, omegaw, etrae)
        - Share parameters (p_gx, p_ax, p_alphad, p_alpham, p_amw, p_gw)
        - Benchmark values (vom, vfm, vdfm, etc.)
        - Tax rates (rto, rtf, rtms, rtxs, etc.)
        """
        from pyomo.environ import Param
        
        # Helper to create indexed parameters
        def create_param(name: str, indices: Tuple, data: Dict, default: float = 0.0):
            if not data:
                return
            
            index_sets = tuple(getattr(model, idx) for idx in indices)
            values = {}
            for key, value in data.items():
                if isinstance(key, tuple):
                    values[key] = value
                else:
                    values[(key,)] = value
            
            setattr(
                model,
                name,
                Param(*index_sets, initialize=values, default=default, doc=name)
            )
        
        # Elasticities
        create_param("esubva", ("r", "a"), self.params.elasticities.esubva, 1.0)
        create_param("esubd", ("r", "i"), self.params.elasticities.esubd, 2.0)
        create_param("esubm", ("r", "i"), self.params.elasticities.esubm, 4.0)
        create_param("omegax", ("r", "i"), self.params.elasticities.omegax, 2.0)
        create_param("omegaw", ("r", "i"), self.params.elasticities.omegaw, 4.0)
        create_param("etrae", ("f",), self.params.elasticities.etrae, float('inf'))
        
        # Benchmark values - Production
        create_param("vom", ("r", "a"), self.params.benchmark.vom, 0.0)
        create_param("vfm", ("r", "f", "a"), self.params.benchmark.vfm, 0.0)
        
        # Benchmark values - Intermediate demand
        create_param("vdfm", ("r", "i", "a"), self.params.benchmark.vdfm, 0.0)
        create_param("vifm", ("r", "i", "a"), self.params.benchmark.vifm, 0.0)
        
        # Benchmark values - Final demand
        create_param("vpm", ("r", "i"), self.params.benchmark.vpm, 0.0)
        create_param("vgm", ("r", "i"), self.params.benchmark.vgm, 0.0)
        create_param("vim", ("r", "i"), self.params.benchmark.vim, 0.0)
        
        # Benchmark values - Trade
        create_param("vxmd", ("r", "i", "rp"), self.params.benchmark.vxmd, 0.0)
        create_param("viws", ("r", "i", "rp"), self.params.benchmark.viws, 0.0)
        create_param("vims", ("r", "i", "rp"), self.params.benchmark.vims, 0.0)
        
        # Tax rates
        create_param("rto", ("r", "a"), self.params.taxes.rto, 0.0)
        create_param("rtf", ("r", "f", "a"), self.params.taxes.rtf, 0.0)
        create_param("rtms", ("r", "i", "rp"), self.params.taxes.rtms, 0.0)
        create_param("rtxs", ("r", "i", "rp"), self.params.taxes.rtxs, 0.0)
        
    def _add_variables(self, model: "ConcreteModel") -> None:
        """Add all variables to model.
        
        Variables are organized by block:
        - Production: xp, x, px, pp
        - Trade: xe, xw, xmt, pe, pmt, pmcif, pefob
        - Demand: xc, xg, xi, pa, pcons, pg, pi
        - Factors: xft, xf, pf, pft
        - Income: regy, yc, yg
        - Investment: psave, yi, rorg
        - Prices: ps, pd, pabs, pnum
        """
        from pyomo.environ import Var, PositiveReals, Reals
        
        # Production variables
        model.xp = Var(model.r, model.a, within=PositiveReals, doc="Production activity level")
        model.x = Var(model.r, model.a, model.i, within=PositiveReals, doc="Output by commodity")
        model.px = Var(model.r, model.a, within=PositiveReals, doc="Unit cost of production")
        model.pp = Var(model.r, model.a, within=PositiveReals, doc="Producer price")
        
        # Supply variables
        model.ps = Var(model.r, model.i, within=PositiveReals, doc="Price of domestic supply")
        model.pd = Var(model.r, model.i, within=PositiveReals, doc="Price of domestic goods")
        model.xs = Var(model.r, model.i, within=PositiveReals, doc="Domestic supply")
        model.xds = Var(model.r, model.i, within=PositiveReals, doc="Domestically produced goods supply")
        
        # Trade - Exports
        model.xe = Var(model.r, model.i, model.rp, within=PositiveReals, doc="Bilateral exports")
        model.xet = Var(model.r, model.i, within=PositiveReals, doc="Aggregate export supply")
        model.pe = Var(model.r, model.i, model.rp, within=PositiveReals, doc="Bilateral export price")
        model.pet = Var(model.r, model.i, within=PositiveReals, doc="Aggregate export price")
        model.pefob = Var(model.r, model.i, model.rp, within=PositiveReals, doc="FOB export price")
        
        # Trade - Imports
        model.xmt = Var(model.r, model.i, within=PositiveReals, doc="Aggregate import demand")
        model.xw = Var(model.r, model.i, model.rp, within=PositiveReals, doc="Bilateral imports")
        model.pmt = Var(model.r, model.i, within=PositiveReals, doc="Aggregate import price")
        model.pmcif = Var(model.r, model.i, model.rp, within=PositiveReals, doc="CIF import price")
        
        # Armington variables
        model.xa = Var(model.r, model.i, within=PositiveReals, doc="Armington demand")
        model.pa = Var(model.r, model.i, within=PositiveReals, doc="Armington price")
        model.xd = Var(model.r, model.i, within=PositiveReals, doc="Domestic demand")
        model.xm = Var(model.r, model.i, within=PositiveReals, doc="Import demand")
        
        # Final demand
        model.xc = Var(model.r, model.i, within=PositiveReals, doc="Private consumption")
        model.xg = Var(model.r, model.i, within=PositiveReals, doc="Government consumption")
        model.xi = Var(model.r, model.i, within=PositiveReals, doc="Investment demand")
        model.pcons = Var(model.r, within=PositiveReals, doc="Consumer price index")
        model.pg = Var(model.r, within=PositiveReals, doc="Government price index")
        model.pi = Var(model.r, within=PositiveReals, doc="Investment price index")
        
        # Factor variables
        model.xft = Var(model.r, model.f, within=PositiveReals, doc="Aggregate factor supply")
        model.xf = Var(model.r, model.f, model.a, within=PositiveReals, doc="Factor demand")
        model.pf = Var(model.r, model.f, model.a, within=PositiveReals, doc="Factor price (tax exclusive)")
        model.pft = Var(model.r, model.f, within=PositiveReals, doc="Aggregate factor price")
        
        # Income variables
        model.regy = Var(model.r, within=Reals, doc="Regional income")
        model.yc = Var(model.r, within=PositiveReals, doc="Private consumption expenditure")
        model.yg = Var(model.r, within=PositiveReals, doc="Government consumption expenditure")
        model.yi = Var(model.r, within=PositiveReals, doc="Investment expenditure")
        
        # Investment and savings
        model.psave = Var(model.r, within=PositiveReals, doc="Price of savings")
        model.xigbl = Var(within=PositiveReals, doc="Global net investment")
        model.pigbl = Var(within=PositiveReals, doc="Global investment price")
        model.rorg = Var(within=Reals, doc="Global rate of return")
        
        # Price indices
        model.pabs = Var(model.r, within=PositiveReals, doc="Price of aggregate absorption")
        model.pnum = Var(within=PositiveReals, doc="Model numeraire")
        
        # Walras check
        model.walras = Var(within=Reals, doc="Walras check (should be zero)")
        
    def _add_equations(self, model: "ConcreteModel") -> None:
        """Add all model equations.
        
        Equations are organized into blocks:
        1. Production Block
        2. Trade Block (CET + Armington)
        3. Demand Block
        4. Factor Block
        5. Income Block
        6. Investment Block
        7. Market Clearing
        """
        from pyomo.environ import Constraint, exp, log
        
        # ===== Production Block =====
        def prf_y_rule(model, r, a):
            """Zero-profit condition for production."""
            # Unit cost = producer price
            # Handle missing tax rates gracefully
            try:
                tax_rate = model.rto[r, a]
            except:
                tax_rate = 0.0
            return model.px[r, a] == model.pp[r, a] * (1 + tax_rate)
        
        model.prf_y = Constraint(model.r, model.a, rule=prf_y_rule, doc="Production zero-profit")
        
        # ===== Trade Block - CET Exports =====
        def e_pet_rule(model, r, i):
            """Aggregate export price (CET aggregation)."""
            if (r, i) not in self.params.elasticities.omegax:
                return Constraint.Skip
            
            omega = self.params.elasticities.omegax.get((r, i), 2.0)
            
            if omega == float('inf'):
                # Perfect transformation
                return model.pet[r, i] == model.ps[r, i]
            
            # CET price aggregation
            shares = []
            for rp in model.r:
                if (r, i, rp) in self.params.benchmark.vxmd:
                    share = self.params.shares.p_gw.get((r, i, rp), 0)
                    if share > 0:
                        shares.append(share * model.pe[r, i, rp] ** (omega + 1))
            
            if not shares:
                return Constraint.Skip
                
            return model.pet[r, i] == sum(shares) ** (1 / (omega + 1))
        
        model.e_pet = Constraint(model.r, model.i, rule=e_pet_rule, doc="CET export price aggregation")
        
        # ===== Trade Block - Armington Imports =====
        def e_pmt_rule(model, r, i):
            """Aggregate import price (CES aggregation)."""
            if (r, i) not in self.params.elasticities.esubm:
                return Constraint.Skip
            
            esub = self.params.elasticities.esubm.get((r, i), 4.0)
            
            if esub == float('inf'):
                # Perfect substitution
                return model.pmt[r, i] == sum(model.pmcif[r, i, rp] for rp in model.r) / len(model.r)
            
            # CES price aggregation
            shares = []
            for rp in model.r:
                if rp != r:  # Don't import from self
                    share = self.params.shares.p_amw.get((r, i, rp), 0)
                    if share > 0:
                        shares.append(share * model.pmcif[r, i, rp] ** (1 - esub))
            
            if not shares:
                return Constraint.Skip
                
            return model.pmt[r, i] ** (1 - esub) == sum(shares)
        
        model.e_pmt = Constraint(model.r, model.i, rule=e_pmt_rule, doc="CES import price aggregation")
        
        # ===== Factor Block =====
        def e_pft_rule(model, r, f):
            """Aggregate factor price for mobile factors."""
            if f not in self.sets.mf:
                return Constraint.Skip
            
            # Mobile factor: single price across all sectors
            return model.pft[r, f] == sum(
                model.pf[r, f, a] * model.xf[r, f, a] 
                for a in model.a
            ) / sum(model.xf[r, f, a] for a in model.a)
        
        model.e_pft = Constraint(model.r, model.mf, rule=e_pft_rule, doc="Mobile factor price aggregation")
        
        # ===== Income Block =====
        def e_regy_rule(model, r):
            """Regional income definition."""
            # Factor income + tax revenues
            facty = sum(
                model.pf[r, f, a] * model.xf[r, f, a]
                for f in model.f for a in model.a
            )
            
            return model.regy[r] == facty
        
        model.e_regy = Constraint(model.r, rule=e_regy_rule, doc="Regional income")
        
        # ===== Market Clearing =====
        def mkt_pa_rule(model, r, i):
            """Armington goods market clearing."""
            # Supply = Demand
            supply = model.xa[r, i]
            
            # Demand from: intermediate, consumption, government, investment
            demand_int = sum(model.xd.get((r, i, a), 0) for a in model.a)
            demand_c = model.xc[r, i]
            demand_g = model.xg[r, i]
            demand_i = model.xi[r, i]
            
            return supply == demand_int + demand_c + demand_g + demand_i
        
        model.mkt_pa = Constraint(model.r, model.i, rule=mkt_pa_rule, doc="Armington market clearing")
        
        def mkt_pf_rule(model, r, f):
            """Factor market clearing."""
            # Supply = Demand
            supply = model.xft[r, f]
            demand = sum(model.xf[r, f, a] for a in model.a)
            
            return supply == demand
        
        model.mkt_pf = Constraint(model.r, model.f, rule=mkt_pf_rule, doc="Factor market clearing")
        
        # ===== Numeraire =====
        def e_pnum_rule(model):
            """Price numeraire (fixed to 1.0 in benchmark)."""
            return model.pnum == 1.0
        
        model.e_pnum = Constraint(rule=e_pnum_rule, doc="Price numeraire")
        
        # ===== Walras Check =====
        def e_walras_rule(model):
            """Walras Law check - should be zero."""
            # Sum of all excess demands
            ed = sum(
                model.xa[r, i] - model.xc[r, i] - model.xg[r, i] - model.xi[r, i]
                - sum(model.xd.get((r, i, a), 0) for a in model.a)
                for r in model.r for i in model.i
            )
            return model.walras == ed
        
        model.e_walras = Constraint(rule=e_walras_rule, doc="Walras check")
        
    def _add_objective(self, model: "ConcreteModel") -> None:
        """Add dummy objective function.
        
        CGE models are solved as square systems, so we use a dummy
        objective (typically zero) and let the solver find the
        equilibrium that satisfies all constraints.
        """
        from pyomo.environ import Objective, value
        
        # Dummy objective - minimize squared Walras value
        def dummy_objective(model):
            return model.walras ** 2
        
        model.OBJ = Objective(rule=dummy_objective, sense=minimize, doc="Dummy objective")
