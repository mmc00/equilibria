"""GTAP v6.2 model equations — Pyomo construction.

Builds the v6.2 CGE model as a Pyomo :class:`ConcreteModel`. Mirrors
the structure of :class:`equilibria.templates.gtap.GTAPModelEquations`
(v7) but is much smaller because v6.2 has:

- No ACT/COMM split (single index ``i ∈ TRAD_COMM``).
- No intermediate bundle (Leontief implicit).
- No MAKE transformation (diagonal make matrix).
- Cobb-Douglas government and trade-margin demand.
- Commodity-level factor markets (no ``tinc(e,a,r)``).
- Investment as a producing sector ``cgds``, not an explicit agent.

Phase 2a status: sets, parameters, and variables are wired into Pyomo;
the calibrated tax rates and aggregates from
:mod:`gtap_v62_calibration` are exposed as Pyomo Params; variable
initial values use the benchmark SAM. The :meth:`_add_equations` step
is a placeholder — Phase 2b and 2c will fill in the actual Constraints.

Reference docs:
    - ``runs/gtap_v62_vs_v7/README.md`` (structural diff)
    - ``runs/gtap_v62_vs_v7/notation_crosswalk.md`` (Table 1 mapping)
    - ``C:\\runGTAP375\\gtap.tab`` (Hertel/Itakura/McDougall 2003)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from equilibria.templates.gtap_v62.gtap_v62_calibration import (
    DerivedV62Calibration,
    derive_calibration,
)
from equilibria.templates.gtap_v62.gtap_v62_contract import GTAPv62ClosureConfig
from equilibria.templates.gtap_v62.gtap_v62_parameters import GTAPv62Parameters
from equilibria.templates.gtap_v62.gtap_v62_sets import GTAPv62Sets

if TYPE_CHECKING:
    from pyomo.environ import ConcreteModel

logger = logging.getLogger(__name__)


# v6.2 government tax stream is a single aggregate (v7 splits into 10).
GTAP_V62_TAX_STREAMS = ("imptx", "exptx", "outtx", "indtx", "facttx", "subdy")


class GTAPv62ModelEquations:
    """Build a Pyomo model for GTAP v6.2.

    Args:
        sets: Loaded :class:`GTAPv62Sets`.
        params: Loaded :class:`GTAPv62Parameters`.
        closure: Closure configuration (defaults to the standard v6.2
            closure with ``pgdpwld`` numeraire).
        derived: Pre-computed :class:`DerivedV62Calibration`. If not
            supplied, :func:`derive_calibration` is invoked at
            construction time.

    Example::

        sets = GTAPv62Sets(); sets.load_from_har(...)
        params = GTAPv62Parameters(); params.load_from_har(..., sets=sets)
        eqs = GTAPv62ModelEquations(sets, params)
        model = eqs.build_model()
        assert hasattr(model, 'qo')
        assert hasattr(model, 'ps')
    """

    def __init__(
        self,
        sets: GTAPv62Sets,
        params: GTAPv62Parameters,
        closure: Optional[GTAPv62ClosureConfig] = None,
        derived: Optional[DerivedV62Calibration] = None,
    ) -> None:
        self.sets = sets
        self.params = params
        self.closure = closure or GTAPv62ClosureConfig()
        self.derived = derived if derived is not None else derive_calibration(sets, params)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build_model(self) -> "ConcreteModel":
        """Construct the full Pyomo model.

        At Phase 2a, the model has all sets, parameters, and variables
        but no equations. Subsequent phases populate
        :meth:`_add_equations`.
        """
        from pyomo.environ import ConcreteModel

        model = ConcreteModel(name="GTAP_v6.2_Model")
        self._add_sets(model)
        self._add_parameters(model)
        self._add_variables(model)
        self._add_equations(model)
        return model

    # ------------------------------------------------------------------
    # Sets
    # ------------------------------------------------------------------

    def _add_sets(self, model: "ConcreteModel") -> None:
        """Declare Pyomo sets matching v6.2 set conventions."""
        from pyomo.environ import Set

        model.r = Set(initialize=self.sets.r, doc="Regions (REG)")
        model.i = Set(initialize=self.sets.i, doc="Traded commodities (TRAD_COMM)")
        model.f = Set(initialize=self.sets.f, doc="Primary factors (ENDW_COMM)")
        model.mf = Set(initialize=self.sets.mf, doc="Mobile factors")
        model.sf = Set(initialize=self.sets.sf, doc="Sluggish factors")
        model.cgds = Set(initialize=self.sets.cgds, doc="Capital goods (CGDS_COMM)")
        model.marg = Set(initialize=self.sets.marg, doc="Margin commodities")
        # PROD_COMM = TRAD_COMM ∪ CGDS_COMM — used as the sector index
        # in many production/intermediate equations.
        model.j = Set(initialize=self.sets.prod_comm, doc="Producing sectors (PROD_COMM)")
        # Tax streams (single-aggregate per region in v6.2)
        model.gy = Set(initialize=GTAP_V62_TAX_STREAMS, doc="Tax revenue streams")
        # Aliases for bilateral trade
        model.s = Set(initialize=self.sets.r, doc="Regions (source alias)")
        model.rp = Set(initialize=self.sets.r, doc="Regions (destination alias)")

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _add_parameters(self, model: "ConcreteModel") -> None:
        """Declare Pyomo Params with benchmark and derived values."""
        from pyomo.environ import Param

        e = self.params.elasticities
        b = self.params.benchmark
        c = self.derived

        # --- Elasticities (commodity / sector / factor / region indexed) ----

        model.esubd = Param(
            model.i,
            initialize=dict(e.esubd),
            mutable=True,
            doc="Top Armington elasticity (ESBD)",
        )
        model.esubm = Param(
            model.i,
            initialize=dict(e.esubm),
            mutable=True,
            doc="Bottom Armington elasticity (ESBM)",
        )
        model.esubt = Param(
            model.j,
            initialize=dict(e.esubt),
            mutable=True,
            doc="Top production-nest elasticity (ESBT). v6.2 default 0 = Leontief.",
        )
        model.esubva = Param(
            model.j,
            initialize=dict(e.esubva),
            mutable=True,
            doc="Value-added nest elasticity (ESBV)",
        )
        model.etrae = Param(
            model.f,
            initialize=dict(e.etrae),
            mutable=True,
            doc="Sluggish factor transformation elasticity (ETRE). -1.0 means inf/mobile.",
        )
        model.rorflex = Param(
            model.r,
            initialize=dict(e.rorflex),
            mutable=True,
            doc="Rate-of-return flexibility (RFLX)",
        )
        # CDE parameters for household demand
        model.incpar = Param(
            model.i, model.r,
            initialize=dict(e.incpar),
            mutable=True,
            doc="CDE expansion parameter (INCP)",
        )
        model.subpar = Param(
            model.i, model.r,
            initialize=dict(e.subpar),
            mutable=True,
            doc="CDE substitution parameter (SUBP)",
        )

        # --- Benchmark value-flow parameters (V** headers) ------------------

        # Use the indexing convention from each HAR header. Some are
        # (factor, sector, region), some (commodity, sector, region), etc.
        # Defaults to 0.0 for missing keys.
        model.vfm = Param(
            model.f, model.j, model.r,
            initialize={k: v for k, v in b.vfm.items()},
            default=0.0, mutable=True,
            doc="Factor purchases at market prices",
        )
        model.evfa = Param(
            model.f, model.j, model.r,
            initialize={k: v for k, v in b.evfa.items()},
            default=0.0, mutable=True,
            doc="Factor purchases at agent prices",
        )
        model.evoa = Param(
            model.f, model.r,
            initialize={k: v for k, v in b.evoa.items()},
            default=0.0, mutable=True,
            doc="Factor output at agent prices",
        )
        model.vdfm = Param(
            model.i, model.j, model.r,
            initialize={k: v for k, v in b.vdfm.items()},
            default=0.0, mutable=True,
            doc="Firm domestic intermediates at market prices",
        )
        model.vdfa = Param(
            model.i, model.j, model.r,
            initialize={k: v for k, v in b.vdfa.items()},
            default=0.0, mutable=True,
            doc="Firm domestic intermediates at agent prices",
        )
        model.vifm = Param(
            model.i, model.j, model.r,
            initialize={k: v for k, v in b.vifm.items()},
            default=0.0, mutable=True,
            doc="Firm imported intermediates at market prices",
        )
        model.vifa = Param(
            model.i, model.j, model.r,
            initialize={k: v for k, v in b.vifa.items()},
            default=0.0, mutable=True,
            doc="Firm imported intermediates at agent prices",
        )
        model.vdpm = Param(
            model.i, model.r,
            initialize={k: v for k, v in b.vdpm.items()},
            default=0.0, mutable=True,
            doc="Household domestic at market prices",
        )
        model.vdpa = Param(
            model.i, model.r,
            initialize={k: v for k, v in b.vdpa.items()},
            default=0.0, mutable=True,
            doc="Household domestic at agent prices",
        )
        model.vipm = Param(
            model.i, model.r,
            initialize={k: v for k, v in b.vipm.items()},
            default=0.0, mutable=True,
            doc="Household imports at market prices",
        )
        model.vipa = Param(
            model.i, model.r,
            initialize={k: v for k, v in b.vipa.items()},
            default=0.0, mutable=True,
            doc="Household imports at agent prices",
        )
        model.vdgm = Param(
            model.i, model.r,
            initialize={k: v for k, v in b.vdgm.items()},
            default=0.0, mutable=True,
            doc="Government domestic at market prices",
        )
        model.vdga = Param(
            model.i, model.r,
            initialize={k: v for k, v in b.vdga.items()},
            default=0.0, mutable=True,
            doc="Government domestic at agent prices",
        )
        model.vigm = Param(
            model.i, model.r,
            initialize={k: v for k, v in b.vigm.items()},
            default=0.0, mutable=True,
            doc="Government imports at market prices",
        )
        model.viga = Param(
            model.i, model.r,
            initialize={k: v for k, v in b.viga.items()},
            default=0.0, mutable=True,
            doc="Government imports at agent prices",
        )
        # Bilateral trade (3-d: commodity, source, destination)
        model.vxmd = Param(
            model.i, model.s, model.rp,
            initialize={k: v for k, v in b.vxmd.items()},
            default=0.0, mutable=True,
            doc="Bilateral exports at market prices (FOB)",
        )
        model.vxwd = Param(
            model.i, model.s, model.rp,
            initialize={k: v for k, v in b.vxwd.items()},
            default=0.0, mutable=True,
            doc="Bilateral exports at world prices",
        )
        model.vims = Param(
            model.i, model.s, model.rp,
            initialize={k: v for k, v in b.vims.items()},
            default=0.0, mutable=True,
            doc="Bilateral imports at market prices",
        )
        model.viws = Param(
            model.i, model.s, model.rp,
            initialize={k: v for k, v in b.viws.items()},
            default=0.0, mutable=True,
            doc="Bilateral imports at world prices",
        )
        # Margins
        model.vst = Param(
            model.marg, model.r,
            initialize={k: v for k, v in b.vst.items()},
            default=0.0, mutable=True,
            doc="Margin sales (VST)",
        )
        model.vtwr = Param(
            model.marg, model.i, model.s, model.rp,
            initialize={k: v for k, v in b.vtwr.items()},
            default=0.0, mutable=True,
            doc="Transport margins (VTWR)",
        )
        # Region aggregates
        model.vkb = Param(model.r, initialize=dict(b.vkb), default=0.0, mutable=True,
                          doc="Beginning capital stock")
        model.vdep = Param(model.r, initialize=dict(b.vdep), default=0.0, mutable=True,
                           doc="Depreciation")
        model.save_p = Param(model.r, initialize=dict(b.save), default=0.0, mutable=True,
                             doc="Savings (SAVE header)")

        # --- Derived calibration: tax rates and aggregate values ------------

        model.tfd = Param(
            model.i, model.j, model.r,
            initialize=dict(c.tfd), default=0.0, mutable=True,
            doc="Firm domestic intermediate tax rate (derived)",
        )
        model.tfi = Param(
            model.i, model.j, model.r,
            initialize=dict(c.tfi), default=0.0, mutable=True,
            doc="Firm imported intermediate tax rate (derived)",
        )
        model.tpd = Param(
            model.i, model.r,
            initialize=dict(c.tpd), default=0.0, mutable=True,
            doc="Private domestic tax rate (derived)",
        )
        model.tpi = Param(
            model.i, model.r,
            initialize=dict(c.tpi), default=0.0, mutable=True,
            doc="Private imported tax rate (derived)",
        )
        model.tgd = Param(
            model.i, model.r,
            initialize=dict(c.tgd), default=0.0, mutable=True,
            doc="Government domestic tax rate (derived)",
        )
        model.tgi = Param(
            model.i, model.r,
            initialize=dict(c.tgi), default=0.0, mutable=True,
            doc="Government imported tax rate (derived)",
        )
        model.tf = Param(
            model.f, model.j, model.r,
            initialize=dict(c.tf), default=0.0, mutable=True,
            doc="Factor tax rate (derived)",
        )
        model.tms = Param(
            model.i, model.s, model.rp,
            initialize=dict(c.tms), default=0.0, mutable=True,
            doc="Bilateral import tariff rate (derived)",
        )
        model.txs = Param(
            model.i, model.s, model.rp,
            initialize=dict(c.txs), default=0.0, mutable=True,
            doc="Bilateral export tax rate (derived)",
        )
        model.to = Param(
            model.j, model.r,
            initialize=dict(c.to), default=0.0, mutable=True,
            doc="Output tax rate (derived)",
        )
        # Aggregate value-flow benchmarks
        model.vom = Param(
            model.j, model.r,
            initialize=dict(c.vom), default=0.0, mutable=True,
            doc="Output at market prices (derived)",
        )
        model.evom = Param(
            model.f, model.r,
            initialize=dict(c.evom), default=0.0, mutable=True,
            doc="Factor income at market prices (derived)",
        )
        model.va_total = Param(
            model.j, model.r,
            initialize=dict(c.va_total), default=0.0, mutable=True,
            doc="Value-added at market prices (derived)",
        )

        # --- Production-block calibrated CES shares -----------------------
        # Top nest (CES between VA and intermediate composite)
        model.share_va = Param(
            model.j, model.r,
            initialize=dict(c.share_va), default=0.0, mutable=False,
            doc="Top nest: VA share of production cost (calibrated)",
        )
        model.share_int = Param(
            model.i, model.j, model.r,
            initialize=dict(c.share_int), default=0.0, mutable=False,
            doc="Top nest: intermediate i share of production cost",
        )
        # VA nest (CES across factors)
        model.share_fac = Param(
            model.f, model.j, model.r,
            initialize=dict(c.share_fac), default=0.0, mutable=False,
            doc="VA nest: factor f share of value-added",
        )
        # Top Armington (CES between domestic and imported per intermediate)
        model.share_dom = Param(
            model.i, model.j, model.r,
            initialize=dict(c.share_dom), default=0.0, mutable=False,
            doc="Top Armington: domestic VALUE share at benchmark",
        )
        model.share_imp = Param(
            model.i, model.j, model.r,
            initialize=dict(c.share_imp), default=0.0, mutable=False,
            doc="Top Armington: imported VALUE share at benchmark",
        )
        # Distribution parameters: absorb benchmark agent-price ratios
        # so the CES first-order conditions hold identically when
        # pfd_0 ≠ pfm_0 (i.e. when there are tax wedges).
        model.alpha_dom = Param(
            model.i, model.j, model.r,
            initialize=dict(c.alpha_dom), default=0.0, mutable=False,
            doc="Top Armington: domestic distribution parameter (calibrated)",
        )
        model.alpha_imp = Param(
            model.i, model.j, model.r,
            initialize=dict(c.alpha_imp), default=0.0, mutable=False,
            doc="Top Armington: imported distribution parameter (calibrated)",
        )
        # Benchmark CES composite price (used for diagnostics; the model
        # solves for pf_int endogenously).
        model.pf_int_0 = Param(
            model.i, model.j, model.r,
            initialize=dict(c.pf_int_0), default=1.0, mutable=False,
            doc="Top Armington: benchmark composite Armington price",
        )

        # --- Phase 2c.1 — Household & government calibration --------------
        model.alpha_dom_hhd = Param(
            model.i, model.r,
            initialize=dict(c.alpha_dom_hhd), default=0.0, mutable=False,
            doc="Household Armington: domestic distribution parameter",
        )
        model.alpha_imp_hhd = Param(
            model.i, model.r,
            initialize=dict(c.alpha_imp_hhd), default=0.0, mutable=False,
            doc="Household Armington: imported distribution parameter",
        )
        model.pp_0 = Param(
            model.i, model.r,
            initialize=dict(c.pp_0), default=1.0, mutable=False,
            doc="Household Armington: benchmark composite price",
        )
        model.share_hhd_cd = Param(
            model.i, model.r,
            initialize=dict(c.share_hhd_cd), default=0.0, mutable=False,
            doc="Household CD budget share on good i in r",
        )
        model.alpha_dom_gov = Param(
            model.i, model.r,
            initialize=dict(c.alpha_dom_gov), default=0.0, mutable=False,
            doc="Government Armington: domestic distribution parameter",
        )
        model.alpha_imp_gov = Param(
            model.i, model.r,
            initialize=dict(c.alpha_imp_gov), default=0.0, mutable=False,
            doc="Government Armington: imported distribution parameter",
        )
        model.pg_0 = Param(
            model.i, model.r,
            initialize=dict(c.pg_0), default=1.0, mutable=False,
            doc="Government Armington: benchmark composite price",
        )
        model.share_gov_cd = Param(
            model.i, model.r,
            initialize=dict(c.share_gov_cd), default=0.0, mutable=False,
            doc="Government CD budget share on good i in r",
        )

        # --- Phase 2c.2 — Trade and margins calibration -------------------
        model.alpha_xs = Param(
            model.i, model.s, model.rp,
            initialize=dict(c.alpha_xs), default=0.0, mutable=False,
            doc="Bottom Armington: source distribution parameter (calibrated)",
        )
        model.pim_0 = Param(
            model.i, model.r,
            initialize=dict(c.pim_0), default=1.0, mutable=False,
            doc="Composite import benchmark price",
        )
        model.qim_0 = Param(
            model.i, model.r,
            initialize=dict(c.qim_0), default=0.0, mutable=False,
            doc="Composite import benchmark quantity",
        )
        # Per-shipment margin shares (amgm) and source-of-margin shares (share_st)
        model.amgm = Param(
            model.marg, model.i, model.s, model.rp,
            initialize=dict(c.amgm), default=0.0, mutable=False,
            doc="Per-shipment margin cost share: VTWR(m,i,s,d) / sum_m VTWR(.,i,s,d)",
        )
        model.share_st = Param(
            model.marg, model.r,
            initialize=dict(c.share_st), default=0.0, mutable=False,
            doc="Margin commodity supply share (CD aggregator for ptmg)",
        )

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------

    def _add_variables(self, model: "ConcreteModel") -> None:
        """Declare Pyomo variables.

        Quantities have a lower bound of 1e-6 (matches the v7 convention)
        to keep CES exponents well-defined. Prices initialize at 1.0
        (the benchmark normalization). Aggregate income variables
        initialize from the calibration ``VOM`` totals.
        """
        from pyomo.environ import NonNegativeReals, Reals, Var

        lb = max(self.closure.numeraire and 1e-6, 1e-6)

        b = self.params.benchmark
        c = self.derived

        def _init_q(value: float) -> float:
            return max(value, lb)

        # --- Output and producer prices -----------------------------------

        # ``qo`` is initialized at the production-cost base (``vop``) so
        # the CES top nest balances exactly at benchmark. The output-side
        # aggregate (``vom``) may differ by the implicit output tax
        # ``to``; this is handled inside ``eq_qo`` via the
        # ``ps * (1 + to)`` wedge.
        model.qo = Var(
            model.j, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, j, r: _init_q(c.vop.get((j, r), c.vom.get((j, r), 1.0))),
            doc="qo(j,r) — output of sector j in region r",
        )
        model.ps = Var(
            model.j, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="ps(j,r) — supply (cost) price of commodity j in region r (basic price)",
        )
        # Domestic supply price (with output tax wedge): pds = ps * (1 + to)
        # Used by downstream agents (firms, households, government) as
        # the basic price of domestic inputs.
        model.pds = Var(
            model.j, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, j, r: 1.0 + c.to.get((j, r), 0.0),
            doc="pds(j,r) — domestic supply price (= ps * (1 + to))",
        )
        model.pm = Var(
            model.j, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="pm(j,r) — market price of commodity j in region r (weighted avg of pds and pim)",
        )

        # --- Value-added aggregate ----------------------------------------

        model.va = Var(
            model.j, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, j, r: _init_q(c.va_total.get((j, r), 1.0)),
            doc="va(j,r) — value-added aggregate (quantity)",
        )
        model.pva = Var(
            model.j, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="pva(j,r) — value-added composite price",
        )

        # --- Factor demands and prices (commodity-level §12) --------------

        model.qfe = Var(
            model.f, model.j, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, f, j, r: _init_q(b.vfm.get((f, j, r), 1.0)),
            doc="qfe(f,j,r) — factor f demand by sector j in region r",
        )
        model.pfe = Var(
            model.f, model.j, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="pfe(f,j,r) — factor f price in sector j, region r",
        )
        # pm(f,r) — regional factor price (mobile factor wage)
        model.pf = Var(
            model.f, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="pf(f,r) — regional factor price for f in r",
        )
        # qoes(f,r) — sluggish factor allocation total
        model.qoes = Var(
            model.f, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, f, r: _init_q(c.evom.get((f, r), 1.0)),
            doc="qoes(f,r) — total factor supply of f in r",
        )

        # --- Intermediate demand (firms) ----------------------------------

        model.qfd = Var(
            model.i, model.j, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, j, r: _init_q(b.vdfm.get((i, j, r), 1.0)),
            doc="qfd(i,j,r) — domestic intermediate demand",
        )
        model.qfm = Var(
            model.i, model.j, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, j, r: _init_q(b.vifm.get((i, j, r), 1.0)),
            doc="qfm(i,j,r) — imported intermediate demand (calibrated to VIFM at benchmark)",
        )
        model.qf = Var(
            model.i, model.j, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, j, r: _init_q(
                b.vdfm.get((i, j, r), 0.0) + b.vifm.get((i, j, r), 0.0)
            ),
            doc="qf(i,j,r) — Armington composite intermediate",
        )
        # Agent prices include both upstream price levels and the firm-
        # side intermediate tax wedge:
        #   pfd(i,j,r) = pds(i,r) * (1 + tfd(i,j,r))
        #   pfm(i,j,r) = pim(i,r) * (1 + tfi(i,j,r))
        # Initialize at the analytic benchmark values so eq_pfd / eq_pfm
        # residuals are zero before solve.
        model.pfd = Var(
            model.i, model.j, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, j, r: (
                (1.0 + c.to.get((i, r), 0.0)) * (1.0 + c.tfd.get((i, j, r), 0.0))
            ),
            doc="pfd(i,j,r) — domestic intermediate agent price",
        )
        model.pfm = Var(
            model.i, model.j, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, j, r: (
                c.pim_0.get((i, r), 1.0) * (1.0 + c.tfi.get((i, j, r), 0.0))
            ),
            doc="pfm(i,j,r) — imported intermediate agent price (= pim * (1+tfi))",
        )
        # Firm Armington composite price. Calibration computes the
        # benchmark value as a cost-weighted average of pfd and pfm
        # (see :data:`DerivedV62Calibration.pf_int_0`); this guarantees
        # the CES first-order conditions hold identically.
        model.pf_int = Var(
            model.i, model.j, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, j, r: max(c.pf_int_0.get((i, j, r), 1.0), lb),
            doc="pf_int(i,j,r) — Armington composite price (firm side)",
        )

        # --- Household & government composite Armington prices (agent) ----
        #
        # Phase 2c.1 adds household and government variants of the firm-
        # side Armington top nest. The variable names follow v6.2 TAB:
        # pp(i,r), ppd(i,r), ppm(i,r) for household; pg(i,r), pgd(i,r),
        # pgm(i,r) for government.

        # --- Household demand (CD allocation across goods; CES Armington) -

        model.qpd = Var(
            model.i, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, r: _init_q(b.vdpm.get((i, r), 1.0)),
            doc="qpd(i,r) — household domestic demand",
        )
        model.qpm = Var(
            model.i, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, r: _init_q(b.vipm.get((i, r), 1.0)),
            doc="qpm(i,r) — household imported demand (= VIPM at benchmark)",
        )
        model.qp = Var(
            model.i, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, r: _init_q(
                b.vdpm.get((i, r), 0.0) + b.vipm.get((i, r), 0.0)
            ),
            doc="qp(i,r) — household composite Armington demand",
        )
        model.pp = Var(
            model.i, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, r: max(c.pp_0.get((i, r), 1.0), lb),
            doc="pp(i,r) — household composite Armington price",
        )
        model.ppd = Var(
            model.i, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, r: (
                (1.0 + c.to.get((i, r), 0.0)) * (1.0 + c.tpd.get((i, r), 0.0))
            ),
            doc="ppd(i,r) — household domestic agent price",
        )
        model.ppm = Var(
            model.i, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, r: c.pim_0.get((i, r), 1.0) * (1.0 + c.tpi.get((i, r), 0.0)),
            doc="ppm(i,r) — household imported agent price (= pim * (1+tpi))",
        )
        # pcons_0 = sum_i share_hhd * pp_0 (consistent with linear CD aggregator).
        def _pcons_init(m, r):
            total = sum(
                c.share_hhd_cd.get((i, r), 0.0) * c.pp_0.get((i, r), 1.0)
                for i in self.sets.i
            )
            return max(total, lb)
        model.pcons = Var(
            model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=_pcons_init,
            doc="pcons(r) — household consumption price index (linear CD aggregator)",
        )
        # up_0 = 1/pcons_0 so up * pcons * yp_0 = yp holds at benchmark.
        def _up_init(m, r):
            pcons_0 = sum(
                c.share_hhd_cd.get((i, r), 0.0) * c.pp_0.get((i, r), 1.0)
                for i in self.sets.i
            )
            return 1.0 / max(pcons_0, 1e-8)
        model.up = Var(
            model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=_up_init,
            doc="up(r) — private utility (Phase 2c.1: CD approximation of CDE)",
        )

        # --- Government demand (Cobb-Douglas, §7) -------------------------

        model.qgd = Var(
            model.i, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, r: _init_q(b.vdgm.get((i, r), 1.0)),
            doc="qgd(i,r) — gov domestic demand",
        )
        model.qgm = Var(
            model.i, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, r: _init_q(b.vigm.get((i, r), 1.0)),
            doc="qgm(i,r) — gov imported demand (= VIGM at benchmark)",
        )
        model.qg = Var(
            model.i, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, r: _init_q(
                b.vdgm.get((i, r), 0.0) + b.vigm.get((i, r), 0.0)
            ),
            doc="qg(i,r) — gov composite Armington demand",
        )
        model.pg = Var(
            model.i, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, r: max(c.pg_0.get((i, r), 1.0), lb),
            doc="pg(i,r) — gov composite Armington price",
        )
        model.pgd = Var(
            model.i, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, r: (
                (1.0 + c.to.get((i, r), 0.0)) * (1.0 + c.tgd.get((i, r), 0.0))
            ),
            doc="pgd(i,r) — gov domestic agent price",
        )
        model.pgm = Var(
            model.i, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, r: c.pim_0.get((i, r), 1.0) * (1.0 + c.tgi.get((i, r), 0.0)),
            doc="pgm(i,r) — gov imported agent price (= pim * (1+tgi))",
        )
        def _pgov_init(m, r):
            total = sum(
                c.share_gov_cd.get((i, r), 0.0) * c.pg_0.get((i, r), 1.0)
                for i in self.sets.i
            )
            return max(total, lb)
        model.pgov = Var(
            model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=_pgov_init,
            doc="pgov(r) — gov price index (linear CD aggregator across goods)",
        )
        def _ug_init(m, r):
            pgov_0 = sum(
                c.share_gov_cd.get((i, r), 0.0) * c.pg_0.get((i, r), 1.0)
                for i in self.sets.i
            )
            return 1.0 / max(pgov_0, 1e-8)
        model.ug = Var(
            model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=_ug_init,
            doc="ug(r) — gov utility (ug_0 = 1/pgov_0)",
        )

        # --- Investment (sector cgds, §8) ---------------------------------

        model.qcgds = Var(
            model.cgds, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, cg, r: _init_q(c.vom.get((cg, r), 1.0)),
            doc="qcgds(cgds,r) — investment output",
        )
        model.pcgds = Var(
            model.cgds, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="pcgds(cgds,r) — investment output price",
        )

        # --- Trade ---------------------------------------------------------

        # Trade variables initialized to the analytic benchmark levels
        # (VXWD as basic-price quantity, computed prices along the chain).
        model.qxs = Var(
            model.i, model.s, model.rp,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, s, rp: max(c.qxs_0.get((i, s, rp), b.vxwd.get((i, s, rp), 0.0)), lb),
            doc="qxs(i,s,rp) — bilateral exports (basic-price quantity = VXWD at benchmark)",
        )
        model.pms = Var(
            model.i, model.s, model.rp,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, s, rp: max(c.pms_0.get((i, s, rp), 1.0), lb),
            doc="pms(i,s,rp) — bilateral market price at destination",
        )
        model.pmcif = Var(
            model.i, model.s, model.rp,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, s, rp: max(c.pmcif_0.get((i, s, rp), 1.0), lb),
            doc="pmcif(i,s,rp) — CIF price (FOB + transport)",
        )
        model.pe = Var(
            model.i, model.s, model.rp,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, s, rp: max(c.pe_0.get((i, s, rp), 1.0), lb),
            doc="pe(i,s,rp) — FOB price (= ps * (1 + txs))",
        )
        model.qim = Var(
            model.i, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, r: max(c.qim_0.get((i, r), 1.0), lb),
            doc="qim(i,r) — composite import (basic-price aggregate)",
        )
        model.pim = Var(
            model.i, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, r: max(c.pim_0.get((i, r), 1.0), lb),
            doc="pim(i,r) — composite import price (CES dual)",
        )
        # Domestic absorption supply
        model.qds = Var(
            model.i, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, r: _init_q(c.vds.get((i, r), 1.0)),
            doc="qds(i,r) — domestic absorption",
        )

        # --- Margins (Cobb-Douglas, §10) ---------------------------------

        model.qst = Var(
            model.marg, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, mg, r: max(c.qst_0.get((mg, r), b.vst.get((mg, r), 0.0)), lb),
            doc="qst(m,r) — margin sales (= VST(m,r) at benchmark)",
        )
        model.pst = Var(
            model.marg, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="pst(m,r) — margin sale price (= ps[m,r] for margin commodity)",
        )
        model.qtm = Var(
            model.marg,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, mg: max(c.qtm_0.get(mg, 1.0), lb),
            doc="qtm(m) — world margin demand (= sum_r VST at benchmark)",
        )
        model.ptmg = Var(
            model.marg,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="ptmg(m) — world margin price (CD aggregator)",
        )
        model.pwmg = Var(
            model.i, model.s, model.rp,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, s, rp: max(c.pwmg_0.get((i, s, rp), 0.0), lb),
            doc="pwmg(i,s,rp) — per-unit transport cost on bilateral shipment",
        )

        # --- Income, savings, investment closure --------------------------

        model.y = Var(
            model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, r: max(
                sum(c.evom.get((f, r), 0.0) for f in self.sets.f),
                lb,
            ),
            doc="y(r) — regional income",
        )
        model.yp = Var(
            model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, r: max(c.yp_0.get(r, 1.0), lb),
            doc="yp(r) — household income (= total consumption expenditure at benchmark)",
        )
        model.yg = Var(
            model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, r: max(c.yg_0.get(r, 1.0), lb),
            doc="yg(r) — gov income (= total gov expenditure at benchmark)",
        )
        model.psave = Var(
            model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="psave(r) — savings price",
        )
        model.savf = Var(
            model.r,
            within=Reals,  # can be negative (capital outflow)
            initialize=0.0,
            doc="savf(r) — net foreign savings",
        )
        model.rorg = Var(
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="rorg — global rate of return",
        )
        model.kb = Var(
            model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, r: _init_q(b.vkb.get(r, 1.0)),
            doc="kb(r) — beginning capital stock",
        )
        model.ke = Var(
            model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, r: _init_q(b.vkb.get(r, 1.0)),
            doc="ke(r) — end-of-period capital stock",
        )

        # --- Tax revenue per stream --------------------------------------

        model.taxrev = Var(
            model.r, model.gy,
            within=Reals,
            initialize=0.0,
            doc="taxrev(r, stream) — tax revenue by stream",
        )

        # --- Numeraire and GDP --------------------------------------------

        model.pgdpwld = Var(
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="pgdpwld — world GDP price index (v6.2 numeraire)",
        )
        model.gdpmp = Var(
            model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, r: max(
                sum(c.evom.get((f, r), 0.0) for f in self.sets.f), lb,
            ),
            doc="gdpmp(r) — nominal GDP at market prices",
        )
        model.rgdpmp = Var(
            model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, r: max(
                sum(c.evom.get((f, r), 0.0) for f in self.sets.f), lb,
            ),
            doc="rgdpmp(r) — real GDP",
        )
        model.pgdpmp = Var(
            model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="pgdpmp(r) — GDP deflator",
        )

        # --- Walras check -------------------------------------------------

        model.walras = Var(
            within=Reals,
            initialize=0.0,
            doc="walras — global market clearing residual",
        )

    # ------------------------------------------------------------------
    # Equations (Phase 2a: placeholder; Phase 2b populates)
    # ------------------------------------------------------------------

    def _add_equations(self, model: "ConcreteModel") -> None:
        """Wire Pyomo Constraints onto the model.

        Phase 2b populates the **production block**, Phase 2c.1 adds
        the **household + government demand block** and **CGDS
        identities**, Phase 2c.2 adds the **trade block** (pricing
        chain + bottom Armington), **margins block** (Cobb-Douglas)
        and **commodity market clearing**.

        Phase 2c.3 (next) will wire the **factor markets** (mobile
        wage clearing + sluggish CET), the **income block** (regional
        income, household and gov income identities, tax revenue per
        stream), and the **closure** (numeraire and Walras).

        Known Phase 2c.2 benchmark residuals (BOOK3X3):

        - eq_qo: ~6% (output-tax wedge).
        - eq_qpm / eq_qgm / eq_pp / eq_pg: small residuals (max ~1.7e4
          in absolute / ~5% in relative) from the cascade of
          benchmark prices through household/gov Armington when
          pim_0 ≠ 1. The calibration uses MARKET-value shares; the
          CES first-order conditions then have a small benchmark
          imbalance proportional to (pim_0 - 1) × VIPM.
        - eq_qtm: ~6.6e4 — self-trade VTWR entries (intra-region
          transport in the SAM) are not represented by the
          bilateral qxs flows (s == d). The model excludes intra-
          region transport demand; full reconcile requires either
          dropping those SAM cells or adding an intra-region freight
          variable. Deferred to Phase 2d.
        - eq_market: ~2.3e4 — market clearing residual reflects
          mixed price levels (basic-price quantities on the use
          side vs producer-price quantities on the output side).
          Phase 2d reconciles via explicit pm = weighted-avg of
          pds and pim.

        The CES exponent convention is:
            σ = 0  → Leontief (handled as special case)
            σ > 0  → standard CES with elasticity σ
            σ = 1  → unit-elastic (Cobb-Douglas; handled by a small
                     perturbation so the (1-σ) exponent is well-defined)
        """
        self._add_production_block(model)
        self._add_household_demand_block(model)
        self._add_government_demand_block(model)
        self._add_investment_identities(model)
        self._add_trade_block(model)
        self._add_margins_block(model)
        self._add_market_clearing(model)

    # ------------------------------------------------------------------
    # Equation blocks — Production
    # ------------------------------------------------------------------

    def _add_production_block(self, model: "ConcreteModel") -> None:
        """Top CES nest + VA CES + firm-side Armington top.

        Equations added:
        - eq_qo:    zero-profit unit cost identity (CES top nest)
        - eq_va:    VA composite quantity from CES top nest
        - eq_qf:    intermediate composite quantity from CES top nest
        - eq_pva:   VA composite price (CES across factors)
        - eq_qfe:   factor demand from VA CES
        - eq_pfe:   factor agent price (regional wage + factor tax)
        - eq_pf_int: Armington firm composite price for each intermediate
        - eq_qfd:   domestic intermediate demand (Armington)
        - eq_qfm:   imported intermediate demand (Armington)
        - eq_pfd:   domestic intermediate agent price (market + tfd wedge)
        - eq_pfm:   imported intermediate agent price (pim + tfi wedge)

        All equations are unit-level CES first-order conditions
        calibrated so that, at the benchmark (all prices = 1, quantities
        = SAM values), residuals are zero.

        References:
            - v6.2 TAB lines ~700-1100 (production section in gtap.tab)
            - notation_crosswalk.md §2-§4 for Tabla 1 mapping
        """
        from pyomo.environ import Constraint, value as pyo_value

        # Convenience helpers
        def _eps_sigma(sigma: float) -> bool:
            """True if σ is small enough to treat as Leontief."""
            return abs(sigma) < 1e-8

        def _ces_cd_sigma(sigma: float) -> float:
            """Perturb σ when it equals 1.0 to avoid (1-σ) = 0 pathologies."""
            if abs(sigma - 1.0) < 1e-8:
                return 1.0 + 1e-3
            return sigma

        # ---------- eq_qo: zero-profit unit cost identity (top CES nest)
        #
        # ``ps(j,r)`` is the BASIC (cost) price: what the producer pays
        # for inputs per unit of output. The output tax wedge is
        # captured separately via :func:`eq_pds` (``pds = ps * (1+to)``)
        # and lives in the downstream price chain (eq_pfd, etc).
        #
        # CES cost function:
        #     ps^(1-σ) = share_va * pva^(1-σ) + sum_i share_int(i) * pf_int(i)^(1-σ)
        # Leontief (σ=0):
        #     ps = share_va * pva + sum_i share_int(i) * pf_int(i)
        def eq_qo_rule(m, j, r):
            if pyo_value(m.vop[j, r]) <= 1e-8 if hasattr(m, "vop") else pyo_value(m.vom[j, r]) <= 1e-8:
                return Constraint.Skip
            sva = float(pyo_value(m.share_va[j, r]))
            sigma_top = float(pyo_value(m.esubt[j]))

            if _eps_sigma(sigma_top):
                int_sum = sum(
                    float(pyo_value(m.share_int[i, j, r])) * m.pf_int[i, j, r]
                    for i in m.i
                )
                return m.ps[j, r] == sva * m.pva[j, r] + int_sum
            sigma_top = _ces_cd_sigma(sigma_top)
            exp = 1.0 - sigma_top
            int_sum = sum(
                float(pyo_value(m.share_int[i, j, r])) * m.pf_int[i, j, r] ** exp
                for i in m.i
            )
            return m.ps[j, r] ** exp == sva * m.pva[j, r] ** exp + int_sum
        model.eq_qo = Constraint(model.j, model.r, rule=eq_qo_rule)

        # ---------- eq_pds: domestic supply price (output tax wedge)
        #
        # pds(j,r) = ps(j,r) * (1 + to(j,r))
        def eq_pds_rule(m, j, r):
            if pyo_value(m.vom[j, r]) <= 1e-8:
                return Constraint.Skip
            return m.pds[j, r] == m.ps[j, r] * (1.0 + pyo_value(m.to[j, r]))
        model.eq_pds = Constraint(model.j, model.r, rule=eq_pds_rule)

        # ---------- eq_va: VA demand from CES top nest

        # CES first-order condition:
        #   va(j,r) = share_va * qo * (ps/pva)^σ
        # Leontief: va(j,r) = share_va * qo
        def eq_va_rule(m, j, r):
            sva = float(pyo_value(m.share_va[j, r]))
            if sva <= 0.0:
                return Constraint.Skip
            sigma_top = float(pyo_value(m.esubt[j]))
            if _eps_sigma(sigma_top):
                return m.va[j, r] == sva * m.qo[j, r]
            sigma_top = _ces_cd_sigma(sigma_top)
            return m.va[j, r] == sva * m.qo[j, r] * (m.ps[j, r] / m.pva[j, r]) ** sigma_top
        model.eq_va = Constraint(model.j, model.r, rule=eq_va_rule)

        # ---------- eq_qf: intermediate composite demand from CES top nest

        # qf(i,j,r) = share_int(i,j,r) * qo * (ps/pf_int(i,j,r))^σ
        # Leontief:  qf(i,j,r) = share_int(i,j,r) * qo
        def eq_qf_rule(m, i, j, r):
            sint = float(pyo_value(m.share_int[i, j, r]))
            if sint <= 0.0:
                return Constraint.Skip
            sigma_top = float(pyo_value(m.esubt[j]))
            if _eps_sigma(sigma_top):
                return m.qf[i, j, r] == sint * m.qo[j, r]
            sigma_top = _ces_cd_sigma(sigma_top)
            return m.qf[i, j, r] == sint * m.qo[j, r] * (m.ps[j, r] / m.pf_int[i, j, r]) ** sigma_top
        model.eq_qf = Constraint(model.i, model.j, model.r, rule=eq_qf_rule)

        # ---------- eq_pva: VA composite price (CES across factors)

        # pva(j,r)^(1-σ_va) = sum_f share_fac(f,j,r) * pfe(f,j,r)^(1-σ_va)
        def eq_pva_rule(m, j, r):
            if pyo_value(m.va_total[j, r]) <= 1e-8:
                return Constraint.Skip
            sigma_va = float(pyo_value(m.esubva[j]))
            sigma_va = _ces_cd_sigma(sigma_va)
            exp = 1.0 - sigma_va
            rhs = sum(
                float(pyo_value(m.share_fac[f, j, r])) * m.pfe[f, j, r] ** exp
                for f in m.f
                if pyo_value(m.share_fac[f, j, r]) > 0.0
            )
            return m.pva[j, r] ** exp == rhs
        model.eq_pva = Constraint(model.j, model.r, rule=eq_pva_rule)

        # ---------- eq_qfe: factor demand from VA CES

        # qfe(f,j,r) = share_fac(f,j,r) * va(j,r) * (pva / pfe)^σ_va
        def eq_qfe_rule(m, f, j, r):
            sfac = float(pyo_value(m.share_fac[f, j, r]))
            if sfac <= 0.0:
                return Constraint.Skip
            sigma_va = float(pyo_value(m.esubva[j]))
            sigma_va = _ces_cd_sigma(sigma_va)
            return m.qfe[f, j, r] == sfac * m.va[j, r] * (m.pva[j, r] / m.pfe[f, j, r]) ** sigma_va
        model.eq_qfe = Constraint(model.f, model.j, model.r, rule=eq_qfe_rule)

        # ---------- eq_pfe: factor agent price = wage * (1 + factor tax)

        # In v6.2, all factors have a regional aggregate price pf(f,r)
        # (mobile or sluggish — Phase 2c distinguishes them via the CET
        # block). Here we use the same identity for both kinds:
        #
        #   pfe(f,j,r) = pf(f,r) * (1 + tf(f,j,r))
        #
        # When the sluggish CET is added in Phase 2c, pf(f,r) will be
        # replaced by pmes(f,r) for sluggish factors and the per-sector
        # gap will reflect the CET shadow-price split.
        def eq_pfe_rule(m, f, j, r):
            sfac = float(pyo_value(m.share_fac[f, j, r]))
            if sfac <= 0.0:
                return Constraint.Skip
            return m.pfe[f, j, r] == m.pf[f, r] * (1.0 + pyo_value(m.tf[f, j, r]))
        model.eq_pfe = Constraint(model.f, model.j, model.r, rule=eq_pfe_rule)

        # ---------- eq_pf_int: Armington composite price for intermediates
        #
        # CES dual price aggregator over AGENT prices, using calibrated
        # distribution parameters (``alpha_dom`` / ``alpha_imp``). The
        # distribution parameters absorb the benchmark agent-price
        # ratios so the equation is identically satisfied at the
        # calibration point even when pfd_0 ≠ pfm_0 (tax wedges).
        #
        #   pf_int^(1-σ_d) = alpha_dom * pfd^(1-σ_d) + alpha_imp * pfm^(1-σ_d)
        def eq_pf_int_rule(m, i, j, r):
            ad = float(pyo_value(m.alpha_dom[i, j, r]))
            ai = float(pyo_value(m.alpha_imp[i, j, r]))
            if ad + ai <= 1e-12:
                return Constraint.Skip
            sigma_d = float(pyo_value(m.esubd[i]))
            sigma_d = _ces_cd_sigma(sigma_d)
            exp = 1.0 - sigma_d
            return m.pf_int[i, j, r] ** exp == ad * m.pfd[i, j, r] ** exp + ai * m.pfm[i, j, r] ** exp
        model.eq_pf_int = Constraint(model.i, model.j, model.r, rule=eq_pf_int_rule)

        # ---------- eq_qfd: domestic intermediate demand (Armington)
        #
        # CES first-order condition with calibrated distribution
        # parameter (absorbs benchmark agent-price ratio):
        #     qfd(i,j,r) = alpha_dom(i,j,r) * qf(i,j,r) * (pf_int / pfd)^σ_d
        def eq_qfd_rule(m, i, j, r):
            ad = float(pyo_value(m.alpha_dom[i, j, r]))
            if ad <= 0.0:
                return Constraint.Skip
            sigma_d = float(pyo_value(m.esubd[i]))
            sigma_d = _ces_cd_sigma(sigma_d)
            return m.qfd[i, j, r] == ad * m.qf[i, j, r] * (m.pf_int[i, j, r] / m.pfd[i, j, r]) ** sigma_d
        model.eq_qfd = Constraint(model.i, model.j, model.r, rule=eq_qfd_rule)

        # ---------- eq_qfm: imported intermediate demand (Armington)
        #
        #     qfm(i,j,r) = alpha_imp(i,j,r) * qf(i,j,r) * (pf_int / pfm)^σ_d
        def eq_qfm_rule(m, i, j, r):
            ai = float(pyo_value(m.alpha_imp[i, j, r]))
            if ai <= 0.0:
                return Constraint.Skip
            sigma_d = float(pyo_value(m.esubd[i]))
            sigma_d = _ces_cd_sigma(sigma_d)
            return m.qfm[i, j, r] == ai * m.qf[i, j, r] * (m.pf_int[i, j, r] / m.pfm[i, j, r]) ** sigma_d
        model.eq_qfm = Constraint(model.i, model.j, model.r, rule=eq_qfm_rule)

        # ---------- eq_pfd: domestic intermediate agent price
        #
        # pfd(i,j,r) = pds(i,r) * (1 + tfd(i,j,r))
        #
        # where pds(i,r) = ps(i,r) * (1 + to(i,r)) carries the output
        # tax wedge from eq_pds.
        def eq_pfd_rule(m, i, j, r):
            sd = float(pyo_value(m.share_dom[i, j, r]))
            if sd <= 0.0:
                return Constraint.Skip
            return m.pfd[i, j, r] == m.pds[i, r] * (1.0 + pyo_value(m.tfd[i, j, r]))
        model.eq_pfd = Constraint(model.i, model.j, model.r, rule=eq_pfd_rule)

        # ---------- eq_pfm: imported intermediate agent price

        # pfm(i,j,r) = pim(i,r) * (1 + tfi(i,j,r))
        def eq_pfm_rule(m, i, j, r):
            si = float(pyo_value(m.share_imp[i, j, r]))
            if si <= 0.0:
                return Constraint.Skip
            return m.pfm[i, j, r] == m.pim[i, r] * (1.0 + pyo_value(m.tfi[i, j, r]))
        model.eq_pfm = Constraint(model.i, model.j, model.r, rule=eq_pfm_rule)

    # ------------------------------------------------------------------
    # Equation blocks — Household demand (Phase 2c.1)
    # ------------------------------------------------------------------

    def _add_household_demand_block(self, model: "ConcreteModel") -> None:
        """Household demand: top Armington (CES) + CD allocation across goods.

        Equations:
        - eq_pp:    composite Armington price (CES dual aggregator)
        - eq_qp:    CD budget allocation: pp * qp = share * yp
        - eq_qpd:   domestic demand from Armington (CES)
        - eq_qpm:   imported demand from Armington
        - eq_ppd:   ppd = pds * (1 + tpd)
        - eq_ppm:   ppm = pim * (1 + tpi)
        - eq_pcons: household price index (sum of pp * share)
        - eq_up:    up = yp / pcons

        Phase 2c.1 note: this is a Cobb-Douglas approximation of v6.2's
        CDE demand system. The two coincide at the benchmark; small
        differences from the CDE form appear under shocks with strong
        income/substitution effects. CDE upgrade is deferred to
        Phase 2d.
        """
        from pyomo.environ import Constraint, value as pyo_value

        def _eps_sigma(sigma: float) -> bool:
            return abs(sigma) < 1e-8

        def _ces_cd_sigma(sigma: float) -> float:
            if abs(sigma - 1.0) < 1e-8:
                return 1.0 + 1e-3
            return sigma

        # eq_pp: Armington composite price for household
        def eq_pp_rule(m, i, r):
            ad = float(pyo_value(m.alpha_dom_hhd[i, r]))
            ai = float(pyo_value(m.alpha_imp_hhd[i, r]))
            if ad + ai <= 1e-12:
                return Constraint.Skip
            sigma_d = _ces_cd_sigma(float(pyo_value(m.esubd[i])))
            exp = 1.0 - sigma_d
            return m.pp[i, r] ** exp == ad * m.ppd[i, r] ** exp + ai * m.ppm[i, r] ** exp
        model.eq_pp = Constraint(model.i, model.r, rule=eq_pp_rule)

        # eq_qp: CD budget allocation
        def eq_qp_rule(m, i, r):
            cs = float(pyo_value(m.share_hhd_cd[i, r]))
            if cs <= 0.0:
                return Constraint.Skip
            return m.pp[i, r] * m.qp[i, r] == cs * m.yp[r]
        model.eq_qp = Constraint(model.i, model.r, rule=eq_qp_rule)

        # eq_qpd: domestic demand (CES first-order condition with α calibrated)
        def eq_qpd_rule(m, i, r):
            ad = float(pyo_value(m.alpha_dom_hhd[i, r]))
            if ad <= 0.0:
                return Constraint.Skip
            sigma_d = _ces_cd_sigma(float(pyo_value(m.esubd[i])))
            return m.qpd[i, r] == ad * m.qp[i, r] * (m.pp[i, r] / m.ppd[i, r]) ** sigma_d
        model.eq_qpd = Constraint(model.i, model.r, rule=eq_qpd_rule)

        # eq_qpm: imported demand
        def eq_qpm_rule(m, i, r):
            ai = float(pyo_value(m.alpha_imp_hhd[i, r]))
            if ai <= 0.0:
                return Constraint.Skip
            sigma_d = _ces_cd_sigma(float(pyo_value(m.esubd[i])))
            return m.qpm[i, r] == ai * m.qp[i, r] * (m.pp[i, r] / m.ppm[i, r]) ** sigma_d
        model.eq_qpm = Constraint(model.i, model.r, rule=eq_qpm_rule)

        # eq_ppd: household domestic agent price
        def eq_ppd_rule(m, i, r):
            if float(pyo_value(m.alpha_dom_hhd[i, r])) <= 0.0:
                return Constraint.Skip
            return m.ppd[i, r] == m.pds[i, r] * (1.0 + pyo_value(m.tpd[i, r]))
        model.eq_ppd = Constraint(model.i, model.r, rule=eq_ppd_rule)

        # eq_ppm: household imported agent price
        def eq_ppm_rule(m, i, r):
            if float(pyo_value(m.alpha_imp_hhd[i, r])) <= 0.0:
                return Constraint.Skip
            return m.ppm[i, r] == m.pim[i, r] * (1.0 + pyo_value(m.tpi[i, r]))
        model.eq_ppm = Constraint(model.i, model.r, rule=eq_ppm_rule)

        # eq_pcons: household price index (CD aggregator, linear form for the
        # benchmark identity sum_i pp * qp = pcons * yp_unit which at u=1 says
        # pcons = sum_i share * pp).
        def eq_pcons_rule(m, r):
            terms = [
                float(pyo_value(m.share_hhd_cd[i, r])) * m.pp[i, r]
                for i in m.i
                if pyo_value(m.share_hhd_cd[i, r]) > 0.0
            ]
            if not terms:
                return Constraint.Skip
            return m.pcons[r] == sum(terms)
        model.eq_pcons = Constraint(model.r, rule=eq_pcons_rule)

        # eq_up: household utility = real income.
        # Normalization: up_0 = 1, so up_0 * yp_0 * pcons_0 = yp_0 ⇒ up = yp / (yp_0 * pcons).
        def eq_up_rule(m, r):
            yp_0 = max(float(self.derived.yp_0.get(r, 1.0)), 1e-8)
            return m.up[r] * yp_0 * m.pcons[r] == m.yp[r]
        model.eq_up = Constraint(model.r, rule=eq_up_rule)

    # ------------------------------------------------------------------
    # Equation blocks — Government demand (Phase 2c.1)
    # ------------------------------------------------------------------

    def _add_government_demand_block(self, model: "ConcreteModel") -> None:
        """Government demand: top Armington (CES) + CD allocation.

        v6.2 government uses Cobb-Douglas by default (no ESBG in v6.2
        TAB). Equation set mirrors the household block:

        - eq_pg:   gov composite Armington price (CES dual)
        - eq_qg:   CD allocation pg * qg = share * yg
        - eq_qgd:  domestic demand from gov Armington
        - eq_qgm:  imported demand
        - eq_pgd:  pgd = pds * (1 + tgd)
        - eq_pgm:  pgm = pim * (1 + tgi)
        - eq_pgov: gov price index (CD aggregator)
        - eq_ug:   ug = yg / pgov   (utility = real gov expenditure)
        """
        from pyomo.environ import Constraint, value as pyo_value

        def _ces_cd_sigma(sigma: float) -> float:
            if abs(sigma - 1.0) < 1e-8:
                return 1.0 + 1e-3
            return sigma

        # eq_pg
        def eq_pg_rule(m, i, r):
            ad = float(pyo_value(m.alpha_dom_gov[i, r]))
            ai = float(pyo_value(m.alpha_imp_gov[i, r]))
            if ad + ai <= 1e-12:
                return Constraint.Skip
            sigma_d = _ces_cd_sigma(float(pyo_value(m.esubd[i])))
            exp = 1.0 - sigma_d
            return m.pg[i, r] ** exp == ad * m.pgd[i, r] ** exp + ai * m.pgm[i, r] ** exp
        model.eq_pg = Constraint(model.i, model.r, rule=eq_pg_rule)

        # eq_qg: CD budget allocation
        def eq_qg_rule(m, i, r):
            cs = float(pyo_value(m.share_gov_cd[i, r]))
            if cs <= 0.0:
                return Constraint.Skip
            return m.pg[i, r] * m.qg[i, r] == cs * m.yg[r]
        model.eq_qg = Constraint(model.i, model.r, rule=eq_qg_rule)

        # eq_qgd
        def eq_qgd_rule(m, i, r):
            ad = float(pyo_value(m.alpha_dom_gov[i, r]))
            if ad <= 0.0:
                return Constraint.Skip
            sigma_d = _ces_cd_sigma(float(pyo_value(m.esubd[i])))
            return m.qgd[i, r] == ad * m.qg[i, r] * (m.pg[i, r] / m.pgd[i, r]) ** sigma_d
        model.eq_qgd = Constraint(model.i, model.r, rule=eq_qgd_rule)

        # eq_qgm
        def eq_qgm_rule(m, i, r):
            ai = float(pyo_value(m.alpha_imp_gov[i, r]))
            if ai <= 0.0:
                return Constraint.Skip
            sigma_d = _ces_cd_sigma(float(pyo_value(m.esubd[i])))
            return m.qgm[i, r] == ai * m.qg[i, r] * (m.pg[i, r] / m.pgm[i, r]) ** sigma_d
        model.eq_qgm = Constraint(model.i, model.r, rule=eq_qgm_rule)

        # eq_pgd / eq_pgm
        def eq_pgd_rule(m, i, r):
            if float(pyo_value(m.alpha_dom_gov[i, r])) <= 0.0:
                return Constraint.Skip
            return m.pgd[i, r] == m.pds[i, r] * (1.0 + pyo_value(m.tgd[i, r]))
        model.eq_pgd = Constraint(model.i, model.r, rule=eq_pgd_rule)

        def eq_pgm_rule(m, i, r):
            if float(pyo_value(m.alpha_imp_gov[i, r])) <= 0.0:
                return Constraint.Skip
            return m.pgm[i, r] == m.pim[i, r] * (1.0 + pyo_value(m.tgi[i, r]))
        model.eq_pgm = Constraint(model.i, model.r, rule=eq_pgm_rule)

        # eq_pgov: gov price index (CD aggregator)
        def eq_pgov_rule(m, r):
            terms = [
                float(pyo_value(m.share_gov_cd[i, r])) * m.pg[i, r]
                for i in m.i
                if pyo_value(m.share_gov_cd[i, r]) > 0.0
            ]
            if not terms:
                return Constraint.Skip
            return m.pgov[r] == sum(terms)
        model.eq_pgov = Constraint(model.r, rule=eq_pgov_rule)

        # eq_ug: government utility = yg / (pgov * yg_0)
        def eq_ug_rule(m, r):
            yg_0 = max(float(self.derived.yg_0.get(r, 1.0)), 1e-8)
            return m.ug[r] * yg_0 * m.pgov[r] == m.yg[r]
        model.eq_ug = Constraint(model.r, rule=eq_ug_rule)

    # ------------------------------------------------------------------
    # Equation blocks — Investment identities (Phase 2c.1)
    # ------------------------------------------------------------------

    def _add_investment_identities(self, model: "ConcreteModel") -> None:
        """Investment as a producing sector cgds.

        v6.2 treats investment as an output of the CGDS sector, not an
        explicit agent (Tabla 1 §8). The CGDS sector consumes
        intermediates (already wired in the production block via
        ``qfd(i, cgds, r)`` and ``qfm(i, cgds, r)``). Phase 2c.1 adds
        two pass-through identities so external code can refer to the
        investment quantity / price via the canonical v6.2 variables
        ``qcgds`` / ``pcgds``:

        - eq_qcgds_id: qcgds(cgds, r) == qo(cgds, r)
        - eq_pcgds_id: pcgds(cgds, r) == ps(cgds, r)
        """
        from pyomo.environ import Constraint

        def eq_qcgds_rule(m, cg, r):
            return m.qcgds[cg, r] == m.qo[cg, r]
        model.eq_qcgds = Constraint(model.cgds, model.r, rule=eq_qcgds_rule)

        def eq_pcgds_rule(m, cg, r):
            return m.pcgds[cg, r] == m.ps[cg, r]
        model.eq_pcgds = Constraint(model.cgds, model.r, rule=eq_pcgds_rule)

    # ------------------------------------------------------------------
    # Equation blocks — Trade (Phase 2c.2)
    # ------------------------------------------------------------------

    def _add_trade_block(self, model: "ConcreteModel") -> None:
        """Bilateral trade pricing chain + CES bottom Armington.

        Wires the v6.2 export-to-import price cascade and the CES
        aggregator across import sources:

        - eq_pe:    pe(i,s,d) = ps(i,s) * (1 + txs(i,s,d))
        - eq_pwmg:  per-unit transport cost = sum_m amgm * ptmg(m)
        - eq_pmcif: pmcif = pe + pwmg
        - eq_pms:   pms = pmcif * (1 + tms)
        - eq_pim:   composite import price (CES dual aggregator)
        - eq_qxs:   bilateral import demand (CES first-order condition)

        The diagonal flow s=d (self-trade) is skipped — only off-diagonal
        bilateral flows are wired. ``qxs`` is in basic-price units
        (= VXWD at benchmark, NOT VXMD).
        """
        from pyomo.environ import Constraint, value as pyo_value

        def _ces_cd_sigma(sigma: float) -> float:
            if abs(sigma - 1.0) < 1e-8:
                return 1.0 + 1e-3
            return sigma

        # eq_pe: FOB price (linear export tax wedge)
        def eq_pe_rule(m, i, s, d):
            if s == d:
                return Constraint.Skip
            if pyo_value(m.qxs[i, s, d]) <= 1e-8 or float(pyo_value(m.alpha_xs[i, s, d])) <= 0.0:
                return Constraint.Skip
            return m.pe[i, s, d] == m.ps[i, s] * (1.0 + pyo_value(m.txs[i, s, d]))
        model.eq_pe = Constraint(model.i, model.s, model.rp, rule=eq_pe_rule)

        # eq_pwmg: per-unit transport cost = sum_m amgm * ptmg(m)
        #
        # At benchmark, pwmg_0 = sum_m amgm * 1 = sum_m VTWR/total_VTWR.
        # That ratio multiplied by total_VTWR/VXWD gives the per-unit
        # cost. But amgm sums to 1 only when there's a non-zero
        # transport flow, so we scale by the benchmark pwmg level.
        def eq_pwmg_rule(m, i, s, d):
            if s == d:
                return Constraint.Skip
            pwmg0 = float(self.derived.pwmg_0.get((i, s, d), 0.0))
            if pwmg0 <= 1e-12:
                return Constraint.Skip
            # Margin price index scaled by benchmark per-unit transport cost.
            return m.pwmg[i, s, d] == pwmg0 * sum(
                float(pyo_value(m.amgm[mg, i, s, d])) * m.ptmg[mg]
                for mg in m.marg
                if pyo_value(m.amgm[mg, i, s, d]) > 0.0
            )
        model.eq_pwmg = Constraint(model.i, model.s, model.rp, rule=eq_pwmg_rule)

        # eq_pmcif: pmcif = pe + pwmg
        def eq_pmcif_rule(m, i, s, d):
            if s == d:
                return Constraint.Skip
            if pyo_value(m.qxs[i, s, d]) <= 1e-8 or float(pyo_value(m.alpha_xs[i, s, d])) <= 0.0:
                return Constraint.Skip
            return m.pmcif[i, s, d] == m.pe[i, s, d] + m.pwmg[i, s, d]
        model.eq_pmcif = Constraint(model.i, model.s, model.rp, rule=eq_pmcif_rule)

        # eq_pms: pms = pmcif * (1 + tms)
        def eq_pms_rule(m, i, s, d):
            if s == d:
                return Constraint.Skip
            if float(pyo_value(m.alpha_xs[i, s, d])) <= 0.0:
                return Constraint.Skip
            return m.pms[i, s, d] == m.pmcif[i, s, d] * (1.0 + pyo_value(m.tms[i, s, d]))
        model.eq_pms = Constraint(model.i, model.s, model.rp, rule=eq_pms_rule)

        # eq_pim: composite import price (CES dual using calibrated α)
        def eq_pim_rule(m, i, d):
            terms = [
                (s, float(pyo_value(m.alpha_xs[i, s, d])))
                for s in m.s if s != d and pyo_value(m.alpha_xs[i, s, d]) > 0.0
            ]
            if not terms:
                return Constraint.Skip
            sigma_m = _ces_cd_sigma(float(pyo_value(m.esubm[i])))
            exp = 1.0 - sigma_m
            return m.pim[i, d] ** exp == sum(
                ax * m.pms[i, s, d] ** exp for s, ax in terms
            )
        model.eq_pim = Constraint(model.i, model.rp, rule=eq_pim_rule)

        # eq_qxs: bilateral import demand (CES FOC)
        def eq_qxs_rule(m, i, s, d):
            if s == d:
                return Constraint.Skip
            ax = float(pyo_value(m.alpha_xs[i, s, d]))
            if ax <= 0.0:
                return Constraint.Skip
            sigma_m = _ces_cd_sigma(float(pyo_value(m.esubm[i])))
            return m.qxs[i, s, d] == ax * m.qim[i, d] * (m.pim[i, d] / m.pms[i, s, d]) ** sigma_m
        model.eq_qxs = Constraint(model.i, model.s, model.rp, rule=eq_qxs_rule)

    # ------------------------------------------------------------------
    # Equation blocks — Margins (Phase 2c.2)
    # ------------------------------------------------------------------

    def _add_margins_block(self, model: "ConcreteModel") -> None:
        """Cobb-Douglas margins block (v6.2 §10).

        v6.2 uses Cobb-Douglas demand for trade & transport margins
        (no ESBS elasticity — that's a v7 addition).

        - eq_pst:  pst(m,r) = ps(m,r)  (margin sale price = supply price)
        - eq_ptmg: ptmg(m) = sum_r share_st(m,r) * pst(m,r)  (CD index)
        - eq_qtm:  world margin demand = sum (i,s,d) of VTWR-derived qty
        - eq_qst:  margin sales = share_st * qtm  (CD demand)
        """
        from pyomo.environ import Constraint, value as pyo_value

        # eq_pst: pst(m,r) = ps(m,r) — margin commodity sold at supply price
        def eq_pst_rule(m, mg, r):
            return m.pst[mg, r] == m.ps[mg, r]
        model.eq_pst = Constraint(model.marg, model.r, rule=eq_pst_rule)

        # eq_ptmg: world margin price as CD aggregator across regional sources
        def eq_ptmg_rule(m, mg):
            terms = [
                (r, float(pyo_value(m.share_st[mg, r])))
                for r in m.r if pyo_value(m.share_st[mg, r]) > 0.0
            ]
            if not terms:
                return Constraint.Skip
            return m.ptmg[mg] == sum(share * m.pst[mg, r] for r, share in terms)
        model.eq_ptmg = Constraint(model.marg, rule=eq_ptmg_rule)

        # eq_qtm: world margin demand from bilateral transport requirements.
        # qtm(m) = sum (i,s,d) of (amgm * pwmg * qxs / ptmg)
        # At benchmark with ptmg=1 and pwmg*qxs = sum_m VTWR(m,i,s,d),
        # this gives qtm_0 = sum (i,s,d) VTWR(m,i,s,d) = total margin services.
        def eq_qtm_rule(m, mg):
            terms = []
            for i in m.i:
                for s in m.s:
                    for d in m.rp:
                        if s == d:
                            continue
                        amg = float(pyo_value(m.amgm[mg, i, s, d]))
                        if amg <= 0.0:
                            continue
                        terms.append(amg * m.pwmg[i, s, d] * m.qxs[i, s, d])
            if not terms:
                return Constraint.Skip
            return m.ptmg[mg] * m.qtm[mg] == sum(terms)
        model.eq_qtm = Constraint(model.marg, rule=eq_qtm_rule)

        # eq_qst: margin sales per region = share * world demand (CD form)
        def eq_qst_rule(m, mg, r):
            sh = float(pyo_value(m.share_st[mg, r]))
            if sh <= 0.0:
                return Constraint.Skip
            return m.qst[mg, r] == sh * m.qtm[mg]
        model.eq_qst = Constraint(model.marg, model.r, rule=eq_qst_rule)

    # ------------------------------------------------------------------
    # Equation blocks — Market clearing (Phase 2c.2)
    # ------------------------------------------------------------------

    def _add_market_clearing(self, model: "ConcreteModel") -> None:
        """Commodity market clearing identity.

        For each traded commodity i in region r, total output must equal
        total uses:

            qo(i,r) * (1 + to(i,r)) = sum_j qfd(i,j,r)
                                    + qpd(i,r) + qgd(i,r)
                                    + sum_d qxs(i,r,d)
                                    + qst(i,r)  (if i is a margin)

        The (1 + to) factor on the left reconciles the output-side
        ``vom`` aggregate with the cost-side ``vop`` used in production
        (eq_qo). For non-margin commodities the qst term is dropped.

        Phase 2c.2 implementation note: at benchmark with the calibrated
        agent-price wedges, this balance does NOT hold exactly because
        the absorption quantities (qfd, qpd, qgd) are basic-price units
        while qo is at production-cost units. Residual is bounded by the
        net tax wedge magnitude. Phase 2d will reconcile via explicit
        pm = weighted-avg equation.
        """
        from pyomo.environ import Constraint, value as pyo_value

        def eq_market_rule(m, i, r):
            if pyo_value(m.vom[i, r]) <= 1e-8:
                return Constraint.Skip

            # Uses side: domestic absorption + exports + margin sales
            uses = sum(m.qfd[i, j, r] for j in m.j)
            uses = uses + m.qpd[i, r] + m.qgd[i, r]
            uses = uses + sum(m.qxs[i, r, d] for d in m.rp if d != r)

            # Margin sales (only if i is a margin commodity)
            if i in m.marg:
                uses = uses + m.qst[i, r]

            # Output side (left): qo at producer level
            return m.qo[i, r] * (1.0 + pyo_value(m.to[i, r])) == uses
        model.eq_market = Constraint(model.i, model.r, rule=eq_market_rule)


__all__ = [
    "GTAPv62ModelEquations",
    "GTAP_V62_TAX_STREAMS",
]
