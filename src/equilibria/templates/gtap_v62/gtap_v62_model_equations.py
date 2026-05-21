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

        model.qo = Var(
            model.j, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, j, r: _init_q(c.vom.get((j, r), 1.0)),
            doc="qo(j,r) — output of sector j in region r",
        )
        model.ps = Var(
            model.j, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="ps(j,r) — supply price of commodity j (output) in region r",
        )
        model.pm = Var(
            model.j, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="pm(j,r) — market price of commodity j in region r",
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
            doc="qfm(i,j,r) — imported intermediate demand",
        )
        model.qf = Var(
            model.i, model.j, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, j, r: _init_q(
                b.vdfm.get((i, j, r), 0.0) + b.vifm.get((i, j, r), 0.0)
            ),
            doc="qf(i,j,r) — Armington composite intermediate",
        )
        # pfd, pfm: agent prices (= (1+tfd) * pm_domestic etc — but
        # we let them solve endogenously, initialize at 1)
        model.pfd = Var(
            model.i, model.j, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="pfd(i,j,r) — domestic intermediate agent price",
        )
        model.pfm = Var(
            model.i, model.j, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="pfm(i,j,r) — imported intermediate agent price",
        )
        model.pf_int = Var(
            model.i, model.j, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="pf_int(i,j,r) — Armington composite price (firm side)",
        )

        # --- Household demand (CDE) ---------------------------------------

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
            doc="qpm(i,r) — household imported demand",
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
            initialize=1.0,
            doc="pp(i,r) — household composite Armington price",
        )
        model.up = Var(
            model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="up(r) — private utility (CDE)",
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
            doc="qgm(i,r) — gov imported demand",
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
            initialize=1.0,
            doc="pg(i,r) — gov composite Armington price",
        )
        model.pgov = Var(
            model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="pgov(r) — gov price index (CD)",
        )
        model.ug = Var(
            model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="ug(r) — gov utility",
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

        model.qxs = Var(
            model.i, model.s, model.rp,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, s, rp: _init_q(b.vxmd.get((i, s, rp), 1.0)),
            doc="qxs(i,s,rp) — bilateral exports",
        )
        model.pms = Var(
            model.i, model.s, model.rp,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="pms(i,s,rp) — bilateral market price",
        )
        model.pmcif = Var(
            model.i, model.s, model.rp,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="pmcif(i,s,rp) — CIF price",
        )
        model.pe = Var(
            model.i, model.s, model.rp,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="pe(i,s,rp) — FOB price",
        )
        model.qim = Var(
            model.i, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=lambda m, i, r: _init_q(c.vim.get((i, r), 1.0)),
            doc="qim(i,r) — composite import",
        )
        model.pim = Var(
            model.i, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="pim(i,r) — composite import price",
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
            initialize=lambda m, mg, r: _init_q(b.vst.get((mg, r), 1.0)),
            doc="qst(m,r) — margin sales",
        )
        model.pst = Var(
            model.marg, model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="pst(m,r) — margin sale price",
        )
        model.qtm = Var(
            model.marg,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="qtm(m) — world margin demand",
        )
        model.ptmg = Var(
            model.marg,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="ptmg(m) — world margin price",
        )
        model.pwmg = Var(
            model.i, model.s, model.rp,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="pwmg(i,s,rp) — bilateral margin cost",
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
            initialize=1.0,
            doc="yp(r) — household income",
        )
        model.yg = Var(
            model.r,
            within=NonNegativeReals, bounds=(lb, None),
            initialize=1.0,
            doc="yg(r) — gov income",
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
        """Placeholder — no equations wired at Phase 2a.

        Phase 2b will populate this method with:
        - Production block (e_qo, e_ps, e_qva, e_pva, e_qfe, e_pfe, e_qf, e_pf)
        - Demand block (CDE households, CD government, investment-as-sector)
        - Trade block (Armington top + bottom, FOB/CIF identities)
        - Factor markets (mobile wages, sluggish CET)
        - Income identities (e_y, e_yp, e_yg, e_taxrev)
        - Closure (e_pgdpwld, e_walras, capital accumulation)
        """
        # Intentionally empty in Phase 2a.
        # The model can still be inspected/initialized without errors.
        return None


__all__ = [
    "GTAPv62ModelEquations",
    "GTAP_V62_TAX_STREAMS",
]
