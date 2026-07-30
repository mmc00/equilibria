"""GTAP INCOME block.

Ports the monolith's regional-income, tax-revenue, GDP and welfare equations
VERBATIM from ``gtap_model_equations.py`` (7316-7739):

  eq_facty (7322, xfflag mask), eq_ytax (7341, per-stream `if gy==...` dispatch),
  eq_ytax_tot (7453), eq_ytax_ind (7459), eq_regy (7465), eq_ytaxshreq (7470),
  eq_yc (7476), eq_yg (7485), eq_yi (7494, Skip residual), eq_rsav (7507),
  eq_pabs (7618), eq_gdpmp (7640), eq_rgdpmp (7656), eq_pgdpmp (7681),
  eq_ev (7687), eq_cv (7715)

Params come from ``_derived_params.demand_income_params`` (the SAME 2258-2785
calibration loop pass unit 5 reads — dedup by name) + the shared FACTOR/
PRODUCTION Params (xfflag/xflag/prdtx_rai/fcttx/fctts/xscale). pabs/ev/cv are
shared with DEMAND+UTILITY (unit 5) — dedup by name (first registration wins).

BASE SNAPSHOTS (Fisher compStat): eq_pabs / eq_ev fold a base-period reference
(base_pa, base_xaa, base_pabs). On the BASE build (t0_snapshot None,
is_counterfactual False) the monolith snapshots these from the model's OWN Var
levels AT BUILD TIME — pa=1.0, pabs=1.0, xaa=the (post-_align_xi_xaa_post_scaling)
init. The block reproduces the benchmark seed (base_pa=1.0, base_pabs=1.0,
base_xaa = purchaser value). The 6 inv-agent xaa cells that
_align_xi_xaa_post_scaling nudges (~1e-7) make eq_pabs a COEFFICIENT-ONLY drift
carry (same Blocker-C family as gd/ge/gw/alpha; composer re-snapshots). eq_gdpmp
uses only LIVE prices/quantities; eq_rgdpmp on the base build is `rgdpmp==gdpmp`
(is_counterfactual False) — both carry-free. eq_ev's base_pa=1.0 folds exactly.

residual_region: eq_yi Skips it (the Walras residual, absorbed by CLOSURE's
eq_walras). Threaded from the composer (gtap7_3x3 comp-stat oracle uses ``ROW``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyomo.environ import sqrt, value

from equilibria.blocks.base import Block
from equilibria.blocks.gtap import _derived_params as dp
from equilibria.blocks.gtap import _ifsub_macros as mac
from equilibria.core.parameters import Parameter
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable

_FLOOR = 1e-8
_REL = 1e-3

_HHD, _GOV, _INV, _TMG = "hhd", "gov", "inv", "tmg"
_FINAL_AGENTS = (_HHD, _GOV, _INV, _TMG)
_MUTABLE = {"g_share", "i_share", "aus", "betap", "betag", "betas", "alphaa_hhd"}


def _price_floor(init: float) -> float:
    """Monolith relative floor: max(1e-8, 1e-3*init) for init>0 (5298-5379)."""
    if init is None or init <= 0.0:
        return _FLOOR
    return max(_FLOOR, _REL * float(init))


class IncomeBlock(Block):
    """GTAP regional income, tax-revenue streams, GDP (Fisher) and welfare."""

    name: str = "GTAP_INCOME"
    description: str = "GTAP regional income, tax streams, GDP and EV/CV welfare"
    sets: Any = None
    params: Any = None
    residual_region: str = "NAmerica"
    # ifSUB toggle: under if_sub, eq_ytax[mt] reads model.imptx (live, so the tariff
    # shock enters the SOLVED revenue) + M_PMCIF inline (pmcif is a frozen report var
    # under ifSUB); eq_gdpmp uses M_PEFOB/M_PMCIF inline for the same reason.
    if_sub: bool = False

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["r", "gy"]

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        regions = list(set_manager.get("r"))
        comms = list(set_manager.get("i"))
        acts = list(set_manager.get("a"))
        facs = list(set_manager.get("f"))
        aa = list(set_manager.get("aa"))
        gy = list(set_manager.get("gy"))
        p = self.params
        s = self.sets
        _byname = {"r": regions, "i": comms, "a": acts, "f": facs, "aa": aa, "gy": gy}
        nr = len(regions)

        # -------- calibration loop (monolith 2258-2785), shared with DEMAND ------
        calib = dp.demand_income_params(p, s, residual_region=self.residual_region)

        def _calib_param(name, doms):
            parameters[name] = Parameter(
                name=name,
                value=dp.to_array(calib[name], [_byname[d] for d in doms], 0.0),
                domains=tuple(doms),
                mutable=(name in _MUTABLE),
            )

        # Params read by this unit's bodies (dedup by name with unit 5 / FACTOR).
        _calib_param("betap", ("r",))
        _calib_param("betag", ("r",))
        _calib_param("betas", ("r",))
        _calib_param("depr", ("r",))
        _calib_param("fdepr", ("r",))
        _calib_param("pop", ("r",))
        _calib_param("c_share", ("r", "i"))
        _calib_param("alphaa_hhd", ("r", "i"))
        _calib_param("bh", ("r", "i"))
        _calib_param("eh", ("r", "i"))

        def _shared_param(name, data, doms, default=0.0, mutable=False):
            parameters[name] = Parameter(
                name=name,
                value=dp.to_array(data, [_byname[d] for d in doms], default),
                domains=tuple(doms),
                mutable=mutable,
            )

        # Shared PRODUCTION/FACTOR Params (same names/values/mutability as those
        # units — dedup by name; declared here so a standalone income build resolves
        # them). fcttx/fctts mutable (monolith 5077-5094), referenced unwrapped.
        _shared_param("xfflag", dp.xfflag_data(p, s), ("r", "f", "a"))
        _shared_param("xflag", dp.xflag_data(p, s), ("r", "a", "i"))
        _shared_param("prdtx_rai", dp.prdtx_rai_data(p, s), ("r", "a", "i"))
        _shared_param("fcttx", dp.fcttx_data(p, s), ("r", "f", "a"), mutable=True)
        _shared_param("fctts", dp.fctts_data(p, s), ("r", "f", "a"), mutable=True)
        _shared_param("xscale", dp.xscale_data(p, s), ("r", "aa"), default=1.0)

        # -------- Variables OWNED by this unit (monolith 4722-4842) -------------
        def _q(name, doms, init, lower=0.0, dom="NonNegativeReals"):
            variables[name] = Variable(
                name=name,
                value=init,
                domains=tuple(doms),
                domain=dom,
                lower=lower,
                upper=float("inf"),
            )

        def _price(name, doms, init):
            lo = np.vectorize(_price_floor)(init)
            variables[name] = Variable(
                name=name,
                value=init,
                domains=tuple(doms),
                domain="NonNegativeReals",
                lower=lo,
                upper=float("inf"),
            )

        ones_r = np.ones(nr)
        # Income vars: within=Reals FREE (monolith 4722-4749).
        regy_init = np.array([self._regy_init(r) for r in regions])
        _q("regy", ("r",), regy_init, lower=float("-inf"), dom="Reals")
        _q(
            "yc",
            ("r",),
            self._yc_init(regions, calib),
            lower=float("-inf"),
            dom="Reals",
        )
        _q(
            "yg",
            ("r",),
            self._yg_init(regions, calib),
            lower=float("-inf"),
            dom="Reals",
        )
        _q("yi", ("r",), self._yi_init(regions), lower=float("-inf"), dom="Reals")
        _q("rsav", ("r",), np.zeros(nr), lower=float("-inf"), dom="Reals")
        # facty: NonNeg lb=0 (4750).
        _q("facty", ("r",), np.array([self._facty_init(r) for r in regions]))
        # ytax/ytaxTot/ytax_ind/ytaxshr: within=Reals FREE (4756-4782).
        ny = len(gy)
        _q("ytax", ("r", "gy"), np.zeros((nr, ny)), lower=float("-inf"), dom="Reals")
        _q("ytaxTot", ("r",), np.zeros(nr), lower=float("-inf"), dom="Reals")
        _q("ytax_ind", ("r",), np.zeros(nr), lower=float("-inf"), dom="Reals")
        _q("ytaxshr", ("r", "gy"), np.zeros((nr, ny)), lower=float("-inf"), dom="Reals")
        # gdpmp/rgdpmp: NonNeg lb=0 (4828-4838). pgdpmp: price floor 1e-3 (4840).
        gdp_init = np.array([self._gdpmp_init(r) for r in regions])
        _q("gdpmp", ("r",), gdp_init)
        _q("rgdpmp", ("r",), gdp_init)
        _price("pgdpmp", ("r",), ones_r)
        # pabs: price floor 1e-3, init 1.0 (4786). Shared w/ FACTOR-eq_xfteq via
        # dedup — but INCOME OWNS it (monolith declares pabs in the income Var
        # block). ev/cv shared with DEMAND (unit 5) — dedup by name.
        _price("pabs", ("r",), ones_r)
        ev_init = np.array([max(self._yc_init_scalar(r, calib), 1e-8) for r in regions])
        _price("ev", ("r",), ev_init)
        _price("cv", ("r",), ev_init)

        # -------- base snapshots (Fisher compStat) — benchmark seed --------------
        # base_pa = 1.0, base_pabs = 1.0 (all prices=1 at benchmark). base_xaa =
        # purchaser value (post-_align nudge ~1e-7 => eq_pabs coefficient carry).
        base_pa: dict[tuple, float] = {}
        base_xaa: dict[tuple, float] = {}
        for r in regions:
            for i in comms:
                for agent in _FINAL_AGENTS:
                    base_pa[(r, i, agent)] = 1.0
                    base_xaa[(r, i, agent)] = max(
                        dp._xaa_purchaser_value(p, s, r, i, agent), 0.0
                    )
        base_pabs = dict.fromkeys(regions, 1.0)
        # base_pefob/base_pmcif = 1.0 at benchmark (all prices=1); base_xw = the
        # xw Var init (bilateral vxsb). Only used by the eq_rgdpmp counterfactual
        # branch (base build returns rgdpmp==gdpmp), so these never enter the base
        # gate — kept for parity with the monolith's _mqtrade signature.
        base_pefob: dict[tuple, float] = {}
        base_pmcif: dict[tuple, float] = {}
        base_xw: dict[tuple, float] = {}
        for r in regions:
            for i in comms:
                for rp in regions:
                    base_pefob[(r, i, rp)] = 1.0
                    base_pmcif[(rp, i, r)] = 1.0
                    base_xw[(r, i, rp)] = self._xw_init(r, i, rp)

        # base_mqabs_00[r] = sum_{i,agent} base_pa * base_xaa (monolith 2609-...).
        base_mqabs_00: dict[str, float] = {}
        for r in regions:
            base_mqabs_00[r] = sum(
                base_pa[(r, i, agent)] * base_xaa[(r, i, agent)]
                for i in comms
                for agent in _FINAL_AGENTS
            )

        # -------- inline-python accessors (read exactly as the monolith) --------
        taxes = p.taxes
        if_sub = self.if_sub
        acts_comm = s.activity_commodities
        residual_region = self.residual_region
        residual_regions = tuple(r for r in regions if str(r) == residual_region)
        # if_sub=False (comp-stat oracle) => _m_pmcif -> model.pmcif; etax/mtax = 0.
        comms_list = comms
        agents_final = _FINAL_AGENTS

        def _etax_value(exporter, commodity, importer):
            return 0.0  # etax fixed 0 at benchmark (comp-stat)

        def _mtax_value(importer, commodity, exporter):
            return 0.0  # mtax fixed 0 at benchmark (comp-stat)

        equations: list[SymbolicEquation] = []

        # ---------------- eq_facty (monolith 7322) ----------------
        class EqFacty(SymbolicEquation):
            name: str = "eq_facty"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                (r,) = indices
                return model.facty[r] == (
                    sum(
                        model.pf[r, f, a] * model.xf[r, f, a] / model.xscale[r, a]
                        for f in model.f
                        for a in model.a
                        if value(model.xfflag[r, f, a]) > 0.0
                    )
                    - model.fdepr[r] * model.pi[r] * model.kstock[r]
                )

        equations.append(EqFacty())

        # ---------------- eq_ytax (monolith 7341) ----------------
        class EqYtax(SymbolicEquation):
            name: str = "eq_ytax"
            domains: tuple = ("r", "gy")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, gy = indices
                if gy == "pt":
                    total = 0.0
                    for a in model.a:
                        for i in acts_comm.get(a, list(model.i)):
                            if value(model.xflag[r, a, i]) <= 0.0:
                                continue
                            total += (
                                model.prdtx_rai[r, a, i]
                                * model.p_rai[r, a, i]
                                * model.x[r, a, i]
                            )
                    return model.ytax[r, gy] == total

                if gy == "ft":
                    total = 0.0
                    for rr, f, a in [
                        (rr, f, a) for (rr, f, a) in model.fcttx if rr == r
                    ]:
                        total += (
                            model.fcttx[rr, f, a]
                            * model.pf[r, f, a]
                            * model.xf[r, f, a]
                            / model.xscale[r, a]
                        )
                    return model.ytax[r, gy] == total

                if gy == "fs":
                    total = 0.0
                    for rr, f, a in [
                        (rr, f, a) for (rr, f, a) in model.fctts if rr == r
                    ]:
                        total += (
                            model.fctts[rr, f, a]
                            * model.pf[r, f, a]
                            * model.xf[r, f, a]
                            / model.xscale[r, a]
                        )
                    return model.ytax[r, gy] == total

                if gy in ("fc", "pc", "gc", "ic"):
                    if gy == "fc":
                        agents = list(model.a)
                    elif gy == "pc":
                        agents = [_HHD]
                    elif gy == "gc":
                        agents = [_GOV]
                    else:
                        agents = [_INV]
                    total = 0.0
                    for ag in agents:
                        for i in model.i:
                            dintx = float(taxes.dintx0.get((r, i, ag), 0.0))
                            mintx = float(taxes.mintx0.get((r, i, ag), 0.0))
                            scale = model.xscale[r, ag] if ag in model.a else 1.0
                            total += (
                                dintx * model.pd[r, i] * model.xda[r, i, ag] / scale
                            )
                            total += (
                                mintx * model.pmt[r, i] * model.xma[r, i, ag] / scale
                            )
                    return model.ytax[r, gy] == total

                if gy == "et":
                    total = 0.0
                    for (rr, i, rp), rtxs in taxes.rtxs.items():
                        if rr != r:
                            continue
                        etax = _etax_value(r, i, rp)
                        total += (
                            (float(rtxs) + etax)
                            * model.pe[r, i, rp]
                            * model.xw[r, i, rp]
                        )
                    return model.ytax[r, gy] == total

                if gy == "mt":
                    total = 0.0
                    for (exporter, i, importer), imptx in taxes.imptx.items():
                        if importer != r:
                            continue
                        mtax = _mtax_value(r, i, exporter)
                        # GAMS ytaxeq (8686): (imptx + mtax)·M_PMCIF·xw. Under ifSUB
                        # read imptx LIVE (model.imptx) so the tariff shock enters the
                        # solved revenue, and M_PMCIF inline (pmcif is a frozen report
                        # var). if_sub=False keeps the baked float + plain pmcif var.
                        imptx_term = (
                            model.imptx[exporter, i, r] if if_sub else float(imptx)
                        )
                        pmcif_term = (
                            mac.m_pmcif(model, p, exporter, i, r)
                            if if_sub
                            else model.pmcif[exporter, i, r]
                        )
                        total += (
                            (imptx_term + mtax) * pmcif_term * model.xw[exporter, i, r]
                        )
                    return model.ytax[r, gy] == total

                if gy == "dt":
                    total = 0.0
                    for f in model.f:
                        for a in model.a:
                            kappa = float(taxes.kappaf_activity.get((r, f, a), 0.0))
                            if kappa == 0.0:
                                kappa = float(taxes.kappaf.get((r, f), 0.0))
                            if kappa == 0.0:
                                continue
                            total += (
                                kappa
                                * model.pf[r, f, a]
                                * model.xf[r, f, a]
                                / model.xscale[r, a]
                            )
                    return model.ytax[r, gy] == total

                return model.ytax[r, gy] == 0.0

        equations.append(EqYtax())

        # ---------------- eq_ytax_tot (monolith 7453) ----------------
        class EqYtaxTot(SymbolicEquation):
            name: str = "eq_ytax_tot"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                (r,) = indices
                return model.ytaxTot[r] == sum(model.ytax[r, gy] for gy in model.gy)

        equations.append(EqYtaxTot())

        # ---------------- eq_ytax_ind (monolith 7459) ----------------
        class EqYtaxInd(SymbolicEquation):
            name: str = "eq_ytax_ind"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                (r,) = indices
                return model.ytax_ind[r] == model.ytaxTot[r] - model.ytax[r, "dt"]

        equations.append(EqYtaxInd())

        # ---------------- eq_regy (monolith 7465) ----------------
        class EqRegy(SymbolicEquation):
            name: str = "eq_regy"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                (r,) = indices
                return model.regy[r] == model.facty[r] + model.ytax_ind[r]

        equations.append(EqRegy())

        # ---------------- eq_ytaxshreq (monolith 7470) ----------------
        class EqYtaxshr(SymbolicEquation):
            name: str = "eq_ytaxshreq"
            domains: tuple = ("r", "gy")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, gy = indices
                return model.ytaxshr[r, gy] == model.ytax[r, gy] / (
                    model.regy[r] + 1e-12
                )

        equations.append(EqYtaxshr())

        # ---------------- eq_yc (monolith 7476) ----------------
        class EqYc(SymbolicEquation):
            name: str = "eq_yc"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                (r,) = indices
                return (
                    model.yc[r]
                    == model.betap[r] * (model.phi[r] / model.phip[r]) * model.regy[r]
                )

        equations.append(EqYc())

        # ---------------- eq_yg (monolith 7485) ----------------
        class EqYg(SymbolicEquation):
            name: str = "eq_yg"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                (r,) = indices
                return model.yg[r] == model.betag[r] * model.phi[r] * model.regy[r]

        equations.append(EqYg())

        # ---------------- eq_yi (monolith 7494) ----------------
        class EqYi(SymbolicEquation):
            name: str = "eq_yi"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                (r,) = indices
                if str(r) in residual_regions:
                    return None
                return (
                    model.yi[r]
                    == model.pi[r] * model.depr[r] * model.kstock[r]
                    + model.rsav[r]
                    + model.savf[r]
                )

        equations.append(EqYi())

        # ---------------- eq_rsav (monolith 7507) ----------------
        class EqRsav(SymbolicEquation):
            name: str = "eq_rsav"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                (r,) = indices
                return model.rsav[r] == model.betas[r] * model.phi[r] * model.regy[r]

        equations.append(EqRsav())

        # -------- Fisher mq helpers (monolith 7555-7616), if_sub=False form ------
        def _mqabs(model, region, *, price_base, quantity_base):
            total = 0.0
            for i in comms_list:
                for agent in agents_final:
                    pa = (
                        base_pa[(region, i, agent)]
                        if price_base
                        else model.pa[region, i, agent]
                    )
                    xa = (
                        base_xaa[(region, i, agent)]
                        if quantity_base
                        else model.xaa[region, i, agent]
                    )
                    total += pa * xa
            return total

        def _mqtrade(model, region, *, price_base, quantity_base):
            # M_PEFOB/M_PMCIF (GAMS balance 9308): plain pefob/pmcif report vars under
            # if_sub=False; under if_sub=True inline (frozen report vars) so the
            # export/import price wedge feeds the trade balance / GDP correctly.
            # base_pefob/base_pmcif/base_xw = 1.0/1.0/xw-init at benchmark; only the
            # eq_rgdpmp counterfactual uses price_base/quantity_base=True (base build
            # returns rgdpmp==gdpmp), so eq_gdpmp only ever hits the all-live branch.
            total = 0.0
            for i in comms_list:
                for rp in model.rp:
                    if price_base:
                        pexp = base_pefob[(region, i, rp)]
                        pimp = base_pmcif[(rp, i, region)]
                    elif if_sub:
                        pexp = mac.m_pefob(model, p, region, i, rp)
                        pimp = mac.m_pmcif(model, p, rp, i, region)
                    else:
                        pexp = model.pefob[region, i, rp]
                        pimp = model.pmcif[rp, i, region]
                    xexp = (
                        base_xw[(region, i, rp)]
                        if quantity_base
                        else model.xw[region, i, rp]
                    )
                    ximp = (
                        base_xw[(rp, i, region)]
                        if quantity_base
                        else model.xw[rp, i, region]
                    )
                    total += pexp * xexp - pimp * ximp
            return total

        def _mqgdp(model, region, *, price_base, quantity_base):
            return _mqabs(
                model, region, price_base=price_base, quantity_base=quantity_base
            ) + _mqtrade(
                model, region, price_base=price_base, quantity_base=quantity_base
            )

        # ---------------- eq_pabs (monolith 7618) ----------------
        class EqPabs(SymbolicEquation):
            name: str = "eq_pabs"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                (r,) = indices
                mqabs_00 = base_mqabs_00[r]
                if mqabs_00 <= 1e-12:
                    return None
                mqabs_t0 = _mqabs(model, r, price_base=False, quantity_base=True)
                mqabs_tt = _mqabs(model, r, price_base=False, quantity_base=False)
                mqabs_0t = _mqabs(model, r, price_base=True, quantity_base=False)
                return model.pabs[r] == base_pabs[r] * sqrt(
                    (mqabs_t0 / mqabs_00) * (mqabs_tt / (mqabs_0t + 1e-12))
                )

        equations.append(EqPabs())

        # ---------------- eq_gdpmp (monolith 7640) ----------------
        class EqGdpmp(SymbolicEquation):
            name: str = "eq_gdpmp"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                (r,) = indices
                return model.gdpmp[r] == _mqgdp(
                    model, r, price_base=False, quantity_base=False
                )

        equations.append(EqGdpmp())

        # ---------------- eq_rgdpmp (monolith 7656) ----------------
        # Base build: is_counterfactual False, t0_snapshot None => rgdpmp == gdpmp.
        class EqRgdpmp(SymbolicEquation):
            name: str = "eq_rgdpmp"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                (r,) = indices
                return model.rgdpmp[r] == model.gdpmp[r]

        equations.append(EqRgdpmp())

        # ---------------- eq_pgdpmp (monolith 7681) ----------------
        class EqPgdpmp(SymbolicEquation):
            name: str = "eq_pgdpmp"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                (r,) = indices
                return model.pgdpmp[r] * model.rgdpmp[r] == model.gdpmp[r]

        equations.append(EqPgdpmp())

        # ---------------- eq_ev (monolith 7687) ----------------
        class EqEv(SymbolicEquation):
            name: str = "eq_ev"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                (r,) = indices
                terms = []
                for i in model.i:
                    alpha = value(model.alphaa_hhd[r, i])
                    bh = value(model.bh[r, i])
                    eh = value(model.eh[r, i])
                    share = value(model.c_share[r, i])
                    if alpha <= 0.0 or share <= 0.0:
                        continue
                    pa0 = base_pa.get((r, i, _HHD), 1.0)
                    terms.append(
                        alpha
                        * (model.uh[r] ** (bh * eh))
                        * ((pa0 * model.pop[r] / model.ev[r]) ** bh)
                    )
                if not terms:
                    return None
                return sum(terms) == 1.0

        equations.append(EqEv())

        # ---------------- eq_cv (monolith 7715) ----------------
        class EqCv(SymbolicEquation):
            name: str = "eq_cv"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                (r,) = indices
                terms = []
                for i in model.i:
                    alpha = value(model.alphaa_hhd[r, i])
                    bh = value(model.bh[r, i])
                    share = value(model.c_share[r, i])
                    if alpha <= 0.0 or share <= 0.0:
                        continue
                    terms.append(
                        alpha
                        * ((model.pa[r, i, _HHD] * model.pop[r] / model.cv[r]) ** bh)
                    )
                if not terms:
                    return None
                return sum(terms) == 1.0

        equations.append(EqCv())

        return equations

    # ------------------------------------------------------------------
    # init helpers (levels only — do not affect the form/domain gates
    # except pabs/pgdpmp/ev/cv floors, which are price-floor bearing)
    # ------------------------------------------------------------------
    def _facty_init(self, r):
        bm = self.params.benchmark
        fi = sum(bm.vfm.get((r, f, a), 0.0) for f in self.sets.f for a in self.sets.a)
        dep = float(bm.vdep.get(r, 0.0))
        return max(fi - dep, 0.0)

    def _regy_init(self, r):
        return self._facty_init(r) + dp._compute_ytax_ind_bench(
            self.params, self.sets, r
        )

    def _yc_init_scalar(self, r, calib):
        betap = calib["betap"].get((r,), 0.0)
        phi = calib["phi0"].get((r,), 1.0)
        phip = calib["phip0"].get((r,), 1.0)
        regy = calib["regy_bench"].get((r,), 0.0)
        return betap * (phi / max(phip, 1e-12)) * regy

    def _yc_init(self, regions, calib):
        return np.array([self._yc_init_scalar(r, calib) for r in regions])

    def _yg_init(self, regions, calib):
        out = []
        for r in regions:
            betag = calib["betag"].get((r,), 0.0)
            phi = calib["phi0"].get((r,), 1.0)
            regy = calib["regy_bench"].get((r,), 0.0)
            out.append(betag * phi * regy)
        return np.array(out)

    def _yi_init(self, regions):
        # get_benchmark_yi = Σ_i get_vim_init(r,i), and get_vim_init =
        # max(get_investment_demand(r,i)[0], 1e-8) — the RESOLVED investment
        # Armington demand (_resolve_final_demand_split of vim/vdip/vmip), NOT the
        # raw vim matrix. Reading raw vim left yi ~16% low (the yi→xiagg→xi→xd
        # cascade), so mirror the monolith exactly.
        bm = self.params.benchmark
        out = []
        for r in regions:
            out.append(
                sum(max(bm.get_investment_demand(r, i)[0], 1e-8) for i in self.sets.i)
            )
        return np.array(out)

    def _xw_init(self, r, i, rp):
        """get_xw_init benchmark: vxsb/pe (pe=1) — for base_xw snapshot."""
        return float(self.params.benchmark.vxsb.get((r, i, rp), 0.0) or 0.0)

    def _gdpmp_init(self, r):
        # get_gdpmp_init benchmark: sum absorption (purchaser value) + net trade.
        # A benchmark seed; init affects only the (0,None)-bounded gdpmp gate-free.
        bm = self.params.benchmark
        absb = sum(
            dp._xaa_purchaser_value(self.params, self.sets, r, i, a)
            for i in self.sets.i
            for a in (_HHD, _GOV, _INV, _TMG)
        )
        return max(absb, 1e-8)
