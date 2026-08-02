"""GTAP DEMAND + UTILITY block (units merged — CDE demand + utility/savings nest).

Ports the monolith's final-demand and utility/savings equations VERBATIM from
``gtap_model_equations.py`` (6921-7313):

  DEMAND:  eq_xc (6927), eq_xg (6949), eq_xi (6968), eq_xiagg (6991)
  UTILITY: eq_zcons (7002), eq_xcshr (7019), eq_phip (7037), eq_phi (7047),
           eq_pcons (7061), eq_pi (7072), eq_uh (7093), eq_pg (7131),
           eq_xg_agg (7146), eq_ug (7159), eq_psave (7165), eq_us (7170),
           eq_u (7178), eq_chisave (7192), eq_pigbl (7202), eq_xigbl (7212),
           eq_savf (7220), eq_chif (7240), eq_capAcct (7251), eq_kapEnd (7257),
           eq_arent (7265), eq_rorc (7285), eq_rore (7292), eq_rorg (7302)

The Params these bodies read come from ``_derived_params.demand_income_params``
(a verbatim transcription of the monolith's 2258-2785 income/demand/beta
calibration loop) — shared with INCOME (unit 6), so both read ONE pass.

CDE private demand (eq_zcons/eq_xcshr/eq_phip/eq_uh) uses ``alphaa_hhd``/``bh``/
``eh``/``pop`` Params; the beta split (eq_phi/eq_yc-family) uses
``betap``/``betag``/``betas``. MUTABILITY is transcribed from the monolith
create_indexed_param calls (2726-2785): g_share/i_share/aus/betap/betag/betas/
alphaa_hhd are mutable (referenced unwrapped -> stay symbolic), the rest fold.

Owned vars carry the monolith's exact domains + the runtime relative price
floor (1e-3*init) on the price/utility/level vars listed in the floor sweep
(5305-5385). kapEnd/arent/rorc/rore are OWNED BY FACTOR (unit 3) — this unit
only adds their EQUATIONS (dedup by name; FACTOR's Var wins). pabs/ev/cv are
shared with INCOME (unit 6) — dedup by name.

residual_region: eq_savf (capFix) Skips the residual region; the composer
passes the real residual region in Task 5. The gate oracle (gtap7_3x3
comp-stat) uses ``ROW`` (last region).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyomo.environ import sqrt, value

from equilibria.blocks.base import Block
from equilibria.blocks.gtap import _derived_params as dp
from equilibria.core.parameters import Parameter
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable

_FLOOR = 1e-8
_REL = 1e-3

# Mutable Params (monolith create_indexed_param mutable=True, 2734-2779).
_MUTABLE = {"g_share", "i_share", "aus", "betap", "betag", "betas", "alphaa_hhd"}


def _price_floor(init: float) -> float:
    """Monolith relative floor: max(1e-8, 1e-3*init) for init>0 (5298-5379)."""
    if init is None or init <= 0.0:
        return _FLOOR
    return max(_FLOOR, _REL * float(init))


class DemandUtilityBlock(Block):
    """GTAP final demand (CDE) + utility/savings/rate-of-return nest."""

    name: str = "GTAP_DEMAND_UTILITY"
    description: str = "GTAP CDE final demand + utility/savings/rate-of-return"
    sets: Any = None
    params: Any = None
    residual_region: str = "NAmerica"
    savf_flag: str = (
        "capFix"  # capital-account closure (GAMS RoRFlag); capFlex = returns equalize
    )

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["r", "i"]

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        regions = list(set_manager.get("r"))
        comms = list(set_manager.get("i"))
        p = self.params
        s = self.sets
        _byname = {"r": regions, "i": comms}
        nr, ni = len(regions), len(comms)

        # -------- calibration loop (monolith 2258-2785), shared with INCOME -----
        calib = dp.demand_income_params(p, s, residual_region=self.residual_region)

        def _param(name, doms, calib_key=None):
            key = calib_key or name
            parameters[name] = Parameter(
                name=name,
                value=dp.to_array(calib[key], [_byname[d] for d in doms], 0.0),
                domains=tuple(doms),
                mutable=(name in _MUTABLE),
            )

        # Params read by this unit's bodies (same names/values/mutability as the
        # monolith create_indexed_param calls 2726-2785).
        _param("c_share", ("r", "i"))
        _param("g_share", ("r", "i"))
        _param("i_share", ("r", "i"))
        _param("alphaa_hhd", ("r", "i"))
        _param("bh", ("r", "i"))
        _param("eh", ("r", "i"))
        _param("axi", ("r",))
        _param("invwgt", ("r",))
        _param("savwgt", ("r",))
        _param("aug", ("r",))
        _param("aus", ("r",))
        _param("au", ("r",))
        _param("betap", ("r",))
        _param("betag", ("r",))
        # betas: folded Param in the standard closures; under capFixDp (dpsave endogenous)
        # it becomes a free Variable (the saving DISTRIBUTION adjusts). Declared as a Var
        # below in that case, seeded at the calibrated share (GAMS cal.gms:621).
        if self.savf_flag != "capFixDp":
            _param("betas", ("r",))
        _param("pop", ("r",))
        _param("depr", ("r",))
        _param("fdepr", ("r",))
        _param("rorflex", ("r",))
        _param("chif0", ("r",))
        _param("savf_bar", ("r",))

        # risk[r] = rorg/rore at benchmark (GAMS cal.gms:676), used by the capFlex
        # capital-account closure (savfeq: risk*rore == rorg). Calibrated from a capFix
        # twin solve by build_block_model._calibrate_capflex_risk and stashed on
        # params._capflex_risk; falls back to 1.0 (uncalibrated) if absent.
        if self.savf_flag == "capFlex":
            _risk_cal = getattr(p, "_capflex_risk", None) or {}
            risk_vals = np.array(
                [float(_risk_cal.get(r, 1.0)) for r in regions], dtype=float
            )
            parameters["risk"] = Parameter(
                name="risk",
                value=risk_vals,
                domains=("r",),
                mutable=True,
            )

        # -------- Variables OWNED by this unit (monolith 4454-4969) -------------
        # Quantity vars (bounds (0,None) — init does not affect the gate).
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

        # capFixDp: betas is a free Variable (dpsave endogenous), seeded at its calibrated
        # benchmark share rsav/(phi·regY) (GAMS cal.gms:621), available as calib["betas"].
        if self.savf_flag == "capFixDp":
            _q(
                "betas",
                ("r",),
                np.array(
                    [float(calib["betas"].get((r,), 0.1)) for r in regions], dtype=float
                ),
                lower=float("-inf"),
                dom="Reals",
            )

        ones_r = np.ones(nr)
        ones_ri = np.ones((nr, ni))
        # Final demand (NonNeg, lb=0; init benchmark, does not affect gate).
        _q("xc", ("r", "i"), self._xc_init(regions, comms))
        _q("xg", ("r", "i"), self._vgm_init(regions, comms))
        _q("xi", ("r", "i"), self._xi_init(regions, comms))
        # xiagg: strictly_positive floor 1e-8 (5333-5348, pass 1 only).
        _q("xiagg", ("r",), np.maximum(self._xiagg_init(regions), _FLOOR), lower=_FLOOR)
        # Price/utility/level vars (relative floor 1e-3*init).
        _price("pcons", ("r",), ones_r)
        _price("pi", ("r",), self._pi_init(regions))
        _price("pg", ("r",), ones_r)
        _price("uh", ("r",), ones_r)
        _price("ug", ("r",), ones_r)
        _price("us", ("r",), ones_r)
        _price("u", ("r",), ones_r)
        _price("psave", ("r",), ones_r)
        # xcshr/zcons/phip/phi/xg_agg: NonNeg, NO floor (not in any floor list).
        _q("xcshr", ("r", "i"), self._xcshr_init(regions, comms))
        _q("zcons", ("r", "i"), self._zcons_init(regions, comms, calib))
        _q("phip", ("r",), self._phip_init(regions, calib))
        _q("phi", ("r",), self._phi_init(regions, calib))
        _q("xg_agg", ("r",), ones_r)
        # savf/chif: within=Reals FREE (4928-4936).
        # savf init = savf_bar (get_savf_init, capFix closure). Left at zeros this
        # broke the yi = pi·depr·kstock + rsav + savf identity (_refresh_macro_
        # initial_state), cascading yi→xiagg→xi→xd ~16% low. The GAMS GDX does not
        # seed savf at benchmark, so the calibration init must be correct here.
        _q(
            "savf",
            ("r",),
            self._savf_init(regions, calib),
            lower=float("-inf"),
            dom="Reals",
        )
        _q(
            "chif",
            ("r",),
            self._chif_init(regions, calib),
            lower=float("-inf"),
            dom="Reals",
        )
        # ev/cv: per-cell relative floor 1e-3*init (init ~ yc). Shared w/ INCOME.
        ev_init = self._ev_init(regions, calib)
        _price("ev", ("r",), ev_init)
        _price("cv", ("r",), ev_init)
        # xigbl/pigbl/chiSave/rorg scalars.
        _price("xigbl", (), np.array([self._xigbl_init(regions)]))
        _price("pigbl", (), np.array([self._pigbl_init(regions)]))
        _price("chiSave", (), np.array([1.0]))
        _price("rorg", (), np.array([self._rorg_init(regions)]))

        # -------- inline-python accessors (read exactly as the monolith) --------
        el = p.elasticities
        residual_region = self.residual_region
        savf_flag = self.savf_flag  # capital-account closure (GAMS RoRFlag)

        equations: list[SymbolicEquation] = []

        # ---------------- eq_xc (monolith 6927) ----------------
        class EqXc(SymbolicEquation):
            name: str = "eq_xc"
            domains: tuple = ("r", "i")

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                r, i = indices
                share = value(m.c_share[r, i])
                if share <= 0.0:
                    return m.xc[r, i] == 0.0
                return m.pa[r, i, "hhd"] * m.xc[r, i] == m.xcshr[r, i] * m.yc[r]

        equations.append(EqXc())

        # ---------------- eq_xg (monolith 6949) ----------------
        class EqXg(SymbolicEquation):
            name: str = "eq_xg"
            domains: tuple = ("r", "i")

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                r, i = indices
                share = value(m.g_share[r, i])
                if share <= 0.0:
                    return m.xg[r, i] == 0.0
                sigmag = float(el.esubg.get(r, 1.0))
                if abs(sigmag - 1.0) < 1e-8:
                    sigmag = 1.01
                return (
                    m.xg[r, i]
                    == share
                    * m.xg_agg[r]
                    * (m.pg[r] / (m.pa[r, i, "gov"] + 1e-12)) ** sigmag
                )

        equations.append(EqXg())

        # ---------------- eq_xi (monolith 6968) ----------------
        class EqXi(SymbolicEquation):
            name: str = "eq_xi"
            domains: tuple = ("r", "i")

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                r, i = indices
                alphaa = value(m.i_share[r, i])
                if alphaa <= 0.0:
                    return m.xi[r, i] == 0.0
                sigmai_raw = el.esubi.get((r,))
                if sigmai_raw is None:
                    sigmai_raw = el.esubi.get(r, 0.0)
                sigmai_raw = float(sigmai_raw or 0.0)
                if abs(sigmai_raw - 1.0) < 1e-8:
                    sigmai_raw = 1.01
                return (
                    m.xi[r, i]
                    == alphaa
                    * m.xiagg[r]
                    * (m.pi[r] / (m.pa[r, i, "inv"] + 1e-12)) ** sigmai_raw
                )

        equations.append(EqXi())

        # ---------------- eq_xiagg (monolith 6991) ----------------
        class EqXiagg(SymbolicEquation):
            name: str = "eq_xiagg"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                return m.pi[r] * m.xiagg[r] == m.yi[r]

        equations.append(EqXiagg())

        # ---------------- eq_zcons (monolith 7002) ----------------
        class EqZcons(SymbolicEquation):
            name: str = "eq_zcons"
            domains: tuple = ("r", "i")

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                r, i = indices
                share = value(m.c_share[r, i])
                alpha = value(m.alphaa_hhd[r, i])
                if share <= 0.0 or alpha <= 0.0:
                    return m.zcons[r, i] == 0.0
                return m.zcons[r, i] == (
                    m.alphaa_hhd[r, i]
                    * m.bh[r, i]
                    * (m.pa[r, i, "hhd"] ** m.bh[r, i])
                    * (m.uh[r] ** (m.eh[r, i] * m.bh[r, i]))
                    * ((m.yc[r] / m.pop[r]) ** (-m.bh[r, i]))
                )

        equations.append(EqZcons())

        # ---------------- eq_xcshr (monolith 7019) ----------------
        class EqXcshr(SymbolicEquation):
            name: str = "eq_xcshr"
            domains: tuple = ("r", "i")

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                r, i = indices
                share = value(m.c_share[r, i])
                if share <= 0.0:
                    return m.xcshr[r, i] == 0.0
                return (
                    m.xcshr[r, i]
                    * sum(m.zcons[r, j] for j in m.i if value(m.c_share[r, j]) > 0.0)
                    == m.zcons[r, i]
                )

        equations.append(EqXcshr())

        # ---------------- eq_phip (monolith 7037) ----------------
        class EqPhip(SymbolicEquation):
            name: str = "eq_phip"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                return m.phip[r] == sum(
                    m.xcshr[r, i] * m.eh[r, i]
                    for i in m.i
                    if value(m.c_share[r, i]) > 0.0
                )

        equations.append(EqPhip())

        # ---------------- eq_phi (monolith 7047) ----------------
        class EqPhi(SymbolicEquation):
            name: str = "eq_phi"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                return (
                    m.phi[r]
                    * (m.betap[r] / (m.phip[r] + 1e-12) + m.betag[r] + m.betas[r])
                    == 1.0
                )

        equations.append(EqPhi())

        # ---------------- eq_pcons (monolith 7061) ----------------
        class EqPcons(SymbolicEquation):
            name: str = "eq_pcons"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                return m.pcons[r] == sum(m.xcshr[r, i] * m.pa[r, i, "hhd"] for i in m.i)

        equations.append(EqPcons())

        # ---------------- eq_pi (monolith 7072) ----------------
        class EqPi(SymbolicEquation):
            name: str = "eq_pi"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                sigmai_raw = float(el.esubi.get(r, 0.0))
                if abs(sigmai_raw - 1.0) < 1e-8:
                    sigmai_raw = 1.01
                expo = 1.0 - sigmai_raw
                terms = [
                    value(m.i_share[r, i]) * m.pa[r, i, "inv"] ** expo
                    for i in m.i
                    if value(m.i_share[r, i]) > 0.0
                ]
                if not terms:
                    return m.pi[r] == 1.0
                return (m.axi[r] * m.pi[r]) ** expo == sum(terms)

        equations.append(EqPi())

        # ---------------- eq_uh (monolith 7093) ----------------
        class EqUh(SymbolicEquation):
            name: str = "eq_uh"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                terms = []
                for i in m.i:
                    if value(m.c_share[r, i]) <= 0.0:
                        continue
                    alphaa = m.alphaa_hhd[r, i]
                    bh = m.bh[r, i]
                    eh = m.eh[r, i]
                    pa = m.pa[r, i, "hhd"]
                    yc = m.yc[r]
                    pop = m.pop[r]
                    terms.append(
                        alphaa
                        * (pa**bh)
                        * (m.uh[r] ** (eh * bh))
                        * ((yc / pop) ** (-bh))
                    )
                if not terms:
                    return m.uh[r] == 1.0
                return sum(terms) == 1.0

        equations.append(EqUh())

        # ---------------- eq_pg (monolith 7131) ----------------
        class EqPg(SymbolicEquation):
            name: str = "eq_pg"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                sigmag = float(el.esubg.get(r, 1.0))
                if abs(sigmag - 1.0) < 1e-8:
                    sigmag = 1.01
                expo = 1.0 - sigmag
                pg_terms = sum(
                    value(m.g_share[r, i]) * m.pa[r, i, "gov"] ** expo
                    for i in m.i
                    if value(m.g_share[r, i]) > 0.0
                )
                return m.pg[r] ** expo == pg_terms

        equations.append(EqPg())

        # ---------------- eq_xg_agg (monolith 7146) ----------------
        class EqXgAgg(SymbolicEquation):
            name: str = "eq_xg_agg"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                return m.pg[r] * m.xg_agg[r] == m.yg[r]

        equations.append(EqXgAgg())

        # ---------------- eq_ug (monolith 7159) ----------------
        class EqUg(SymbolicEquation):
            name: str = "eq_ug"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                return m.ug[r] == m.aug[r] * m.xg_agg[r] / m.pop[r]

        equations.append(EqUg())

        # ---------------- eq_psave (monolith 7165) ----------------
        class EqPsave(SymbolicEquation):
            name: str = "eq_psave"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                return m.psave[r] == m.chiSave * m.pi[r]

        equations.append(EqPsave())

        # ---------------- eq_us (monolith 7170) ----------------
        class EqUs(SymbolicEquation):
            name: str = "eq_us"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                return m.us[r] == m.aus[r] * m.rsav[r] / (m.psave[r] * m.pop[r] + 1e-12)

        equations.append(EqUs())

        # ---------------- eq_u (monolith 7178) ----------------
        class EqU(SymbolicEquation):
            name: str = "eq_u"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                bp = value(m.betap[r])
                bg = value(m.betag[r])
                bs = value(m.betas[r])
                rhs = m.au[r] * (m.uh[r] ** bp) * (m.ug[r] ** bg)
                if abs(bs) > 1e-12:
                    rhs = rhs * (m.us[r] ** bs)
                return m.u[r] == rhs

        equations.append(EqU())

        # ---------------- eq_chisave (monolith 7192) ----------------
        class EqChisave(SymbolicEquation):
            name: str = "eq_chisave"
            domains: tuple = ()

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                numer = sum(m.invwgt[r] * m.pi[r] for r in m.r)
                denom = sum(m.savwgt[r] * m.pi[r] for r in m.r)
                return (m.chiSave**2) * denom == numer

        equations.append(EqChisave())

        # ---------------- eq_pigbl (monolith 7202) ----------------
        class EqPigbl(SymbolicEquation):
            name: str = "eq_pigbl"
            domains: tuple = ()

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                net_inv_value = sum(
                    m.pi[r] * (m.xiagg[r] - m.depr[r] * m.kstock[r]) for r in m.r
                )
                return m.pigbl * m.xigbl == net_inv_value

        equations.append(EqPigbl())

        # ---------------- eq_xigbl (monolith 7212) ----------------
        class EqXigbl(SymbolicEquation):
            name: str = "eq_xigbl"
            domains: tuple = ()

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                return m.xigbl == sum(m.xiagg[r] - m.depr[r] * m.kstock[r] for r in m.r)

        equations.append(EqXigbl())

        # ---------------- eq_savf (monolith 7220) ----------------
        class EqSavf(SymbolicEquation):
            name: str = "eq_savf"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                is_residual = str(r) == residual_region
                if savf_flag == "capFlex":
                    # GAMS savfeq under capFlex (model.gms:1161-1163): returns equalize
                    # across ALL regions (no residual skip) — risk[r]*rore[r] == rorg.
                    return m.risk[r] * m.rore[r] == m.rorg
                if savf_flag == "capFix":
                    if is_residual:
                        return None
                    return m.savf[r] == m.pigbl * m.savf_bar[r]
                if savf_flag == "capSFix":
                    if is_residual:
                        return None
                    return m.savf[r] == m.chif[r] * m.regy[r]
                return m.savf[r] == m.chif[r] * m.regy[r]

        equations.append(EqSavf())

        # ---------------- eq_chif (monolith 7240) ----------------
        class EqChif(SymbolicEquation):
            name: str = "eq_chif"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                if savf_flag == "capSFix":
                    return m.chif[r] == m.chif0[r]
                return m.savf[r] == m.chif[r] * m.regy[r]

        equations.append(EqChif())

        # ---------------- eq_capAcct (monolith 7251) ----------------
        class EqCapAcct(SymbolicEquation):
            name: str = "eq_capAcct"
            domains: tuple = ()

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                return sum(m.savf[r] for r in m.r) == 0.0

        equations.append(EqCapAcct())

        # ---------------- eq_kapEnd (monolith 7257) ----------------
        class EqKapEnd(SymbolicEquation):
            name: str = "eq_kapEnd"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                return m.kapEnd[r] == (1.0 - m.depr[r]) * m.kstock[r] + m.xiagg[r]

        equations.append(EqKapEnd())

        # ---------------- eq_arent (monolith 7265) ----------------
        taxes = p.taxes

        class EqArent(SymbolicEquation):
            name: str = "eq_arent"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                capital_factors = [
                    f for f in m.f if str(f).lower() in ("capital", "cap", "k", "kap")
                ]
                if not capital_factors:
                    return m.arent[r] == 0.0
                cap_return = 0.0
                for f in capital_factors:
                    for a in m.a:
                        kappa = float(taxes.kappaf_activity.get((r, f, a), 0.0))
                        cap_return += (
                            (1.0 - kappa)
                            * m.pf[r, f, a]
                            * m.xf[r, f, a]
                            / m.xscale[r, a]
                        )
                return m.arent[r] == cap_return / (m.kstock[r] + 1e-12)

        equations.append(EqArent())

        # ---------------- eq_rorc (monolith 7285) ----------------
        class EqRorc(SymbolicEquation):
            name: str = "eq_rorc"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                return m.rorc[r] == m.arent[r] / (m.pi[r] + 1e-12) - m.fdepr[r]

        equations.append(EqRorc())

        # ---------------- eq_rore (monolith 7292) ----------------
        class EqRore(SymbolicEquation):
            name: str = "eq_rore"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                return (
                    m.rore[r]
                    == m.rorc[r] * (m.kstock[r] / (m.kapEnd[r] + 1e-12)) ** m.rorflex[r]
                )

        equations.append(EqRore())

        # ---------------- eq_rorg (monolith 7302) ----------------
        class EqRorg(SymbolicEquation):
            name: str = "eq_rorg"
            domains: tuple = ()

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                numer = 0.0
                denom = 0.0
                for r in m.r:
                    net_invest = m.pi[r] * (m.xiagg[r] - m.depr[r] * m.kstock[r])
                    numer += m.rore[r] * net_invest
                    denom += net_invest
                return m.rorg == numer / (denom + 1e-12)

        # capFlex (GAMS rorgeq $(RoRFlag ne capFlex)): rorgeq is INACTIVE — rorg is closed
        # by capAcct (sum savf == 0) through the equalized returns. Don't register eq_rorg.
        if savf_flag != "capFlex":
            equations.append(EqRorg())

        _ = sqrt  # (imported for parity with the monolith's utility helpers)
        return equations

    # ------------------------------------------------------------------
    # init helpers (levels only — quantity/floor inits; see block docstring)
    # ------------------------------------------------------------------
    def _xc_init(self, regions, comms):
        bm = self.params.benchmark
        return np.array(
            [
                [max(bm.get_private_demand(r, i)[0] or 0.0, 0.0) for i in comms]
                for r in regions
            ]
        )

    def _savf_init(self, regions, calib):
        """Foreign savings at benchmark = savf_bar (get_savf_init, capFix closure —
        the comp-stat default). savf_bar is the calibrated capital-account residual
        (Σ vcif − vfob − vst, with the residual-region rebalance) from the shared
        calibration loop."""
        sb = calib["savf_bar"]
        return np.array([sb.get((r,), 0.0) for r in regions])

    def _xcshr_init(self, regions, comms):
        """Household budget share xcshr[r,i] = private_demand(r,i) / Σ_j private
        _demand(r,j), from the SAM benchmark (mirrors the monolith's
        get_xcshr_init — a calibration seed the GAMS GDX does not fill because its
        xcshr symbol is 4-key (r,i,hhd,t) vs this 3-key (r,i,t) var, so the init
        must be correct here rather than left at a placeholder)."""
        bm = self.params.benchmark
        out = np.empty((len(regions), len(comms)))
        for ri, r in enumerate(regions):
            total = sum(bm.get_private_demand(r, j)[0] for j in comms)
            for ci, i in enumerate(comms):
                out[ri, ci] = (
                    (bm.get_private_demand(r, i)[0] / total) if total > 0.0 else 0.0
                )
        return out

    def _vgm_init(self, regions, comms):
        bm = self.params.benchmark
        return np.array(
            [
                [max(bm.get_government_demand(r, i)[0] or 0.0, 0.0) for i in comms]
                for r in regions
            ]
        )

    def _xi_init(self, regions, comms):
        bm = self.params.benchmark
        return np.array(
            [
                [max(bm.get_investment_demand(r, i)[0] or 0.0, 0.0) for i in comms]
                for r in regions
            ]
        )

    def _xiagg_init(self, regions):
        bm = self.params.benchmark
        return np.array(
            [
                max(sum(bm.vim.get((r, i), 0.0) for i in self.sets.i), 0.0)
                for r in regions
            ]
        )

    def _pi_init(self, regions):
        return np.ones(len(regions))

    def _zcons_init(self, regions, comms, calib):
        z = calib["zcons_init"]
        return np.array([[z.get((r, i), 0.0) for i in comms] for r in regions])

    def _phip_init(self, regions, calib):
        p = calib["phip0"]
        return np.array([p.get((r,), 1.0) for r in regions])

    def _phi_init(self, regions, calib):
        p = calib["phi0"]
        return np.array([p.get((r,), 1.0) for r in regions])

    def _chif_init(self, regions, calib):
        c = calib["chif0"]
        return np.array([c.get((r,), 0.0) for r in regions])

    def _ev_init(self, regions, calib):
        # ev/cv init = max(yc_bench, 1e-8) (monolith 4940-4951 via yc).
        # yc_bench = betap * (phi/phip) * regy_bench (monolith 2503).
        out = []
        for r in regions:
            betap = calib["betap"].get((r,), 0.0)
            phi = calib["phi0"].get((r,), 1.0)
            phip = calib["phip0"].get((r,), 1.0)
            regy = calib["regy_bench"].get((r,), 0.0)
            yc = betap * (phi / max(phip, 1e-12)) * regy
            out.append(max(yc, 1e-8))
        return np.array(out)

    def _xigbl_init(self, regions):
        bm = self.params.benchmark
        total = 0.0
        for r in regions:
            inv = sum(bm.vim.get((r, i), 0.0) for i in self.sets.i)
            vkb = float(bm.vkb.get(r, 0.0))
            vdep = float(bm.vdep.get(r, 0.0))
            depr = (vdep / vkb) if vkb > 0.0 else 0.0
            total += max(inv - depr * vkb, 0.0)
        return max(total, 1e-8)

    def _pigbl_init(self, regions):
        return 1.0

    def _rorg_init(self, regions):
        # A positive benchmark scalar; the relative floor uses this. The monolith
        # seeds rorg=1.0 (4966) then the floor uses the SCALED value — a composer
        # snapshot carry (see __init__ checklist item 7); block seeds 1.0.
        return 1.0
