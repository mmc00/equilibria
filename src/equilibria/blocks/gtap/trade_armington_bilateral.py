"""GTAP ARMINGTON + BILATERAL trade block (units merged).

Ports the monolith's top-Armington nest and bilateral import-source/export
allocation equations VERBATIM from ``gtap_model_equations.py`` (6014-6631):

  ARMINGTON: eq_xaa_activity (6014), eq_xaa_hhd (6042), eq_xaa_gov (6047),
             eq_xaa_inv (6052), eq_xaa_tmg (6067), eq_dintxeq (6241),
             eq_mintxeq (6283), eq_xda (6325), eq_xma (6341), eq_xd_agg (6358),
             eq_xmt_agg (6365), eq_paa (6374), eq_xwmg (6402), eq_xmgm (6409),
             eq_pwmg (6421), eq_xtmg (6434), eq_ptmg (6447)
  BILATERAL: eq_xweq (6489), eq_pmteq (6506), eq_pmeq (6531), eq_pmcifeq (6546),
             eq_pefobeq (6559), eq_peeq (6572), eq_peteq (6589), eq_pdeq (6624)

The 3-D/4-D unit: xw/pm/pmcif/pefob/pe (3-D), xmgm (4-D). ``rp`` is the region
alias set (same elements as ``r``). Note the index ORDER: xw/pm/pmcif are
declared/accessed exporter-first; pefob/pe are exporter-first-as-r; the
constraints iterate the ordering the monolith uses.

EXPRESSION ALIASES (NOT Vars — monolith 3695-3729): paa[r,i,aa]=pa[r,i,aa];
pdp[r,i,aa]=(1+dintx[r,i,aa])*pd[r,i]; pmp[r,i,aa]=(1+mintx[r,i,aa])*pmt[r,i].
These are Pyomo Expression in the monolith; the block INLINES them wherever an
eq body references them (the monolith's constraint .expr expands them inline),
so the ported form string-matches the oracle.

ifSUB: ``_m_xwmg``/``_m_xmgm``/``_m_pm`` return the PLAIN VAR under if_sub=False
(the comp-stat oracle). This block ports the if_sub=False form; the composer
deactivates eq_xwmg/xmgm/pwmg/pefobeq/pmcifeq/pmeq and fixes their paired vars
under if_sub=True (see blocks/gtap/__init__.py).

CARRY (unit-4 pre-adjudicated, Blocker-C family): the top-Armington shares
alphad/alpham (inlined into eq_xda/eq_xma/eq_paa) are computed by the monolith
from POST-apply_production_scaling Var levels; the block reads the benchmark
SEED (dp.armington_shares), so those coefficients DRIFT on active CES cells. The
composer re-runs the recompute with the scaled model. On gtap7_3x3 gw_share is
INERT (all omegaw=inf -> eq_peeq/peteq take the Leontief branch that never
references gw_share). Structure + Skip mask are identical.

FIDELITY: equation bodies are the monolith's, transcribed. Params via
_derived_params verbatim transcriptions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyomo.environ import value

from equilibria.blocks.base import Block
from equilibria.blocks.gtap import _derived_params as dp
from equilibria.core.parameters import Parameter
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable

_FLOOR = 1e-8
_REL = 1e-3
_HHD, _GOV, _INV, _TMG = "hhd", "gov", "inv", "tmg"


def _price_floor(init: float) -> float:
    if init is None or init <= 0.0:
        return _FLOOR
    return max(_FLOOR, _REL * float(init))


class ArmingtonBilateralBlock(Block):
    """GTAP top-Armington nest + bilateral import/export allocation."""

    name: str = "GTAP_ARMINGTON_BILATERAL"
    description: str = "GTAP top Armington CES nest + bilateral trade (CES/CET)"
    sets: Any = None
    params: Any = None

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["r", "i", "aa", "rp", "m"]

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        regions = list(set_manager.get("r"))
        comms = list(set_manager.get("i"))
        acts = list(set_manager.get("a"))
        aa_list = list(set_manager.get("aa"))
        rp_list = list(set_manager.get("rp"))
        margins = list(set_manager.get("m"))
        p = self.params
        s = self.sets
        _byname = {
            "r": regions,
            "i": comms,
            "a": acts,
            "aa": aa_list,
            "rp": rp_list,
            "m": margins,
        }

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        def _param(name, data, doms, default=0.0, mutable=False):
            parameters[name] = Parameter(
                name=name,
                value=dp.to_array(data, [_byname[d] for d in doms], default),
                domains=tuple(doms),
                mutable=mutable,
            )

        _param("tmarg", dp.tmarg_data(p, s), ("r", "i", "rp"))
        _param("amgm", dp.amgm_data(p, s), ("m", "r", "i", "rp"))
        _param("lambdamg", dp.lambdamg_data(p, s), ("m", "r", "i", "rp"), default=1.0)
        _param("xw_flag", dp.xw_flag_data(p, s), ("r", "i", "rp"), mutable=True)
        _param("xet_flag", dp.xet_flag_data(p, s), ("r", "i"), mutable=True)
        _param("gw_share", dp.gw_share_data(p, s), ("r", "i", "rp"), mutable=True)
        # imptx: mutable in the monolith (5017), referenced UNWRAPPED in eq_pmeq
        # -> stays symbolic. Indexed (r,i,rp) = (exporter,commodity,importer).
        _param("imptx", dp.imptx_data(p, s), ("r", "i", "rp"), mutable=True)
        # shared with PRODUCTION_SUPPLY / FACTOR (dedup by name); declared so a
        # standalone armington build resolves io_param/p_io/lambdaio/xscale.
        _param("io_param", dp.io_param_data(p, s), ("r", "i", "a"))
        _param("p_io", dp.p_io_data(p, s), ("r", "i", "a"))
        _param("lambdaio", dp.lambdaio_data(p, s), ("r", "i", "a"), default=1.0)
        _param("xscale", dp.xscale_data(p, s), ("r", "aa"), default=1.0)

        # ------------------------------------------------------------------
        # Variables OWNED by this unit (monolith 3585-3922, 4323-4491).
        # ------------------------------------------------------------------
        def _q(name, doms, init):
            variables[name] = Variable(
                name=name,
                value=init,
                domains=tuple(doms),
                domain="NonNegativeReals",
                lower=0.0,
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

        nr, ni, naa = len(regions), len(comms), len(aa_list)
        nrp, nm = len(rp_list), len(margins)
        bm = p.benchmark

        # pa (r,i,aa) NonNeg with the runtime price floor. The monolith .fix(1.0)es
        # the no-demand cells (3595-3602) which sets value+fixed flag but NOT the
        # bounds — bounds stay (0.001,None) from the floor sweep (verified vs the
        # oracle: fixed pa[..,tmg] bounds = (0.001,None), not (1,1)). The composer
        # applies the .fix (a value/flag mode-switch), not a bound change.
        pa_init = np.ones((nr, ni, naa))
        _price("pa", ("r", "i", "aa"), pa_init)
        # dintx/mintx (r,i,aa) within=Reals bounds=(-0.999,None) (3603-3620).
        dintx_init = np.array(
            [
                [[dp._dintx_target(p, s, r, i, aa) for aa in aa_list] for i in comms]
                for r in regions
            ]
        )
        mintx_init = np.array(
            [
                [[dp._mintx_target(p, s, r, i, aa) for aa in aa_list] for i in comms]
                for r in regions
            ]
        )
        variables["dintx"] = Variable(
            name="dintx",
            value=dintx_init,
            domains=("r", "i", "aa"),
            domain="Reals",
            lower=-0.999,
            upper=float("inf"),
        )
        variables["mintx"] = Variable(
            name="mintx",
            value=mintx_init,
            domains=("r", "i", "aa"),
            domain="Reals",
            lower=-0.999,
            upper=float("inf"),
        )
        # xd/xmt/pmt (r,i)
        _q("xd", ("r", "i"), np.ones((nr, ni)))
        _q("xmt", ("r", "i"), np.ones((nr, ni)))
        _price("pmt", ("r", "i"), np.ones((nr, ni)))
        # xda/xma (r,i,aa)
        trade = dp._agent_trade_cache(p, s)
        xda_init = np.array(
            [
                [
                    [max(trade.get((r, i, aa), (0.0, 0.0))[0], 0.0) for aa in aa_list]
                    for i in comms
                ]
                for r in regions
            ]
        )
        xma_init = np.array(
            [
                [
                    [max(trade.get((r, i, aa), (0.0, 0.0))[1], 0.0) for aa in aa_list]
                    for i in comms
                ]
                for r in regions
            ]
        )
        _q("xda", ("r", "i", "aa"), xda_init)
        _q("xma", ("r", "i", "aa"), xma_init)
        # xaa (r,i,aa)
        xaa_init = np.array(
            [
                [
                    [
                        max(dp._xaa_purchaser_value(p, s, r, i, aa), 0.0)
                        for aa in aa_list
                    ]
                    for i in comms
                ]
                for r in regions
            ]
        )
        _q("xaa", ("r", "i", "aa"), xaa_init)
        # xe/xw/pe (r,i,rp). xw is FREE (within=Reals, NO lb) — fabricated-corner
        # lesson (3847-3854). xe NonNeg. pe price floor.
        _q("xe", ("r", "i", "rp"), np.zeros((nr, ni, nrp)))
        xw_init = np.array(
            [
                [[max(self._xw_init(r, i, rp), 0.0) for rp in rp_list] for i in comms]
                for r in regions
            ]
        )
        variables["xw"] = Variable(
            name="xw",
            value=xw_init,
            domains=("r", "i", "rp"),
            domain="Reals",
            lower=float("-inf"),
            upper=float("inf"),
        )
        _price("pe", ("r", "i", "rp"), np.ones((nr, ni, nrp)))
        # pwmg/xwmg (r,i,rp): pwmg price floor, xwmg quantity.
        _price("pwmg", ("r", "i", "rp"), np.ones((nr, ni, nrp)))
        xwmg_init = np.array(
            [
                [[max(self._xwmg_init(r, i, rp), 0.0) for rp in rp_list] for i in comms]
                for r in regions
            ]
        )
        _q("xwmg", ("r", "i", "rp"), xwmg_init)
        # xmgm 4-D (m,r,i,rp)
        xmgm_init = np.array(
            [
                [
                    [
                        [
                            max(float(bm.vtwr.get((r, i, rp, mm), 0.0) or 0.0), 0.0)
                            for rp in rp_list
                        ]
                        for i in comms
                    ]
                    for r in regions
                ]
                for mm in margins
            ]
        )
        _q("xmgm", ("m", "r", "i", "rp"), xmgm_init)
        # xtmg/ptmg (m): ptmg price (init 1.0 -> floor 1e-3); xtmg quantity.
        xtmg_init = np.array([max(self._xtmg_init(mm), 0.0) for mm in margins])
        _q("xtmg", ("m",), xtmg_init)
        _price("ptmg", ("m",), np.ones(nm))
        # pm/pmcif (rp,i,r) exporter-first; pefob (r,i,rp) — all price floors.
        _price("pm", ("rp", "i", "r"), np.ones((nrp, ni, nr)))
        _price("pmcif", ("rp", "i", "r"), np.ones((nrp, ni, nr)))
        _price("pefob", ("r", "i", "rp"), np.ones((nr, ni, nrp)))

        # ------------------------------------------------------------------
        # Inline-python accessors + caches (mirror the monolith rule reads).
        # ------------------------------------------------------------------
        el = p.elasticities
        taxes = p.taxes
        shifts_lambdaio = p.shifts.lambdaio
        margin_commodities = {str(mm) for mm in s.m}

        # alphaa_tmg (monolith 6060-6065)
        alphaa_tmg: dict[tuple[str, str], float] = {}
        for i_m in margin_commodities:
            denom = sum(dp._vst_value(p, str(rp), i_m) for rp in s.r)
            if denom > 1e-12:
                for r in s.r:
                    alphaa_tmg[(str(r), i_m)] = dp._vst_value(p, str(r), i_m) / denom

        # top-Armington shares (benchmark-seed CARRY, monolith 6122-6229)
        arm_shares = dp.armington_shares(p, s)

        def _shares(r, i, aa):
            return arm_shares.get((r, i, aa), (0.0, 0.0))

        def _top_sigma(r, i, aa):
            return float(el.esubd.get((r, i), 2.0))

        # import-source share cache (monolith 6472-6485)
        import_src = p.shares.normalized.import_source_share

        def _get_import_source_share(importer, commodity, exporter):
            return float(import_src.get((importer, commodity, exporter), 0.0) or 0.0)

        def _get_sigmand(r, a):
            return el.sigmand.get((r, a), 1.0)

        def _lambdam_value(exporter, commodity, importer):
            lm = el.esubm  # placeholder; lambdam read below
            return max(
                float(_safe(p, "lambdam", (exporter, commodity, importer), 1.0)), 1e-12
            )

        def _chipm_value(exporter, commodity, importer):
            return max(
                float(_safe(p, "chipm", (exporter, commodity, importer), 1.0)), 1e-12
            )

        def _mtax_value(importer, commodity, exporter):
            return float(_safe(p, "mtax", (importer, commodity), 0.0))

        def _etax_value(exporter, commodity, importer):
            return float(_safe(p, "etax", (exporter, commodity), 0.0))

        equations: list[SymbolicEquation] = []

        # ---------------- eq_xaa_activity (monolith 6014) ----------------
        class EqXaaActivity(SymbolicEquation):
            name: str = "eq_xaa_activity"
            domains: tuple = ("r", "i", "a")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, i, a = indices
                io_val = (
                    value(model.io_param[r, i, a])
                    if hasattr(model, "io_param")
                    else value(model.p_io[r, i, a])
                )
                if not shifts_lambdaio:
                    io_val = value(model.p_io[r, i, a])
                if io_val <= 0.0:
                    return model.xaa[r, i, a] == 0.0
                sigmand = _get_sigmand(r, a)
                lambdaio = max(value(model.lambdaio[r, i, a]), 1e-8)
                if abs(sigmand) < 1e-12:
                    return model.xaa[r, i, a] == io_val * model.nd[r, a] / lambdaio
                return model.xaa[r, i, a] == (
                    io_val
                    * model.nd[r, a]
                    * (model.pnd[r, a] / model.pa[r, i, a]) ** sigmand
                    * (lambdaio ** (sigmand - 1.0))
                )

        equations.append(EqXaaActivity())

        # ---------------- eq_xaa_hhd/gov/inv (monolith 6042/6047/6052) ----------------
        class EqXaaHhd(SymbolicEquation):
            name: str = "eq_xaa_hhd"
            domains: tuple = ("r", "i")

            def build_expression(self, pyomo_model, indices):
                r, i = indices
                return pyomo_model.xaa[r, i, _HHD] == pyomo_model.xc[r, i]

        equations.append(EqXaaHhd())

        class EqXaaGov(SymbolicEquation):
            name: str = "eq_xaa_gov"
            domains: tuple = ("r", "i")

            def build_expression(self, pyomo_model, indices):
                r, i = indices
                return pyomo_model.xaa[r, i, _GOV] == pyomo_model.xg[r, i]

        equations.append(EqXaaGov())

        class EqXaaInv(SymbolicEquation):
            name: str = "eq_xaa_inv"
            domains: tuple = ("r", "i")

            def build_expression(self, pyomo_model, indices):
                r, i = indices
                return pyomo_model.xaa[r, i, _INV] == pyomo_model.xi[r, i]

        equations.append(EqXaaInv())

        # ---------------- eq_xaa_tmg (monolith 6067) ----------------
        class EqXaaTmg(SymbolicEquation):
            name: str = "eq_xaa_tmg"
            domains: tuple = ("r", "i")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, i = indices
                i_str = str(i)
                if i_str not in margin_commodities:
                    return model.xaa[r, i, _TMG] == 0.0
                alpha = alphaa_tmg.get((str(r), i_str), 0.0)
                if alpha <= 0.0:
                    return model.xaa[r, i, _TMG] == 0.0
                sigmamg = float(el.sigmam.get(i_str, 1.0))
                if abs(sigmamg - 1.0) < 1e-8:
                    sigmamg = 1.01
                return model.xaa[r, i, _TMG] == (
                    alpha
                    * model.xtmg[i]
                    * (model.ptmg[i] / (model.pa[r, i, _TMG] + 1e-12)) ** sigmamg
                )

        equations.append(EqXaaTmg())

        # ---------------- eq_dintxeq / eq_mintxeq (monolith 6241/6283) ----------------
        class EqDintxeq(SymbolicEquation):
            name: str = "eq_dintxeq"
            domains: tuple = ("r", "i", "aa")

            def build_expression(self, pyomo_model, indices):
                r, i, aa = indices
                target = dp._dintx_target(p, s, r, i, aa)
                return pyomo_model.dintx[r, i, aa] == target

        equations.append(EqDintxeq())

        class EqMintxeq(SymbolicEquation):
            name: str = "eq_mintxeq"
            domains: tuple = ("r", "i", "aa")

            def build_expression(self, pyomo_model, indices):
                r, i, aa = indices
                target = dp._mintx_target(p, s, r, i, aa)
                return pyomo_model.mintx[r, i, aa] == target

        equations.append(EqMintxeq())

        # ---------------- eq_xda / eq_xma (monolith 6325/6341) ----------------
        # pdp/pmp/paa are Expression aliases -> INLINE (pdp=(1+dintx)*pd,
        # pmp=(1+mintx)*pmt, paa=pa) exactly as the oracle expands them.
        class EqXda(SymbolicEquation):
            name: str = "eq_xda"
            domains: tuple = ("r", "i", "aa")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, i, aa = indices
                domestic_share, _ = _shares(r, i, aa)
                if domestic_share <= 0.0:
                    return model.xda[r, i, aa] == 0.0
                sigma_m = _top_sigma(r, i, aa)
                if sigma_m == float("inf"):
                    return (1.0 + model.dintx[r, i, aa]) * model.pd[r, i] == model.pa[
                        r, i, aa
                    ]
                return model.xda[r, i, aa] == (
                    domestic_share
                    * model.xaa[r, i, aa]
                    * (
                        model.pa[r, i, aa]
                        / ((1.0 + model.dintx[r, i, aa]) * model.pd[r, i])
                    )
                    ** sigma_m
                )

        equations.append(EqXda())

        class EqXma(SymbolicEquation):
            name: str = "eq_xma"
            domains: tuple = ("r", "i", "aa")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, i, aa = indices
                _, import_share = _shares(r, i, aa)
                if import_share <= 0.0:
                    return model.xma[r, i, aa] == 0.0
                sigma_m = _top_sigma(r, i, aa)
                if sigma_m == float("inf"):
                    return (1.0 + model.mintx[r, i, aa]) * model.pmt[r, i] == model.pa[
                        r, i, aa
                    ]
                return model.xma[r, i, aa] == (
                    import_share
                    * model.xaa[r, i, aa]
                    * (
                        model.pa[r, i, aa]
                        / ((1.0 + model.mintx[r, i, aa]) * model.pmt[r, i])
                    )
                    ** sigma_m
                )

        equations.append(EqXma())

        # ---------------- eq_xd_agg / eq_xmt_agg (monolith 6358/6365) ----------------
        class EqXdAgg(SymbolicEquation):
            name: str = "eq_xd_agg"
            domains: tuple = ("r", "i")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, i = indices
                return model.xd[r, i] == sum(
                    model.xda[r, i, aa] / model.xscale[r, aa] for aa in model.aa
                )

        equations.append(EqXdAgg())

        class EqXmtAgg(SymbolicEquation):
            name: str = "eq_xmt_agg"
            domains: tuple = ("r", "i")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, i = indices
                return model.xmt[r, i] == sum(
                    model.xma[r, i, aa] / model.xscale[r, aa] for aa in model.aa
                )

        equations.append(EqXmtAgg())

        # ---------------- eq_paa (monolith 6374) ----------------
        class EqPaa(SymbolicEquation):
            name: str = "eq_paa"
            domains: tuple = ("r", "i", "aa")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, i, aa = indices
                alphad, alpham = _shares(r, i, aa)
                if alphad <= 0.0 and alpham <= 0.0:
                    return None
                sigma_m = _top_sigma(r, i, aa)
                expo = 1.0 - sigma_m
                if abs(expo) < 1e-8:
                    return model.pa[r, i, aa] == (
                        ((1.0 + model.dintx[r, i, aa]) * model.pd[r, i]) ** alphad
                        * ((1.0 + model.mintx[r, i, aa]) * model.pmt[r, i]) ** alpham
                    )
                return model.pa[r, i, aa] ** expo == (
                    alphad * ((1.0 + model.dintx[r, i, aa]) * model.pd[r, i]) ** expo
                    + alpham * ((1.0 + model.mintx[r, i, aa]) * model.pmt[r, i]) ** expo
                )

        equations.append(EqPaa())

        # ---------------- eq_xwmg / eq_xmgm / eq_pwmg (monolith 6402/6409/6421) ----------------
        class EqXwmg(SymbolicEquation):
            name: str = "eq_xwmg"
            domains: tuple = ("r", "i", "rp")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, i, rp = indices
                if value(model.tmarg[r, i, rp]) <= 0.0:
                    return None
                return (
                    model.xwmg[r, i, rp] == model.tmarg[r, i, rp] * model.xw[r, i, rp]
                )

        equations.append(EqXwmg())

        class EqXmgm(SymbolicEquation):
            name: str = "eq_xmgm"
            domains: tuple = ("m", "r", "i", "rp")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                m, r, i, rp = indices
                share = value(model.amgm[m, r, i, rp])
                if share <= 0.0:
                    return None
                # if_sub=False: _m_xwmg(r,i,rp) -> model.xwmg[r,i,rp]
                return model.xmgm[m, r, i, rp] == share * model.xwmg[r, i, rp] / (
                    model.lambdamg[m, r, i, rp] + 1e-12
                )

        equations.append(EqXmgm())

        class EqPwmg(SymbolicEquation):
            name: str = "eq_pwmg"
            domains: tuple = ("r", "i", "rp")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, i, rp = indices
                if value(model.tmarg[r, i, rp]) <= 0.0:
                    return None
                total = sum(
                    model.amgm[m, r, i, rp]
                    * model.ptmg[m]
                    / (model.lambdamg[m, r, i, rp] + 1e-12)
                    for m in model.m
                )
                return model.pwmg[r, i, rp] == total

        equations.append(EqPwmg())

        # ---------------- eq_xtmg (monolith 6434) ----------------
        class EqXtmg(SymbolicEquation):
            name: str = "eq_xtmg"
            domains: tuple = ("m",)

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                (m,) = indices
                # if_sub=False: _m_xmgm(m,r,i,rp) -> model.xmgm[m,r,i,rp]
                return model.xtmg[m] == sum(
                    model.xmgm[m, r, i, rp]
                    for r in model.r
                    for i in model.i
                    for rp in model.rp
                )

        equations.append(EqXtmg())

        # ---------------- eq_ptmg (monolith 6447) ----------------
        class EqPtmg(SymbolicEquation):
            name: str = "eq_ptmg"
            domains: tuple = ("m",)

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                (m,) = indices
                i_str = str(m)
                has_supply = any(
                    alphaa_tmg.get((str(r), i_str), 0.0) > 0.0 for r in s.r
                )
                if not has_supply:
                    return model.ptmg[m] == model.pnum
                sigmamg = float(el.sigmam.get(i_str, 1.0))
                if abs(sigmamg - 1.0) < 1e-8:
                    sigmamg = 1.01
                expo = 1.0 - sigmamg
                terms = sum(
                    alphaa_tmg.get((str(r), i_str), 0.0) * model.pa[r, m, _TMG] ** expo
                    for r in model.r
                    if alphaa_tmg.get((str(r), i_str), 0.0) > 0.0
                )
                return model.ptmg[m] ** expo == terms

        equations.append(EqPtmg())

        # ---------------- eq_xweq (monolith 6489) — Constraint(rp,i,r) ----------------
        class EqXweq(SymbolicEquation):
            name: str = "eq_xweq"
            domains: tuple = ("rp", "i", "r")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                rp, i, r = indices
                amw = _get_import_source_share(r, i, rp)
                if amw <= 0.0:
                    return None
                esubm = el.esubm.get((r, i), 5.0)
                lambdam = _lambdam_value(rp, i, r)
                # if_sub=False: _m_pm(rp,i,r) -> model.pm[rp,i,r]
                return model.xw[rp, i, r] == (
                    amw
                    * model.xmt[r, i]
                    * (model.pmt[r, i] / model.pm[rp, i, r]) ** esubm
                    * (lambdam ** (esubm - 1.0))
                )

        equations.append(EqXweq())

        # ---------------- eq_pmteq (monolith 6506) ----------------
        class EqPmteq(SymbolicEquation):
            name: str = "eq_pmteq"
            domains: tuple = ("r", "i")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, i = indices
                esubm = el.esubm.get((r, i), 5.0)
                expo = 1.0 - esubm
                if abs(expo) < 1e-8:
                    return None
                active_shares = [_get_import_source_share(r, i, rp) for rp in model.rp]
                if not any(share > 0.0 for share in active_shares):
                    return model.pmt[r, i] == 1.0
                terms = []
                for rp in model.rp:
                    amw = _get_import_source_share(r, i, rp)
                    if amw <= 0.0:
                        continue
                    lambdam = _lambdam_value(rp, i, r)
                    # if_sub=False: _m_pm(rp,i,r) -> model.pm[rp,i,r]
                    terms.append(amw * (model.pm[rp, i, r] / lambdam) ** expo)
                if not terms:
                    return model.pmt[r, i] == 1.0
                return model.pmt[r, i] ** expo == sum(terms)

        equations.append(EqPmteq())

        # ---------------- eq_pmeq (monolith 6531) — Constraint(rp,i,r) ----------------
        class EqPmeq(SymbolicEquation):
            name: str = "eq_pmeq"
            domains: tuple = ("rp", "i", "r")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                rp, i, r = indices
                if value(model.xw_flag[rp, i, r]) <= 0.0:
                    return None
                # _imptx_rate_importer(r,i,rp) -> model.imptx[rp,i,r] (symbolic,
                # indexed exporter,commodity,importer). mtax/chipm inlined floats.
                mtax = _mtax_value(r, i, rp)
                chipm = _chipm_value(rp, i, r)
                return (
                    model.pm[rp, i, r]
                    == ((1.0 + model.imptx[rp, i, r] + mtax) * model.pmcif[rp, i, r])
                    / chipm
                )

        equations.append(EqPmeq())

        # ---------------- eq_pmcifeq (monolith 6546) — Constraint(rp,i,r) ----------------
        class EqPmcifeq(SymbolicEquation):
            name: str = "eq_pmcifeq"
            domains: tuple = ("rp", "i", "r")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                rp, i, r = indices
                if value(model.xw_flag[rp, i, r]) <= 0.0:
                    return None
                tmarg = value(model.tmarg[rp, i, r])
                return model.pmcif[rp, i, r] == (
                    model.pefob[rp, i, r] + model.pwmg[rp, i, r] * tmarg
                )

        equations.append(EqPmcifeq())

        # ---------------- eq_pefobeq (monolith 6559) — Constraint(r,i,rp) ----------------
        class EqPefobeq(SymbolicEquation):
            name: str = "eq_pefobeq"
            domains: tuple = ("r", "i", "rp")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, i, rp = indices
                if value(model.xw_flag[r, i, rp]) <= 0.0:
                    return None
                export_tax = float(taxes.rtxs.get((r, i, rp), 0.0))
                etax = _etax_value(r, i, rp)
                return (
                    model.pefob[r, i, rp]
                    == (1.0 + export_tax + etax) * model.pe[r, i, rp]
                )

        equations.append(EqPefobeq())

        # ---------------- eq_peeq (monolith 6572) — Constraint(r,i,rp) ----------------
        class EqPeeq(SymbolicEquation):
            name: str = "eq_peeq"
            domains: tuple = ("r", "i", "rp")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, i, rp = indices
                if value(model.xw_flag[r, i, rp]) <= 0.0:
                    return None
                omegaw = el.omegaw.get((r, i), float("inf"))
                if omegaw == float("inf"):
                    return model.pe[r, i, rp] == model.pet[r, i]
                return model.xw[r, i, rp] == (
                    model.gw_share[r, i, rp]
                    * model.xet[r, i]
                    * (model.pe[r, i, rp] / model.pet[r, i]) ** omegaw
                )

        equations.append(EqPeeq())

        # ---------------- eq_peteq (monolith 6589) ----------------
        class EqPeteq(SymbolicEquation):
            name: str = "eq_peteq"
            domains: tuple = ("r", "i")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, i = indices
                if value(model.xet_flag[r, i]) <= 0.0:
                    return None
                active_routes: list[str] = []
                for rp in model.rp:
                    if value(model.xw_flag[r, i, rp]) > 0.0:
                        active_routes.append(rp)
                if not active_routes:
                    return None
                omegaw = el.omegaw.get((r, i), float("inf"))
                if omegaw == float("inf"):
                    return model.xet[r, i] == sum(
                        model.xw[r, i, rp] for rp in active_routes
                    )
                exponent = 1.0 + omegaw
                terms = []
                for rp in active_routes:
                    gw = (
                        model.gw_share[r, i, rp]
                        if hasattr(model, "gw_share")
                        else float(p.shares.p_gw.get((r, i, rp), 0.0))
                    )
                    terms.append(gw * model.pe[r, i, rp] ** exponent)
                if not terms:
                    return None
                return model.pet[r, i] ** exponent == sum(terms)

        equations.append(EqPeteq())

        # ---------------- eq_pdeq (monolith 6624) ----------------
        class EqPdeq(SymbolicEquation):
            name: str = "eq_pdeq"
            domains: tuple = ("r", "i")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, i = indices
                return model.xds[r, i] == sum(
                    model.xda[r, i, aa] / model.xscale[r, aa]
                    for aa in model.aa
                    if _shares(r, i, aa)[0] > 0.0
                )

        equations.append(EqPdeq())

        return equations

    # ------------------------------------------------------------------
    # init helpers (price-floor-bearing + owned levels)
    # ------------------------------------------------------------------
    def _xw_init(self, r, i, rp):
        """get_xw_init benchmark: vxsb/pe (pe=1) — monolith get_xw_init."""
        return float(self.params.benchmark.vxsb.get((r, i, rp), 0.0) or 0.0)

    def _xwmg_init(self, r, i, rp):
        """get_xwmg_init: sum_m vtwr — monolith 3871-3878."""
        return sum(
            float(self.params.benchmark.vtwr.get((r, i, rp, m), 0.0) or 0.0)
            for m in self.sets.m
        )

    def _xtmg_init(self, margin):
        """get_xtmg_init: sum_{r,i,rp} vtwr — monolith 3883-3889."""
        total = 0.0
        for r in self.sets.r:
            for i in self.sets.i:
                for rp in self.sets.r:
                    total += float(
                        self.params.benchmark.vtwr.get((r, i, rp, margin), 0.0) or 0.0
                    )
        return total


def _safe(params: Any, comp_name: str, key, default: float) -> float:
    """Fixed-shifter lookup (lambdam/chipm/mtax/etax) from params.shares/taxes.

    The monolith reads these off model Params (fixed at benchmark); on 3x3
    chipm=1, lambdam=1, mtax=etax=0 (the calibration leaves them at benchmark).
    Read from the analogous params container, defaulting as the monolith's
    _safe_component_value does (1.0 for lambdam/chipm, 0.0 for mtax/etax).
    """
    containers = (
        getattr(params, "shares", None),
        getattr(params, "taxes", None),
        getattr(params, "benchmark", None),
    )
    for c in containers:
        if c is None:
            continue
        raw = getattr(c, comp_name, None)
        if raw is not None and hasattr(raw, "get"):
            val = raw.get(key)
            if val is not None:
                return float(val)
    return default
