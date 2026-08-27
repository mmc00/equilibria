"""GTAP6 TRADE_ARMINGTON block (leaf unit).

Ports the v6.2 monolith's Armington + bilateral trade + margin equations
VERBATIM from ``scripts/gtap6/_v62_monolith_oracle.py``
(``GTAP6MonolithOracle``), following the same fidelity discipline
``blocks/gtap/trade_cet.py``/``blocks/gtap/trade_armington_bilateral.py`` used
for GTAP7.

v6.2 differences from GTAP7's ``ArmingtonBilateralBlock``: no MRIO, no
region-indexed esubd/esubm (commodity-only: ``esubd(i)``/``esubm(i)``),
no CET export allocation (v6.2 has no ``xet``/CET — the FOB price ``pe``
is a pure linear tax-wedge identity off the supply price ``ps``), margins
are Cobb-Douglas (v6.2 has no ESUBS elasticity), no ifSUB macro
substitution (v6.2 has no ``ifSUB`` GAMS flag at all).

Oracle -> contract equation-name mapping (the oracle's method names differ
from the contract's ``e_*`` IDs; this is NOT a re-derivation, only a rename):

  e_qfd_arm -> eq_qfd_rule   (oracle ``_add_production_block``, monolith 1680)
  e_qfm_arm -> eq_qfm_rule   (1696)
  e_qfa     -> eq_qf_rule    (1583) — top-nest Armington composite demand
  e_pfa     -> eq_pf_int_rule (1660) — top-nest Armington composite price
  e_qxs     -> eq_qxs_rule   (oracle ``_add_trade_block``, 2158)
  e_pms     -> eq_pms_rule   (2131)
  e_pmcif   -> eq_pmcif_rule (2120)
  e_pe      -> eq_pe_rule    (2095)
  e_pim     -> eq_pim_rule   (2140)
  e_qst     -> eq_qst_rule   (oracle ``_add_margins_block``, 2228)
  e_pst     -> eq_pst_rule   (2189)
  e_qtm     -> eq_qtm_rule   (2211)
  e_ptmg    -> eq_ptmg_rule  (2195)
  e_pwmg    -> eq_pwmg_rule  (2107)

Two contract IDs have NO corresponding oracle equation (the oracle declares
the Var but never wires a Constraint for it — verified: ``grep -n "qds\\|
qtmfsd" _v62_monolith_oracle.py`` finds only the ``model.qds = Var(...)``
declaration at line 1231, no ``eq_qds`` anywhere, and no ``qtmfsd``/``xmgm``
symbol at all):

  e_qds:    the oracle's own calibration module defines ``vds(i,r)`` (the
            benchmark seed for the ``qds`` Var) as EXACTLY
            ``sum_j VDFM(i,j,r) + VDPM(i,r) + VDGM(i,r)``
            (``gtap6_calibration.py`` lines 305-313) — the domestic-use leg
            of ``eq_market``'s uses side (uses minus exports minus margin
            sales). This is not invented: it is the oracle's own documented
            identity for what ``qds`` represents, transcribed directly using
            the block's OWNED ``qfd`` var plus stubbed ``qpd``/``qgd``.
  e_qtmfsd: the oracle's ``eq_qtm_rule`` sums the per-shipment margin
            requirement ``amgm[mg,i,s,d] * pwmg[i,s,d] * qxs[i,s,d]`` inline
            without ever materializing it as its own Var/Constraint. GTAP7's
            analogous per-shipment quantity is ``xmgm`` (see
            ``trade_armington_bilateral.py`` ``EqXmgm``, monolith 6409):
            ``xmgm = amgm * xwmg / lambdamg``. v6.2 has no lambdamg
            shifter and its per-unit transport cost is already denominated
            via ``pwmg`` (not a separate xwmg quantity), so the v6.2
            per-shipment identity is the disaggregated summand of
            ``eq_qtm`` divided by the margin price: ``qtmfsd(mg,i,s,d) =
            amgm(mg,i,s,d) * pwmg(i,s,d) * qxs(i,s,d) / ptmg(mg)``. Summing
            this over (i,s,d) exactly reproduces ``eq_qtm``'s RHS/ptmg[mg],
            so this is the oracle's own summed term made explicit, not new
            algebra.

FIDELITY: every equation that DOES exist in the oracle is transcribed
byte-for-byte (same Skip conditions, same ``_ces_cd_sigma`` perturbation,
same floor checks). The two synthesized equations above are documented
extensions of the oracle's own stated identities, not independent economic
derivations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from equilibria.blocks.base import Block
from equilibria.core.parameters import Parameter
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable

_LB = 1e-6


def _ces_cd_sigma(sigma: float) -> float:
    """Perturb sigma when it equals 1.0 to avoid (1-sigma) = 0 pathologies.

    Verbatim transcription of the oracle's local helper (redefined
    identically inside each ``_add_*_block`` method, e.g. monolith
    ``_add_trade_block`` line 2087).
    """
    if abs(sigma - 1.0) < 1e-8:
        return 1.0 + 1e-3
    return sigma


def _to_dict(mapping: Any) -> dict:
    """Coerce a params.*.get-style mapping to a plain dict (defensive)."""
    if mapping is None:
        return {}
    if isinstance(mapping, dict):
        return mapping
    return dict(mapping)


class TradeArmingtonBlock(Block):
    """GTAP6 top Armington + bilateral trade + Cobb-Douglas margins."""

    name: str = "GTAP6_TRADE_ARMINGTON"
    description: str = (
        "GTAP6 top Armington (firm intermediates), bilateral CES trade, "
        "Cobb-Douglas margins"
    )
    sets: Any = None
    params: Any = None
    derived: Any = None

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["r", "i", "j", "marg"]

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        regions = list(set_manager.get("r"))
        comms = list(set_manager.get("i"))
        prod_secs = list(set_manager.get("j")) if set_manager.has("j") else list(comms)
        margins = list(set_manager.get("marg")) if set_manager.has("marg") else []
        # Bilateral aliases: v6.2 uses (s, d/rp) aliases of r for source and
        # destination. If the caller hasn't registered "s"/"rp" separately,
        # fall back to "r" (same elements — matches the oracle's model.s /
        # model.rp aliasing of model.r).
        srcs = list(set_manager.get("s")) if set_manager.has("s") else list(regions)
        dests = list(set_manager.get("rp")) if set_manager.has("rp") else list(regions)

        p = self.params
        s = self.sets
        d = self.derived
        el = p.elasticities
        bm = p.benchmark

        nr, ni, nj = len(regions), len(comms), len(prod_secs)
        nsrc, ndest, nmarg = len(srcs), len(dests), len(margins)

        # ------------------------------------------------------------------
        # Params (mirror the oracle's Pyomo Param declarations, dict-valued
        # lookups via .get with the oracle's defaults).
        # ------------------------------------------------------------------
        def _p3(name, data, dims, default=0.0):
            arr = np.full(tuple(len(x) for x in dims), default, dtype=float)
            data = _to_dict(data)
            for key, val in data.items():
                try:
                    idx = tuple(dim.index(k) for dim, k in zip(dims, key, strict=True))
                except (ValueError, TypeError):
                    continue
                arr[idx] = float(val or 0.0)
            return arr

        esubd_arr = np.array([float(el.esubd.get(i, 1.0)) for i in comms], dtype=float)
        esubm_arr = np.array([float(el.esubm.get(i, 1.0)) for i in comms], dtype=float)
        parameters["esubd"] = Parameter(
            name="esubd", value=esubd_arr, domains=("i",), mutable=True
        )
        parameters["esubm"] = Parameter(
            name="esubm", value=esubm_arr, domains=("i",), mutable=True
        )

        alpha_dom_arr = _p3("alpha_dom", d.alpha_dom, (comms, prod_secs, regions))
        alpha_imp_arr = _p3("alpha_imp", d.alpha_imp, (comms, prod_secs, regions))
        parameters["alpha_dom"] = Parameter(
            name="alpha_dom", value=alpha_dom_arr, domains=("i", "j", "r")
        )
        parameters["alpha_imp"] = Parameter(
            name="alpha_imp", value=alpha_imp_arr, domains=("i", "j", "r")
        )
        share_int_arr = _p3("share_int", d.share_int, (comms, prod_secs, regions))
        parameters["share_int"] = Parameter(
            name="share_int", value=share_int_arr, domains=("i", "j", "r")
        )

        tfd_arr = _p3("tfd", d.tfd, (comms, prod_secs, regions))
        tfi_arr = _p3("tfi", d.tfi, (comms, prod_secs, regions))
        parameters["tfd"] = Parameter(
            name="tfd", value=tfd_arr, domains=("i", "j", "r"), mutable=True
        )
        parameters["tfi"] = Parameter(
            name="tfi", value=tfi_arr, domains=("i", "j", "r"), mutable=True
        )

        to_arr = _p3("to", d.to, (prod_secs, regions))
        parameters["to"] = Parameter(
            name="to", value=to_arr, domains=("j", "r"), mutable=True
        )

        alpha_xs_arr = _p3("alpha_xs", d.alpha_xs, (comms, srcs, dests))
        parameters["alpha_xs"] = Parameter(
            name="alpha_xs", value=alpha_xs_arr, domains=("i", "s", "rp")
        )

        txs_arr = _p3("txs", d.txs, (comms, srcs, dests))
        tms_arr = _p3("tms", d.tms, (comms, srcs, dests))
        parameters["txs"] = Parameter(
            name="txs", value=txs_arr, domains=("i", "s", "rp"), mutable=True
        )
        parameters["tms"] = Parameter(
            name="tms", value=tms_arr, domains=("i", "s", "rp"), mutable=True
        )

        amgm_arr = np.zeros((nmarg, ni, nsrc, ndest), dtype=float)
        amgm_data = _to_dict(d.amgm)
        for (mg, i, src, dst), val in amgm_data.items():
            try:
                idx = (
                    margins.index(mg),
                    comms.index(i),
                    srcs.index(src),
                    dests.index(dst),
                )
            except ValueError:
                continue
            amgm_arr[idx] = float(val or 0.0)
        parameters["amgm"] = Parameter(
            name="amgm", value=amgm_arr, domains=("marg", "i", "s", "rp")
        )

        share_st_arr = _p3("share_st", d.share_st, (margins, regions))
        parameters["share_st"] = Parameter(
            name="share_st", value=share_st_arr, domains=("marg", "r")
        )

        pwmg0_arr = _p3("pwmg_0", d.pwmg_0, (comms, srcs, dests))
        parameters["pwmg_0"] = Parameter(
            name="pwmg_0", value=pwmg0_arr, domains=("i", "s", "rp")
        )

        # ------------------------------------------------------------------
        # Variables OWNED by this unit.
        # ------------------------------------------------------------------
        def _q(name, doms, init):
            variables[name] = Variable(
                name=name,
                value=np.maximum(init, _LB),
                domains=tuple(doms),
                domain="NonNegativeReals",
                lower=_LB,
                upper=float("inf"),
            )

        def _price(name, doms, init):
            variables[name] = Variable(
                name=name,
                value=np.maximum(init, _LB),
                domains=tuple(doms),
                domain="NonNegativeReals",
                lower=_LB,
                upper=float("inf"),
            )

        # qfd/qfm (i,j,r) — top-Armington firm intermediate demand.
        qfd_init = np.array(
            [
                [
                    [
                        max(float(bm.vdfm.get((i, j, r), 0.0) or 0.0), _LB)
                        for r in regions
                    ]
                    for j in prod_secs
                ]
                for i in comms
            ]
        )
        qfm_init = np.array(
            [
                [
                    [
                        max(float(bm.vifm.get((i, j, r), 0.0) or 0.0), _LB)
                        for r in regions
                    ]
                    for j in prod_secs
                ]
                for i in comms
            ]
        )
        _q("qfd", ("i", "j", "r"), qfd_init)
        _q("qfm", ("i", "j", "r"), qfm_init)

        # qfa (=oracle's qf): Armington composite intermediate demand.
        qfa_init = np.array(
            [
                [
                    [
                        max(
                            float(bm.vdfm.get((i, j, r), 0.0) or 0.0)
                            + float(bm.vifm.get((i, j, r), 0.0) or 0.0),
                            _LB,
                        )
                        for r in regions
                    ]
                    for j in prod_secs
                ]
                for i in comms
            ]
        )
        _q("qfa", ("i", "j", "r"), qfa_init)

        # pfa (=oracle's pf_int): Armington composite price.
        pf_int0 = _to_dict(d.pf_int_0)
        pfa_init = np.array(
            [
                [
                    [
                        max(float(pf_int0.get((i, j, r), 1.0) or 1.0), _LB)
                        for r in regions
                    ]
                    for j in prod_secs
                ]
                for i in comms
            ]
        )
        _price("pfa", ("i", "j", "r"), pfa_init)

        # qxs (i,s,rp) — bilateral exports (basic-price quantity).
        qxs0 = _to_dict(d.qxs_0)
        qxs_init = np.array(
            [
                [
                    [
                        max(
                            float(
                                qxs0.get((i, sr, dt), bm.vxwd.get((i, sr, dt), 0.0))
                                or 0.0
                            ),
                            _LB,
                        )
                        for dt in dests
                    ]
                    for sr in srcs
                ]
                for i in comms
            ]
        )
        _q("qxs", ("i", "s", "rp"), qxs_init)

        # pms/pmcif/pe (i,s,rp) — bilateral price chain (importer/CIF/FOB).
        pms0 = _to_dict(d.pms_0)
        pmcif0 = _to_dict(d.pmcif_0)
        pe0 = _to_dict(d.pe_0)
        pms_init = np.array(
            [
                [
                    [max(float(pms0.get((i, sr, dt), 1.0) or 1.0), _LB) for dt in dests]
                    for sr in srcs
                ]
                for i in comms
            ]
        )
        pmcif_init = np.array(
            [
                [
                    [
                        max(float(pmcif0.get((i, sr, dt), 1.0) or 1.0), _LB)
                        for dt in dests
                    ]
                    for sr in srcs
                ]
                for i in comms
            ]
        )
        pe_init = np.array(
            [
                [
                    [max(float(pe0.get((i, sr, dt), 1.0) or 1.0), _LB) for dt in dests]
                    for sr in srcs
                ]
                for i in comms
            ]
        )
        _price("pms", ("i", "s", "rp"), pms_init)
        _price("pmcif", ("i", "s", "rp"), pmcif_init)
        _price("pe", ("i", "s", "rp"), pe_init)

        # pwmg (i,s,rp) — per-unit transport cost on bilateral shipment.
        pwmg0 = _to_dict(d.pwmg_0)
        pwmg_init = np.array(
            [
                [
                    [
                        max(float(pwmg0.get((i, sr, dt), 0.0) or 0.0), _LB)
                        for dt in dests
                    ]
                    for sr in srcs
                ]
                for i in comms
            ]
        )
        _price("pwmg", ("i", "s", "rp"), pwmg_init)

        # qim/pim (i,r) — composite import quantity + price (CES dual).
        qim0 = _to_dict(d.qim_0)
        pim0 = _to_dict(d.pim_0)
        qim_init = np.array(
            [
                [max(float(qim0.get((i, r), 1.0) or 1.0), _LB) for r in dests]
                for i in comms
            ]
        )
        pim_init = np.array(
            [
                [max(float(pim0.get((i, r), 1.0) or 1.0), _LB) for r in dests]
                for i in comms
            ]
        )
        _q("qim", ("i", "rp"), qim_init)
        _price("pim", ("i", "rp"), pim_init)

        # qds (i,r) — domestic absorption (documented oracle identity; see
        # module docstring). Seeded from the oracle's own vds calibration.
        vds = _to_dict(d.vds)
        qds_init = np.array(
            [
                [max(float(vds.get((i, r), 1.0) or 1.0), _LB) for r in regions]
                for i in comms
            ]
        )
        _q("qds", ("i", "r"), qds_init)

        # qst/pst (marg,r) — margin sale supply/price.
        qst0 = _to_dict(getattr(d, "qst_0", {}))
        qst_init = np.array(
            [
                [
                    max(float(qst0.get((mg, r), bm.vst.get((mg, r), 0.0)) or 0.0), _LB)
                    for r in regions
                ]
                for mg in margins
            ]
        )
        _q("qst", ("marg", "r"), qst_init)
        pst_init = np.ones((nmarg, nr))
        _price("pst", ("marg", "r"), pst_init)

        # qtm/ptmg (marg) — world margin demand + price.
        qtm0 = _to_dict(getattr(d, "qtm_0", {}))
        qtm_init = np.array(
            [max(float(qtm0.get(mg, 1.0) or 1.0), _LB) for mg in margins]
        )
        _q("qtm", ("marg",), qtm_init)
        ptmg_init = np.ones(nmarg)
        _price("ptmg", ("marg",), ptmg_init)

        # qtmfsd (marg,i,s,rp) — per-shipment margin demand (documented
        # oracle-summand identity; see module docstring).
        qtmfsd_init = np.zeros((nmarg, ni, nsrc, ndest))
        for mi, mg in enumerate(margins):
            for ii, i in enumerate(comms):
                for si, sr in enumerate(srcs):
                    for di, dt in enumerate(dests):
                        amg = amgm_arr[mi, ii, si, di]
                        if amg <= 0.0:
                            continue
                        qtmfsd_init[mi, ii, si, di] = max(
                            amg * pwmg_init[ii, si, di] * qxs_init[ii, si, di], _LB
                        )
        _q("qtmfsd", ("marg", "i", "s", "rp"), qtmfsd_init)

        # ------------------------------------------------------------------
        # STUB variables — owned by later blocks (not yet built). Following
        # the GTAP7 leaf-block "first registration wins" dedup pattern
        # (blocks/gtap/trade_armington_bilateral.py, blocks/gtap/__init__.py).
        # ------------------------------------------------------------------
        def _stub(name, doms, init, domain="NonNegativeReals", lower=_LB):
            if name in variables:
                return
            variables[name] = Variable(
                name=name,
                value=init,
                domains=tuple(doms),
                domain=domain,
                lower=lower,
                upper=float("inf"),
            )

        # qo/ps — PRODUCTION block (Task 7): top-nest output qty + supply
        # (basic cost) price, referenced by eq_qf/eq_qo/eq_pe/eq_pst.
        _stub("qo", ("j", "r"), np.ones((nj, nr)))
        _stub("ps", ("j", "r"), np.ones((nj, nr)))

        # pfd/pfm — PRODUCTION block (Task 7): domestic/imported agent
        # prices for firm intermediates (eq_pfd/eq_pfm own them there;
        # eq_qfd/eq_qfm here only READ them).
        _stub("pfd", ("i", "j", "r"), np.ones((ni, nj, nr)))
        _stub("pfm", ("i", "j", "r"), np.ones((ni, nj, nr)))

        # qpd/qgd — FINAL_DEMAND block (household/gov, Task 8): domestic
        # legs of household/government demand, referenced by the qds
        # domestic-absorption identity.
        qpd_init = np.array(
            [
                [max(float(bm.vdpm.get((i, r), 0.0) or 0.0), _LB) for r in regions]
                for i in comms
            ]
        )
        qgd_init = np.array(
            [
                [max(float(bm.vdgm.get((i, r), 0.0) or 0.0), _LB) for r in regions]
                for i in comms
            ]
        )
        _stub("qpd", ("i", "r"), qpd_init)
        _stub("qgd", ("i", "r"), qgd_init)

        equations: list[SymbolicEquation] = []

        # ================================================================
        # TRADE equations
        # ================================================================

        # ---------------- e_qfd_arm (oracle eq_qfd, monolith 1680) --------
        class EqQfdArm(SymbolicEquation):
            name: str = "e_qfd_arm"
            domains: tuple = ("i", "j", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, j, r = indices
                ad = float(pyo_value(m.alpha_dom[i, j, r]))
                if ad <= 0.0:
                    return None
                sigma_d = _ces_cd_sigma(float(pyo_value(m.esubd[i])))
                return m.qfd[i, j, r] == (
                    ad * m.qfa[i, j, r] * (m.pfa[i, j, r] / m.pfd[i, j, r]) ** sigma_d
                )

        equations.append(EqQfdArm())

        # ---------------- e_qfm_arm (oracle eq_qfm, monolith 1696) --------
        class EqQfmArm(SymbolicEquation):
            name: str = "e_qfm_arm"
            domains: tuple = ("i", "j", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, j, r = indices
                ai = float(pyo_value(m.alpha_imp[i, j, r]))
                if ai <= 0.0:
                    return None
                sigma_d = _ces_cd_sigma(float(pyo_value(m.esubd[i])))
                return m.qfm[i, j, r] == (
                    ai * m.qfa[i, j, r] * (m.pfa[i, j, r] / m.pfm[i, j, r]) ** sigma_d
                )

        equations.append(EqQfmArm())

        # ---------------- e_qfa (oracle eq_qf, monolith 1583) -------------
        # NOTE: the oracle's eq_qf reads share_int/esubt/qo/ps (top CES
        # PRODUCTION nest, not yet a block) — those are read via stubs here.
        class EqQfa(SymbolicEquation):
            name: str = "e_qfa"
            domains: tuple = ("i", "j", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, j, r = indices
                sint = float(pyo_value(m.share_int[i, j, r]))
                if sint <= 0.0:
                    return None
                sigma_top = float(pyo_value(m.esubt[j])) if hasattr(m, "esubt") else 0.0
                if abs(sigma_top) < 1e-8:
                    return m.qfa[i, j, r] == sint * m.qo[j, r]
                sigma_top = _ces_cd_sigma(sigma_top)
                return m.qfa[i, j, r] == (
                    sint * m.qo[j, r] * (m.ps[j, r] / m.pfa[i, j, r]) ** sigma_top
                )

        equations.append(EqQfa())

        # ---------------- e_pfa (oracle eq_pf_int, monolith 1660) ---------
        class EqPfa(SymbolicEquation):
            name: str = "e_pfa"
            domains: tuple = ("i", "j", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, j, r = indices
                ad = float(pyo_value(m.alpha_dom[i, j, r]))
                ai = float(pyo_value(m.alpha_imp[i, j, r]))
                if ad + ai <= 1e-12:
                    return None
                sigma_d = _ces_cd_sigma(float(pyo_value(m.esubd[i])))
                expo = 1.0 - sigma_d
                return m.pfa[i, j, r] ** expo == (
                    ad * m.pfd[i, j, r] ** expo + ai * m.pfm[i, j, r] ** expo
                )

        equations.append(EqPfa())

        # ---------------- e_qxs (oracle eq_qxs, monolith 2158) ------------
        class EqQxs(SymbolicEquation):
            name: str = "e_qxs"
            domains: tuple = ("i", "s", "rp")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, src, dst = indices
                ax = float(pyo_value(m.alpha_xs[i, src, dst]))
                if ax <= 0.0:
                    return None
                sigma_m = _ces_cd_sigma(float(pyo_value(m.esubm[i])))
                return m.qxs[i, src, dst] == (
                    ax * m.qim[i, dst] * (m.pim[i, dst] / m.pms[i, src, dst]) ** sigma_m
                )

        equations.append(EqQxs())

        # ---------------- e_pms (oracle eq_pms, monolith 2131) ------------
        class EqPms(SymbolicEquation):
            name: str = "e_pms"
            domains: tuple = ("i", "s", "rp")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, src, dst = indices
                if float(pyo_value(m.alpha_xs[i, src, dst])) <= 0.0:
                    return None
                return m.pms[i, src, dst] == m.pmcif[i, src, dst] * (
                    1.0 + m.tms[i, src, dst]
                )

        equations.append(EqPms())

        # ---------------- e_pmcif (oracle eq_pmcif, monolith 2120) --------
        class EqPmcif(SymbolicEquation):
            name: str = "e_pmcif"
            domains: tuple = ("i", "s", "rp")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, src, dst = indices
                if (
                    pyo_value(m.qxs[i, src, dst]) <= 1e-8
                    or float(pyo_value(m.alpha_xs[i, src, dst])) <= 0.0
                ):
                    return None
                return m.pmcif[i, src, dst] == m.ps[i, src] + m.pwmg[i, src, dst]

        equations.append(EqPmcif())

        # ---------------- e_pe (oracle eq_pe, monolith 2095) --------------
        class EqPe(SymbolicEquation):
            name: str = "e_pe"
            domains: tuple = ("i", "s", "rp")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, src, dst = indices
                if (
                    pyo_value(m.qxs[i, src, dst]) <= 1e-8
                    or float(pyo_value(m.alpha_xs[i, src, dst])) <= 0.0
                ):
                    return None
                return m.pe[i, src, dst] == m.ps[i, src] * (1.0 + m.txs[i, src, dst])

        equations.append(EqPe())

        # ---------------- e_pim (oracle eq_pim, monolith 2140) ------------
        class EqPim(SymbolicEquation):
            name: str = "e_pim"
            domains: tuple = ("i", "rp")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, dst = indices
                terms = [
                    (src, float(pyo_value(m.alpha_xs[i, src, dst])))
                    for src in m.s
                    if pyo_value(m.alpha_xs[i, src, dst]) > 0.0
                ]
                if not terms:
                    return None
                sigma_m = _ces_cd_sigma(float(pyo_value(m.esubm[i])))
                expo = 1.0 - sigma_m
                return m.pim[i, dst] ** expo == sum(
                    ax * m.pms[i, src, dst] ** expo for src, ax in terms
                )

        equations.append(EqPim())

        # ---------------- e_qds (documented oracle vds identity) ----------
        class EqQds(SymbolicEquation):
            name: str = "e_qds"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                i, r = indices
                return m.qds[i, r] == (
                    sum(m.qfd[i, j, r] for j in m.j) + m.qpd[i, r] + m.qgd[i, r]
                )

        equations.append(EqQds())

        # ================================================================
        # MARGINS equations
        # ================================================================

        # ---------------- e_pst (oracle eq_pst, monolith 2189) ------------
        class EqPst(SymbolicEquation):
            name: str = "e_pst"
            domains: tuple = ("marg", "r")

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                mg, r = indices
                return m.pst[mg, r] == m.ps[mg, r]

        equations.append(EqPst())

        # ---------------- e_ptmg (oracle eq_ptmg, monolith 2195) ----------
        class EqPtmg(SymbolicEquation):
            name: str = "e_ptmg"
            domains: tuple = ("marg",)

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                (mg,) = indices
                terms = [
                    (r, float(pyo_value(m.share_st[mg, r])))
                    for r in m.r
                    if pyo_value(m.share_st[mg, r]) > 0.0
                ]
                if not terms:
                    return None
                return m.ptmg[mg] == sum(share * m.pst[mg, r] for r, share in terms)

        equations.append(EqPtmg())

        # ---------------- e_qtm (oracle eq_qtm, monolith 2211) ------------
        class EqQtm(SymbolicEquation):
            name: str = "e_qtm"
            domains: tuple = ("marg",)

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                (mg,) = indices
                terms = []
                for i in m.i:
                    for src in m.s:
                        for dst in m.rp:
                            amg = float(pyo_value(m.amgm[mg, i, src, dst]))
                            if amg <= 0.0:
                                continue
                            terms.append(amg * m.pwmg[i, src, dst] * m.qxs[i, src, dst])
                if not terms:
                    return None
                return m.ptmg[mg] * m.qtm[mg] == sum(terms)

        equations.append(EqQtm())

        # ---------------- e_qst (oracle eq_qst, monolith 2228) ------------
        class EqQst(SymbolicEquation):
            name: str = "e_qst"
            domains: tuple = ("marg", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                mg, r = indices
                sh = float(pyo_value(m.share_st[mg, r]))
                if sh <= 0.0:
                    return None
                return m.qst[mg, r] == sh * m.qtm[mg]

        equations.append(EqQst())

        # ---------------- e_pwmg (oracle eq_pwmg, monolith 2107) ----------
        pwmg_0_map = _to_dict(d.pwmg_0)

        class EqPwmg(SymbolicEquation):
            name: str = "e_pwmg"
            domains: tuple = ("i", "s", "rp")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, src, dst = indices
                # Oracle reads self.derived.pwmg_0.get((i,s,d), 0.0) directly
                # (a captured Python dict, NOT a Pyomo Param) — monolith 2108.
                pwmg0_val = float(pwmg_0_map.get((i, src, dst), 0.0) or 0.0)
                if pwmg0_val <= 1e-12:
                    return None
                return m.pwmg[i, src, dst] == pwmg0_val * sum(
                    float(pyo_value(m.amgm[mg, i, src, dst])) * m.ptmg[mg]
                    for mg in m.marg
                    if pyo_value(m.amgm[mg, i, src, dst]) > 0.0
                )

        equations.append(EqPwmg())

        # ---------------- e_qtmfsd (documented eq_qtm summand) ------------
        class EqQtmfsd(SymbolicEquation):
            name: str = "e_qtmfsd"
            domains: tuple = ("marg", "i", "s", "rp")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                mg, i, src, dst = indices
                amg = float(pyo_value(m.amgm[mg, i, src, dst]))
                if amg <= 0.0:
                    return None
                return m.qtmfsd[mg, i, src, dst] == (
                    amg * m.pwmg[i, src, dst] * m.qxs[i, src, dst] / m.ptmg[mg]
                )

        equations.append(EqQtmfsd())

        return equations
