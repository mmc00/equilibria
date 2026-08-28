"""GTAP6 DEMAND/UTILITY block (leaf unit).

Ports the v6.2 monolith's household CDE demand, government Cobb-Douglas
demand, and investment-as-a-sector (``cgds``) identities, following the
same fidelity discipline ``blocks/gtap6/trade_armington.py``/
``blocks/gtap6/production.py``/``blocks/gtap6/factor.py`` used for the
other leaf units.

This block owns 16 of the 18 IDs in ``_GTAP6_FINAL_DEMAND``: ``e_qpd,
e_qpm, e_qp, e_pp, e_pq, e_up, e_qgd, e_qgm, e_qg, e_pg, e_pgov, e_ug,
e_qcgds, e_pcgds, e_qfd_cgds, e_qfm_cgds``. The remaining two,
``e_yp``/``e_yg`` (household/gov income identities), are reserved for
Task 9b's ``IncomeClosureBlock`` per the controller's ruling — confirmed
by grep that ``_GTAP6_FINAL_DEMAND`` really does contain all 18 names
(``gtap6_contract.py`` lines 58-80), and that this split is internally
consistent: the oracle's own ``eq_yp``/``eq_yg`` (monolith 2414/2424) read
``m.c_p``/``m.c_g``/``m.y``/``m.pcons`` and belong economically with the
Phase 3.38 ``sav``-as-Var income/closure family this block does NOT touch.

CDE fidelity note (task brief's own warning, verified before transcribing
by reading the 3 phase-history docs at ``gtap/v62-multiperiod``):

  - Phase 3.19 (``docs/findings/gtap_v62_phase319_cde_preferences.md``)
    first replaced the Cobb-Douglas household demand with a *log-linear*
    CDE approximation (frozen EP/EY elasticities, first-order Taylor
    expansion of the true CDE around the benchmark) — this is the WRONG
    (superseded) form to transcribe.
  - Phase 3.20 (``docs/findings/gtap_v62_phase320_levels_cde.md``)
    replaced it again with the TRUE Hanoch-Hertel **levels** CDE
    (Hanoch 1975, HT F1-F3), where the expenditure shares are endogenous
    functions of ``(up, pp, yp)`` rather than frozen at the benchmark.
    This is the version actually wired in the oracle today (verified by
    reading ``eq_qp_rule``/``eq_pcons_rule``/``eq_up_rule``,
    monolith 1789-1920) and the one transcribed below.
  - Phase 3.21 (``docs/findings/gtap_v62_phase321_cde_income_split.md``)
    only touches the ``yp``/``yg`` income-SPLIT equations (Task 9b's
    equations, not this block's) — irrelevant to ``e_qp``/``e_pp``/
    ``e_pq``/``e_up`` here.

Oracle -> contract equation-name mapping (grep-verified against
``scripts/gtap6/_v62_monolith_oracle.py``; some are renames, matching the
pattern established by Tasks 6-8):

  e_qpd -> eq_qpd_rule    (monolith 1842) — household domestic demand
           (CES first-order condition, alpha_dom_hhd-weighted).
  e_qpm -> eq_qpm_rule    (1854) — household imported demand.
  e_qp  -> eq_qp_rule     (1828) — Hanoch-Hertel levels CDE demand:
           pp_i * qp_i == yp * share_i(up, pp, yp), share_i defined by
           ``_cde_term`` (1809-1826).
  e_pp  -> eq_pp_rule     (1776) — household Armington composite price
           (CES dual across ppd/ppm), UNCHANGED by the CDE phases (this
           is the bottom-nest aggregator, not part of the CDE utility
           system itself).
  e_pq  -> eq_pcons_rule  (1898) — the CDE EXPENDITURE-FUNCTION IDENTITY
           (HT F1-F3): sum_i share_i(up, pp, yp) == 1, which implicitly
           defines ``up``. The oracle's own docstring (1895-1897) says it
           "re-use[s] the variable name `pcons` to host this identity for
           closure-matching reasons" — i.e. despite the Pyomo attribute
           being named ``pcons``, this Constraint is NOT a linear price
           index, it IS the CDE aggregator the contract calls
           ``e_pq``/``pq``. This block's own ``pq`` Var is therefore a
           rename of the oracle's ``pcons``, exactly as Task 6 renamed
           ``qf``/``pf_int`` to ``qfa``/``pfa``.
  e_up  -> eq_up_rule     (1916) — welfare identity ``up * pq == yp /
           yp_0`` (renamed from the oracle's ``up * pcons == yp/yp_0``,
           consistent with the ``pq`` rename above).
  e_qgd -> eq_qgd_rule    (1973) — gov domestic demand (CES FOC).
  e_qgm -> eq_qgm_rule    (1985) — gov imported demand.
  e_qg  -> eq_qg_rule     (1964) — gov Cobb-Douglas budget allocation:
           pg_i * qg_i == share_gov_cd_i * yg (v6.2 has no ESUBG so gov
           demand stays Cobb-Douglas even after the household CDE
           upgrade — confirmed by the oracle's OWN docstring at monolith
           1926-1939, "v6.2 uses Cobb-Douglas by default (no ESBG in
           v6.2 TAB)").
  e_pg  -> eq_pg_rule     (1950) — gov Armington composite price.
  e_pgov -> eq_pgov_rule  (2012) — gov CD price index (linear aggregator,
           NOT a CDE construct — v6.2 gov demand has no CDE analogue).
  e_ug  -> eq_ug_rule     (2027) — gov utility = real gov expenditure.
  e_qcgds -> eq_qcgds_rule (2053) — investment sector output identity:
           qcgds(cgds,r) == qo(cgds,r) (qo is ProductionBlock's stub,
           already declared over the full ``j`` = prod_comm domain that
           includes cgds).
  e_pcgds -> eq_pcgds_rule (2058) — investment sector price identity:
           pcgds(cgds,r) == ps(cgds,r).
  e_qfd_cgds / e_qfm_cgds -> the SAME oracle ``eq_qfd``/``eq_qfm``
           Constraints TradeArmingtonBlock already ports (as
           ``e_qfd_arm``/``e_qfm_arm``, domains (i,j,r) over the FULL
           ``j`` = prod_comm including cgds) -- RESTRICTED here to
           ``j == cgds`` only. This is not a re-derivation: v6.2 has no
           separate "investment agent" the way GTAP7 does (module
           docstring of ``_add_investment_identities``, monolith 2037-
           2050, "v6.2 treats investment as an output of the CGDS
           sector, not an explicit agent"), so the cgds intermediate-
           demand cells of eq_qfd/eq_qfm ARE the investment-good
           production recipe the contract names e_qfd_cgds/e_qfm_cgds.
           Declaring a second SymbolicEquation under a new name for the
           same j==cgds cells (rather than only relying on
           TradeArmingtonBlock's e_qfd_arm/e_qfm_arm, which cover ALL j)
           satisfies the contract's own explicit ID split — the same
           "duplicate the economics under two contract names" precedent
           ProductionBlock's module docstring documents for e_qf/e_pf vs
           e_qfa/e_pfa.

Non-contracted intermediate Vars this block must compute (no equation ID
of their own, exactly the ``pfd``/``pfm`` precedent from
``production.py``'s module docstring): ``ppd``, ``ppm`` (household
domestic/import agent prices, oracle ``eq_ppd``/``eq_ppm``, monolith
1866-1879) and ``pgd``, ``pgm`` (gov domestic/import agent prices, oracle
``eq_pgd``/``eq_pgm``, monolith 1997-2009). These four have NO ID
anywhere in ``gtap6_contract.py`` (grepped: absent) — only the
Constraints that READ them (``e_pp``/``e_pg``) are contracted. This block
declares them as real Vars with economically correct values (not
placeholder stubs), matching how ``production.py`` declared ``pfd``/
``pfm`` for the analogous firm-side role.

FIDELITY: every equation ported here is transcribed byte-for-byte from
the oracle (same Skip conditions, same ``_ces_cd_sigma`` branching, same
CDE ``_cde_term`` helper).
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
    identically in ``_add_household_demand_block`` and
    ``_add_government_demand_block``).
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


class DemandUtilityBlock(Block):
    """GTAP6 household CDE + government CD demand + cgds investment sector."""

    name: str = "GTAP6_DEMAND_UTILITY"
    description: str = (
        "GTAP6 final demand: household Hanoch-Hertel levels CDE, "
        "government Cobb-Douglas, investment-as-sector cgds identities"
    )
    sets: Any = None
    params: Any = None
    derived: Any = None

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["r", "i"]

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        regions = list(set_manager.get("r"))
        comms = list(set_manager.get("i"))
        prod_secs = list(set_manager.get("j")) if set_manager.has("j") else list(comms)
        cgds_secs = (
            list(set_manager.get("cgds"))
            if set_manager.has("cgds")
            else list(self.sets.cgds)
        )

        p = self.params
        d = self.derived
        el = p.elasticities
        bm = p.benchmark

        nr, ni = len(regions), len(comms)
        nj = len(prod_secs)
        ncg = len(cgds_secs)

        # ------------------------------------------------------------------
        # Params (mirror the oracle's Pyomo Param declarations).
        # ------------------------------------------------------------------
        def _p1(name, data, dim, default=0.0):
            arr = np.full((len(dim),), default, dtype=float)
            data = _to_dict(data)
            for key, val in data.items():
                try:
                    idx = dim.index(key)
                except (ValueError, TypeError):
                    continue
                arr[idx] = float(val or 0.0)
            return arr

        def _p2(name, data, dims, default=0.0):
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
        parameters["esubd"] = Parameter(
            name="esubd", value=esubd_arr, domains=("i",), mutable=True
        )

        to_arr = _p2("to", d.to, (prod_secs, regions))
        parameters["to"] = Parameter(
            name="to", value=to_arr, domains=("j", "r"), mutable=True
        )

        tpd_arr = _p2("tpd", d.tpd, (comms, regions))
        tpi_arr = _p2("tpi", d.tpi, (comms, regions))
        parameters["tpd"] = Parameter(
            name="tpd", value=tpd_arr, domains=("i", "r"), mutable=True
        )
        parameters["tpi"] = Parameter(
            name="tpi", value=tpi_arr, domains=("i", "r"), mutable=True
        )

        tgd_arr = _p2("tgd", d.tgd, (comms, regions))
        tgi_arr = _p2("tgi", d.tgi, (comms, regions))
        parameters["tgd"] = Parameter(
            name="tgd", value=tgd_arr, domains=("i", "r"), mutable=True
        )
        parameters["tgi"] = Parameter(
            name="tgi", value=tgi_arr, domains=("i", "r"), mutable=True
        )

        alpha_dom_hhd_arr = _p2("alpha_dom_hhd", d.alpha_dom_hhd, (comms, regions))
        alpha_imp_hhd_arr = _p2("alpha_imp_hhd", d.alpha_imp_hhd, (comms, regions))
        parameters["alpha_dom_hhd"] = Parameter(
            name="alpha_dom_hhd", value=alpha_dom_hhd_arr, domains=("i", "r")
        )
        parameters["alpha_imp_hhd"] = Parameter(
            name="alpha_imp_hhd", value=alpha_imp_hhd_arr, domains=("i", "r")
        )

        alpha_dom_gov_arr = _p2("alpha_dom_gov", d.alpha_dom_gov, (comms, regions))
        alpha_imp_gov_arr = _p2("alpha_imp_gov", d.alpha_imp_gov, (comms, regions))
        parameters["alpha_dom_gov"] = Parameter(
            name="alpha_dom_gov", value=alpha_dom_gov_arr, domains=("i", "r")
        )
        parameters["alpha_imp_gov"] = Parameter(
            name="alpha_imp_gov", value=alpha_imp_gov_arr, domains=("i", "r")
        )

        share_gov_cd_arr = _p2("share_gov_cd", d.share_gov_cd, (comms, regions))
        parameters["share_gov_cd"] = Parameter(
            name="share_gov_cd", value=share_gov_cd_arr, domains=("i", "r")
        )

        # CDE coefficients (Phase 3.20 levels form): CONSHR_0 (share_hhd_cd),
        # INCPAR, SUBPAR, pp_0, yp_0. All read live inside _cde_term below,
        # exactly mirroring the oracle's own closures over `derived`/
        # `params_e` rather than baking them into Pyomo Params (the oracle
        # itself does the same — eq_qp_rule/eq_pcons_rule close over
        # `self.derived`/`self.params.elasticities` directly rather than
        # registering CONSHR_0/INCPAR/SUBPAR as Pyomo Param components).
        share_hhd_cd_map = _to_dict(d.share_hhd_cd)
        incpar_map = _to_dict(el.incpar)
        subpar_map = _to_dict(el.subpar)
        pp_0_map = _to_dict(d.pp_0)
        yp_0_map = _to_dict(d.yp_0)
        pg_0_map = _to_dict(d.pg_0)
        qp_0_map = _to_dict(d.qp_0)

        def _safe_pos(x: float, eps: float = 1e-12) -> float:
            return x if x > eps else eps

        def _cde_term(m, i, r):
            """One summand of the CDE expenditure function / share
            expression (byte-identical to the oracle's ``_cde_term``,
            monolith 1809-1826):
                CONSHR_i_0 * up^(INCPAR*SUBPAR)
                          * ((pp_i/pp_i_0) / (yp/yp_0))^SUBPAR_i
            Returns (share_expression, conshr_0).
            """
            cshr_0 = float(share_hhd_cd_map.get((i, r), 0.0))
            if cshr_0 <= 0.0:
                return None, 0.0
            subp = float(subpar_map.get((i, r), 0.0))
            incp = float(incpar_map.get((i, r), 1.0))
            pp_i_0 = _safe_pos(float(pp_0_map.get((i, r), 1.0)))
            yp_0 = _safe_pos(float(yp_0_map.get(r, 1.0)))
            ratio = (m.pp[i, r] / pp_i_0) / (m.yp[r] / yp_0)
            expfn_term = cshr_0 * (m.up[r] ** (incp * subp)) * (ratio**subp)
            return expfn_term, cshr_0

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

        vdpm_map = _to_dict(bm.vdpm)
        vipm_map = _to_dict(bm.vipm)
        vdgm_map = _to_dict(bm.vdgm)
        vigm_map = _to_dict(bm.vigm)

        # qpd/qpm — household domestic/imported demand.
        qpd_init = np.array(
            [
                [max(float(vdpm_map.get((i, r), 1.0) or 1.0), _LB) for r in regions]
                for i in comms
            ]
        )
        qpm_init = np.array(
            [
                [max(float(vipm_map.get((i, r), 1.0) or 1.0), _LB) for r in regions]
                for i in comms
            ]
        )
        _q("qpd", ("i", "r"), qpd_init)
        _q("qpm", ("i", "r"), qpm_init)

        # qp — household composite Armington demand (qpd + qpm at benchmark).
        qp_init = qpd_init + qpm_init
        _q("qp", ("i", "r"), qp_init)

        # pp — household composite Armington price, benchmark-normalized.
        pp_init = np.array(
            [
                [max(float(pp_0_map.get((i, r), 1.0) or 1.0), _LB) for r in regions]
                for i in comms
            ]
        )
        _price("pp", ("i", "r"), pp_init)

        # pq — CDE expenditure-function aggregator (renamed oracle `pcons`;
        # see module docstring). Phase 3.20 normalization: pq_0 = 1.
        _price("pq", ("r",), np.ones(nr))

        # up — household utility (CDE), Phase 3.20 normalization up_0 = 1.
        _price("up", ("r",), np.ones(nr))

        # qgd/qgm — gov domestic/imported demand.
        qgd_init = np.array(
            [
                [max(float(vdgm_map.get((i, r), 1.0) or 1.0), _LB) for r in regions]
                for i in comms
            ]
        )
        qgm_init = np.array(
            [
                [max(float(vigm_map.get((i, r), 1.0) or 1.0), _LB) for r in regions]
                for i in comms
            ]
        )
        _q("qgd", ("i", "r"), qgd_init)
        _q("qgm", ("i", "r"), qgm_init)

        # qg — gov composite Armington demand.
        qg_init = qgd_init + qgm_init
        _q("qg", ("i", "r"), qg_init)

        # pg — gov composite Armington price, benchmark-normalized.
        pg_init = np.array(
            [
                [max(float(pg_0_map.get((i, r), 1.0) or 1.0), _LB) for r in regions]
                for i in comms
            ]
        )
        _price("pg", ("i", "r"), pg_init)

        # pgov — gov CD price index.
        share_gov_cd_map = _to_dict(d.share_gov_cd)
        pgov_init = np.array(
            [
                max(
                    sum(
                        float(share_gov_cd_map.get((i, r), 0.0) or 0.0)
                        * float(pg_0_map.get((i, r), 1.0) or 1.0)
                        for i in comms
                    ),
                    _LB,
                )
                for r in regions
            ]
        )
        _price("pgov", ("r",), pgov_init)

        # ug — gov utility, ug_0 = 1/pgov_0.
        ug_init = np.array([1.0 / max(float(v), 1e-8) for v in pgov_init])
        _price("ug", ("r",), ug_init)

        # qcgds/pcgds — investment sector output/price.
        vom_map = _to_dict(d.vom)
        qcgds_init = np.array(
            [
                [max(float(vom_map.get((cg, r), 1.0) or 1.0), _LB) for r in regions]
                for cg in cgds_secs
            ]
        )
        _q("qcgds", ("cgds", "r"), qcgds_init)
        _price("pcgds", ("cgds", "r"), np.ones((ncg, nr)))

        # ------------------------------------------------------------------
        # Non-contracted intermediate Vars this block must compute (no
        # equation ID of their own — see module docstring, same precedent
        # as production.py's pfd/pfm): ppd, ppm, pgd, pgm.
        # ------------------------------------------------------------------
        pim_0_map = _to_dict(d.pim_0)

        ppd_init = np.array(
            [
                [
                    max(
                        (1.0 + float(to_arr[prod_secs.index(i), regions.index(r)]))
                        * (1.0 + float(tpd_arr[comms.index(i), regions.index(r)])),
                        _LB,
                    )
                    if i in prod_secs
                    else max(
                        1.0 + float(tpd_arr[comms.index(i), regions.index(r)]), _LB
                    )
                    for r in regions
                ]
                for i in comms
            ]
        )
        ppm_init = np.array(
            [
                [
                    max(
                        float(pim_0_map.get((i, r), 1.0) or 1.0)
                        * (1.0 + float(tpi_arr[comms.index(i), regions.index(r)])),
                        _LB,
                    )
                    for r in regions
                ]
                for i in comms
            ]
        )
        variables["ppd"] = Variable(
            name="ppd",
            value=ppd_init,
            domains=("i", "r"),
            domain="NonNegativeReals",
            lower=_LB,
            upper=float("inf"),
        )
        variables["ppm"] = Variable(
            name="ppm",
            value=ppm_init,
            domains=("i", "r"),
            domain="NonNegativeReals",
            lower=_LB,
            upper=float("inf"),
        )

        pgd_init = np.array(
            [
                [
                    max(
                        (1.0 + float(to_arr[prod_secs.index(i), regions.index(r)]))
                        * (1.0 + float(tgd_arr[comms.index(i), regions.index(r)])),
                        _LB,
                    )
                    if i in prod_secs
                    else max(
                        1.0 + float(tgd_arr[comms.index(i), regions.index(r)]), _LB
                    )
                    for r in regions
                ]
                for i in comms
            ]
        )
        pgm_init = np.array(
            [
                [
                    max(
                        float(pim_0_map.get((i, r), 1.0) or 1.0)
                        * (1.0 + float(tgi_arr[comms.index(i), regions.index(r)])),
                        _LB,
                    )
                    for r in regions
                ]
                for i in comms
            ]
        )
        variables["pgd"] = Variable(
            name="pgd",
            value=pgd_init,
            domains=("i", "r"),
            domain="NonNegativeReals",
            lower=_LB,
            upper=float("inf"),
        )
        variables["pgm"] = Variable(
            name="pgm",
            value=pgm_init,
            domains=("i", "r"),
            domain="NonNegativeReals",
            lower=_LB,
            upper=float("inf"),
        )

        # ------------------------------------------------------------------
        # STUB variables — owned by other blocks.
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

        # pds (j,r) — domestic supply price, owned by ProductionBlock/
        # TradeArmingtonBlock's wider trade chain. Needed here only to
        # compute ppd/pgd's economically-correct seed values above (via
        # the (1+to) proxy, matching the oracle's own eq_ppd_rule which
        # actually reads pds directly: ppd = pds*(1+tpd). This block's
        # SymbolicEquations read `m.pds` off the shared pyomo_model
        # directly, not off this seed.
        _stub("pds", ("j", "r"), np.ones((nj, nr)))

        # pim (i,r) — composite import price, owned by TradeArmingtonBlock
        # (declared there under domains (i,rp); the oracle's own pim is
        # (i,r) — see module docstring). Declared here as a guarded
        # fallback stub under (i,r) so a standalone DemandUtilityBlock
        # build has a complete variable set.
        _stub("pim", ("i", "r"), np.ones((ni, nr)))

        # qo/ps (j,r) — PRODUCTION block: activity output qty + supply
        # price, referenced by e_qcgds/e_pcgds (qcgds==qo, pcgds==ps at
        # the cgds slice).
        _stub("qo", ("j", "r"), np.ones((nj, nr)))
        _stub("ps", ("j", "r"), np.ones((nj, nr)))

        # qfd/qfm (i,j,r) — TRADE block: firm intermediate demand,
        # referenced (restricted to j==cgds) by e_qfd_cgds/e_qfm_cgds.
        _stub("qfd", ("i", "j", "r"), np.ones((ni, nj, nr)))
        _stub("qfm", ("i", "j", "r"), np.ones((ni, nj, nr)))

        # pfa (i,j,r) — TRADE block: Armington composite firm price,
        # read by e_qfd_cgds/e_qfm_cgds (same RHS as e_qfd_arm/e_qfm_arm).
        _stub("pfa", ("i", "j", "r"), np.ones((ni, nj, nr)))

        # pfd/pfm (i,j,r) — PRODUCTION block: domestic/imported agent
        # prices for firm intermediates, read by e_qfd_cgds/e_qfm_cgds.
        _stub("pfd", ("i", "j", "r"), np.ones((ni, nj, nr)))
        _stub("pfm", ("i", "j", "r"), np.ones((ni, nj, nr)))

        # alpha_dom/alpha_imp (i,j,r) — PRODUCTION block: Armington share
        # calibration parameters read by e_qfd_cgds/e_qfm_cgds.
        alpha_dom_full = _p2("alpha_dom", d.alpha_dom, (comms, prod_secs, regions))
        alpha_imp_full = _p2("alpha_imp", d.alpha_imp, (comms, prod_secs, regions))
        if "alpha_dom" not in parameters:
            parameters["alpha_dom"] = Parameter(
                name="alpha_dom", value=alpha_dom_full, domains=("i", "j", "r")
            )
        if "alpha_imp" not in parameters:
            parameters["alpha_imp"] = Parameter(
                name="alpha_imp", value=alpha_imp_full, domains=("i", "j", "r")
            )

        # yp/yg — INCOME_CLOSURE block (Task 9b): household/gov income,
        # read by e_qp/e_pq/e_up/e_qg/e_pgov/e_ug. Declared here as a
        # guarded stub, seeded at the benchmark yp_0/yg_0 (economically
        # correct seed, not a placeholder ones(...)), so a standalone
        # DemandUtilityBlock build is internally consistent even before
        # Task 9b runs.
        yg_0_map = _to_dict(d.yg_0)
        yp_init = np.array(
            [max(float(yp_0_map.get(r, 1.0) or 1.0), _LB) for r in regions]
        )
        yg_init = np.array(
            [max(float(yg_0_map.get(r, 1.0) or 1.0), _LB) for r in regions]
        )
        _stub("yp", ("r",), yp_init)
        _stub("yg", ("r",), yg_init)

        equations: list[SymbolicEquation] = []

        # ================================================================
        # Household CDE demand
        # ================================================================

        # ---------------- e_pp (oracle eq_pp, monolith 1776) --------------
        class EqPp(SymbolicEquation):
            name: str = "e_pp"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, r = indices
                ad = float(pyo_value(m.alpha_dom_hhd[i, r]))
                ai = float(pyo_value(m.alpha_imp_hhd[i, r]))
                if ad + ai <= 1e-12:
                    return None
                sigma_d = _ces_cd_sigma(float(pyo_value(m.esubd[i])))
                exp = 1.0 - sigma_d
                return (
                    m.pp[i, r] ** exp
                    == ad * m.ppd[i, r] ** exp + ai * m.ppm[i, r] ** exp
                )

        equations.append(EqPp())

        # ---------------- e_ppd (oracle eq_ppd, monolith 1866) ------------
        # ppd(i,r) = pds(i,r) * (1 + tpd(i,r)) -- household domestic agent
        # price. MISSING from the original block port (Task 10b
        # diagnostic): `ppd`/`ppm` were declared as real (non-stub) owned
        # variables here but with NO defining equation anywhere in the
        # composed model -- genuinely free variables, part of the ~139-cell
        # gap (alongside pfd/pfm/pgd/pgm/pds) that left the canary solve's
        # IPOPT search with dozens of unconstrained directions.
        class EqPpd(SymbolicEquation):
            name: str = "e_ppd"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, r = indices
                ad = float(pyo_value(m.alpha_dom_hhd[i, r]))
                if ad <= 0.0:
                    return None
                return m.ppd[i, r] == m.pds[i, r] * (1.0 + m.tpd[i, r])

        equations.append(EqPpd())

        # ---------------- e_ppm (oracle eq_ppm, monolith 1874) ------------
        # ppm(i,r) = pim(i,r) * (1 + tpi(i,r)) -- household imported agent
        # price. MISSING from the original block port (see e_ppd comment).
        class EqPpm(SymbolicEquation):
            name: str = "e_ppm"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, r = indices
                ai = float(pyo_value(m.alpha_imp_hhd[i, r]))
                if ai <= 0.0:
                    return None
                return m.ppm[i, r] == m.pim[i, r] * (1.0 + m.tpi[i, r])

        equations.append(EqPpm())

        # ---------------- e_qp (oracle eq_qp, monolith 1828) --------------
        class EqQp(SymbolicEquation):
            name: str = "e_qp"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                i, r = indices
                qp_0 = float(qp_0_map.get((i, r), 0.0))
                if qp_0 <= 1e-12:
                    return m.qp[i, r] == 0.0
                share_expr, cshr_0 = _cde_term(m, i, r)
                if share_expr is None:
                    return m.qp[i, r] == 0.0
                return m.pp[i, r] * m.qp[i, r] == m.yp[r] * share_expr

        equations.append(EqQp())

        # ---------------- e_qpd (oracle eq_qpd, monolith 1842) ------------
        class EqQpd(SymbolicEquation):
            name: str = "e_qpd"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, r = indices
                ad = float(pyo_value(m.alpha_dom_hhd[i, r]))
                if ad <= 0.0:
                    return None
                sigma_d = _ces_cd_sigma(float(pyo_value(m.esubd[i])))
                return m.qpd[i, r] == (
                    ad * m.qp[i, r] * (m.pp[i, r] / m.ppd[i, r]) ** sigma_d
                )

        equations.append(EqQpd())

        # ---------------- e_qpm (oracle eq_qpm, monolith 1854) ------------
        class EqQpm(SymbolicEquation):
            name: str = "e_qpm"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, r = indices
                ai = float(pyo_value(m.alpha_imp_hhd[i, r]))
                if ai <= 0.0:
                    return None
                sigma_d = _ces_cd_sigma(float(pyo_value(m.esubd[i])))
                return m.qpm[i, r] == (
                    ai * m.qp[i, r] * (m.pp[i, r] / m.ppm[i, r]) ** sigma_d
                )

        equations.append(EqQpm())

        # ---------------- e_pq (oracle eq_pcons, monolith 1898) -----------
        # CDE expenditure-function identity: sum_i share_i(up,pp,yp) == 1,
        # implicitly defining `up`. Renamed from the oracle's `pcons`
        # Constraint per the module docstring.
        class EqPq(SymbolicEquation):
            name: str = "e_pq"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                terms = []
                for i in m.i:
                    share_expr, cshr_0 = _cde_term(m, i, r)
                    if share_expr is None:
                        continue
                    terms.append(share_expr)
                if not terms:
                    return None
                return sum(terms) == 1.0

        equations.append(EqPq())

        # ---------------- e_up (oracle eq_up, monolith 1916) --------------
        class EqUp(SymbolicEquation):
            name: str = "e_up"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                yp_0 = max(float(yp_0_map.get(r, 1.0)), 1e-8)
                return m.up[r] * m.pq[r] == m.yp[r] / yp_0

        equations.append(EqUp())

        # ================================================================
        # Government Cobb-Douglas demand
        # ================================================================

        # ---------------- e_pg (oracle eq_pg, monolith 1950) --------------
        class EqPg(SymbolicEquation):
            name: str = "e_pg"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, r = indices
                ad = float(pyo_value(m.alpha_dom_gov[i, r]))
                ai = float(pyo_value(m.alpha_imp_gov[i, r]))
                if ad + ai <= 1e-12:
                    return None
                sigma_d = _ces_cd_sigma(float(pyo_value(m.esubd[i])))
                exp = 1.0 - sigma_d
                return (
                    m.pg[i, r] ** exp
                    == ad * m.pgd[i, r] ** exp + ai * m.pgm[i, r] ** exp
                )

        equations.append(EqPg())

        # ---------------- e_pgd (oracle eq_pgd, monolith 1997) ------------
        # pgd(i,r) = pds(i,r) * (1 + tgd(i,r)) -- government domestic agent
        # price. MISSING from the original block port (see e_ppd comment
        # above in the household nest -- same gap, government nest).
        class EqPgd(SymbolicEquation):
            name: str = "e_pgd"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, r = indices
                ad = float(pyo_value(m.alpha_dom_gov[i, r]))
                if ad <= 0.0:
                    return None
                return m.pgd[i, r] == m.pds[i, r] * (1.0 + m.tgd[i, r])

        equations.append(EqPgd())

        # ---------------- e_pgm (oracle eq_pgm, monolith 2004) ------------
        # pgm(i,r) = pim(i,r) * (1 + tgi(i,r)) -- government imported agent
        # price. MISSING from the original block port.
        class EqPgm(SymbolicEquation):
            name: str = "e_pgm"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, r = indices
                ai = float(pyo_value(m.alpha_imp_gov[i, r]))
                if ai <= 0.0:
                    return None
                return m.pgm[i, r] == m.pim[i, r] * (1.0 + m.tgi[i, r])

        equations.append(EqPgm())

        # ---------------- e_qg (oracle eq_qg, monolith 1964) --------------
        class EqQg(SymbolicEquation):
            name: str = "e_qg"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, r = indices
                cs = float(pyo_value(m.share_gov_cd[i, r]))
                if cs <= 0.0:
                    return None
                return m.pg[i, r] * m.qg[i, r] == cs * m.yg[r]

        equations.append(EqQg())

        # ---------------- e_qgd (oracle eq_qgd, monolith 1973) ------------
        class EqQgd(SymbolicEquation):
            name: str = "e_qgd"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, r = indices
                ad = float(pyo_value(m.alpha_dom_gov[i, r]))
                if ad <= 0.0:
                    return None
                sigma_d = _ces_cd_sigma(float(pyo_value(m.esubd[i])))
                return m.qgd[i, r] == (
                    ad * m.qg[i, r] * (m.pg[i, r] / m.pgd[i, r]) ** sigma_d
                )

        equations.append(EqQgd())

        # ---------------- e_qgm (oracle eq_qgm, monolith 1985) ------------
        class EqQgm(SymbolicEquation):
            name: str = "e_qgm"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, r = indices
                ai = float(pyo_value(m.alpha_imp_gov[i, r]))
                if ai <= 0.0:
                    return None
                sigma_d = _ces_cd_sigma(float(pyo_value(m.esubd[i])))
                return m.qgm[i, r] == (
                    ai * m.qg[i, r] * (m.pg[i, r] / m.pgm[i, r]) ** sigma_d
                )

        equations.append(EqQgm())

        # ---------------- e_pgov (oracle eq_pgov, monolith 2012) ----------
        class EqPgov(SymbolicEquation):
            name: str = "e_pgov"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                (r,) = indices
                terms = [
                    float(pyo_value(m.share_gov_cd[i, r])) * m.pg[i, r]
                    for i in m.i
                    if pyo_value(m.share_gov_cd[i, r]) > 0.0
                ]
                if not terms:
                    return None
                return m.pgov[r] == sum(terms)

        equations.append(EqPgov())

        # ---------------- e_ug (oracle eq_ug, monolith 2027) --------------
        class EqUg(SymbolicEquation):
            name: str = "e_ug"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                yg_0 = max(float(yg_0_map.get(r, 1.0)), 1e-8)
                return m.ug[r] * m.pgov[r] == m.yg[r] / yg_0

        equations.append(EqUg())

        # ================================================================
        # Investment-as-a-sector (cgds)
        # ================================================================

        # ---------------- e_qcgds (oracle eq_qcgds, monolith 2053) -------
        class EqQcgds(SymbolicEquation):
            name: str = "e_qcgds"
            domains: tuple = ("cgds", "r")

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                cg, r = indices
                return m.qcgds[cg, r] == m.qo[cg, r]

        equations.append(EqQcgds())

        # ---------------- e_pcgds (oracle eq_pcgds, monolith 2058) -------
        class EqPcgds(SymbolicEquation):
            name: str = "e_pcgds"
            domains: tuple = ("cgds", "r")

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                cg, r = indices
                return m.pcgds[cg, r] == m.ps[cg, r]

        equations.append(EqPcgds())

        # ---------------- e_qfd_cgds (oracle eq_qfd, j==cgds slice) -------
        # Same algebra as TradeArmingtonBlock's e_qfd_arm, restricted to
        # j in cgds (see module docstring: v6.2's investment "agent" IS
        # the cgds intermediate-demand recipe, there is no separate
        # investment-agent Armington nest to port).
        class EqQfdCgds(SymbolicEquation):
            name: str = "e_qfd_cgds"
            domains: tuple = ("i", "cgds", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, cg, r = indices
                ad = float(pyo_value(m.alpha_dom[i, cg, r]))
                if ad <= 0.0:
                    return None
                sigma_d = _ces_cd_sigma(float(pyo_value(m.esubd[i])))
                return m.qfd[i, cg, r] == (
                    ad
                    * m.qfa[i, cg, r]
                    * (m.pfa[i, cg, r] / m.pfd[i, cg, r]) ** sigma_d
                )

        equations.append(EqQfdCgds())

        # ---------------- e_qfm_cgds (oracle eq_qfm, j==cgds slice) -------
        class EqQfmCgds(SymbolicEquation):
            name: str = "e_qfm_cgds"
            domains: tuple = ("i", "cgds", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, cg, r = indices
                ai = float(pyo_value(m.alpha_imp[i, cg, r]))
                if ai <= 0.0:
                    return None
                sigma_d = _ces_cd_sigma(float(pyo_value(m.esubd[i])))
                return m.qfm[i, cg, r] == (
                    ai
                    * m.qfa[i, cg, r]
                    * (m.pfa[i, cg, r] / m.pfm[i, cg, r]) ** sigma_d
                )

        equations.append(EqQfmCgds())

        return equations
