"""GTAP6 PRODUCTION block (leaf unit).

Ports the v6.2 monolith's top production nest (CES between value-added and
the aggregate intermediate composite, with Leontief as the ``esubt == 0``
special case) + VA CES nest, VERBATIM from
``scripts/gtap6/_v62_monolith_oracle.py`` (``GTAP6MonolithOracle``,
``_add_production_block``), following the same fidelity discipline
``blocks/gtap6/trade_armington.py`` used for the trade unit.

v6.2's top nest is NOT pure Leontief — the oracle's ``eq_qo_rule``/
``eq_va_rule``/``eq_qf_rule`` implement a genuine CES cost function with a
``σ = esubt(j)`` elasticity per PROD_COMM sector, collapsing to the
Leontief identity only when ``esubt(j) ≈ 0`` (v6.2's calibrated default).
This block transcribes BOTH branches exactly as the oracle does — it does
NOT assume Leontief a priori, per the task brief's own caution to read the
oracle's actual equation bodies rather than guess the functional form.

Oracle -> contract equation-name mapping (the oracle's method/Constraint
names differ from the contract's ``e_*`` IDs; this is NOT a re-derivation,
only a rename, exactly as Task 6 did for ``e_qfa``/``e_pfa`` -> ``eq_qf``/
``eq_pf_int``):

  e_qo  -> eq_market_rule (oracle ``_add_market_clearing``, monolith 2264)
           — "qo = activity output identity": qo*(1+to) = sum of uses.
           NOTE this is a DIFFERENT oracle Constraint than the one the
           oracle happens to also NAME ``eq_qo`` (see e_ps below) — the
           contract's own comment ("qo = activity output identity") only
           matches the market-clearing balance, not the zero-profit dual.
  e_ps  -> eq_qo_rule    (oracle ``_add_production_block``, monolith 1524)
           — "ps = unit cost of production (zero-profit)": this IS the
           oracle's ``eq_qo`` Constraint, which despite its name pins
           ``ps`` via the CES/Leontief zero-profit condition. The oracle
           names the Constraint after the Var it was written before
           (``qo``), but its body solves for ``ps``.
  e_qf  -> eq_qf_rule    (1583) — production top-nest intermediate
           composite demand (CES/Leontief across (i,j,r)).
  e_pf  -> eq_pf_int_rule (1660) — Armington composite price for
           intermediates, dual to e_qf.
  e_qva -> eq_va_rule    (1565) — value-added demand from the top nest.
  e_pva -> eq_pva_rule   (1601) — value-added composite price (CES
           across factors).
  e_qfe -> eq_qfe_rule   (1619) — factor demand from the VA CES nest.
  e_pfe -> eq_pfe_rule   (1643) — factor agent price (regional factor
           wage + factor tax wedge).

IMPORTANT — a genuine oracle-level duplication, not a block bug: the
oracle's ``eq_qf``/``eq_pf_int`` Constraints (which this block ports as
``e_qf``/``e_pf``) are THE SAME equations Task 6's ``TradeArmingtonBlock``
already ported as ``e_qfa``/``e_pfa`` (see that block's module docstring).
The v6.2 contract (``gtap6_contract.py``) assigns these two DIFFERENT
equation IDs across two DIFFERENT blocks (``_GTAP6_PRODUCTION.e_qf``/
``e_pf`` vs ``_GTAP6_TRADE.e_qfa``/``e_pfa``) even though the underlying
oracle has only one ``qf``/``pf_int`` Var pair and one Constraint pair.
This block therefore declares its OWN ``qf``/``pf`` variables (distinct
dict keys from TradeArmingtonBlock's ``qfa``/``pfa``) and its OWN
``e_qf``/``e_pf`` SymbolicEquations with byte-identical algebra to the
oracle's ``eq_qf_rule``/``eq_pf_int_rule`` — this duplicates the economics
under two contract names, exactly mirroring how the oracle's single
Constraint pair is read from two different nest "angles" (top-production
CES vs top-Armington CES) by the contract's authors. A future composer
task should decide whether to alias these to a single shared Var (as
Task 6 already aliases oracle ``qf``/``pf_int`` onto ``qfa``/``pfa`` for
the test oracle) rather than solve them as two independent variable pairs.

FIDELITY: every equation ported here is transcribed byte-for-byte from the
oracle (same Skip conditions, same ``_ces_cd_sigma``/Leontief branching).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from equilibria.blocks.base import Block
from equilibria.core.parameters import Parameter
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable

_LB = 1e-6


def _eps_sigma(sigma: float) -> bool:
    """True if sigma is small enough to treat as Leontief.

    Verbatim transcription of the oracle's local helper (redefined
    identically inside ``_add_production_block``, monolith line 1503).
    """
    return abs(sigma) < 1e-8


def _ces_cd_sigma(sigma: float) -> float:
    """Perturb sigma when it equals 1.0 to avoid (1-sigma) = 0 pathologies.

    Verbatim transcription of the oracle's local helper (monolith 1507).
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


class ProductionBlock(Block):
    """GTAP6 top CES production nest + VA CES nest."""

    name: str = "GTAP6_PRODUCTION"
    description: str = (
        "GTAP6 top production nest (CES VA vs intermediate composite, "
        "Leontief when esubt=0) + VA CES across factors"
    )
    sets: Any = None
    params: Any = None
    derived: Any = None

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["r", "i", "j", "f"]

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        regions = list(set_manager.get("r"))
        comms = list(set_manager.get("i"))
        prod_secs = list(set_manager.get("j")) if set_manager.has("j") else list(comms)
        factors = list(set_manager.get("f")) if set_manager.has("f") else []
        # Bilateral destination alias (needed for eq_market's export term,
        # i.e. sum_d qxs[i,r,d]) — falls back to "r" if not registered
        # separately, matching TradeArmingtonBlock's own fallback.
        dests = list(set_manager.get("rp")) if set_manager.has("rp") else list(regions)
        margins = list(set_manager.get("marg")) if set_manager.has("marg") else []

        p = self.params
        d = self.derived
        el = p.elasticities
        bm = p.benchmark

        nr, ni, nj, nf = len(regions), len(comms), len(prod_secs), len(factors)

        # ------------------------------------------------------------------
        # Params (mirror the oracle's Pyomo Param declarations).
        # ------------------------------------------------------------------
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

        esubt_arr = np.array(
            [float(el.esubt.get(j, 0.0)) for j in prod_secs], dtype=float
        )
        esubva_arr = np.array(
            [float(el.esubva.get(j, 1.0)) for j in prod_secs], dtype=float
        )
        parameters["esubt"] = Parameter(
            name="esubt", value=esubt_arr, domains=("j",), mutable=True
        )
        parameters["esubva"] = Parameter(
            name="esubva", value=esubva_arr, domains=("j",), mutable=True
        )

        to_arr = _p2("to", d.to, (prod_secs, regions))
        vom_arr = _p2("vom", d.vom, (prod_secs, regions))
        va_total_arr = _p2("va_total", d.va_total, (prod_secs, regions))
        parameters["to"] = Parameter(
            name="to", value=to_arr, domains=("j", "r"), mutable=True
        )
        parameters["vom"] = Parameter(
            name="vom", value=vom_arr, domains=("j", "r"), mutable=True
        )
        parameters["va_total"] = Parameter(
            name="va_total", value=va_total_arr, domains=("j", "r"), mutable=True
        )

        share_va_arr = _p2("share_va", d.share_va, (prod_secs, regions))
        parameters["share_va"] = Parameter(
            name="share_va", value=share_va_arr, domains=("j", "r")
        )

        share_int_arr = _p2("share_int", d.share_int, (comms, prod_secs, regions))
        parameters["share_int"] = Parameter(
            name="share_int", value=share_int_arr, domains=("i", "j", "r")
        )

        share_fac_arr = _p2("share_fac", d.share_fac, (factors, prod_secs, regions))
        parameters["share_fac"] = Parameter(
            name="share_fac", value=share_fac_arr, domains=("f", "j", "r")
        )

        tf_arr = _p2("tf", d.tf, (factors, prod_secs, regions))
        parameters["tf"] = Parameter(
            name="tf", value=tf_arr, domains=("f", "j", "r"), mutable=True
        )

        alpha_dom_arr = _p2("alpha_dom", d.alpha_dom, (comms, prod_secs, regions))
        alpha_imp_arr = _p2("alpha_imp", d.alpha_imp, (comms, prod_secs, regions))
        parameters["alpha_dom"] = Parameter(
            name="alpha_dom", value=alpha_dom_arr, domains=("i", "j", "r")
        )
        parameters["alpha_imp"] = Parameter(
            name="alpha_imp", value=alpha_imp_arr, domains=("i", "j", "r")
        )

        esubd_arr = np.array([float(el.esubd.get(i, 1.0)) for i in comms], dtype=float)
        parameters["esubd"] = Parameter(
            name="esubd", value=esubd_arr, domains=("i",), mutable=True
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

        # qo (j,r) — output of sector j in region r, seeded from the
        # production-cost base (vop), falling back to vom (oracle
        # monolith 811-825).
        vop_map = _to_dict(d.vop)
        vom_map = _to_dict(d.vom)
        qo_init = np.array(
            [
                [
                    max(
                        float(vop_map.get((j, r), vom_map.get((j, r), 1.0)) or 1.0),
                        _LB,
                    )
                    for r in regions
                ]
                for j in prod_secs
            ]
        )
        _q("qo", ("j", "r"), qo_init)

        # ps (j,r) — supply (cost) price, benchmark-normalized to 1.0
        # (oracle monolith 826-833).
        _price("ps", ("j", "r"), np.ones((nj, nr)))

        # va (=contract's qva) (j,r) — value-added aggregate quantity,
        # seeded from va_total (oracle monolith 856-862).
        va_total_map = _to_dict(d.va_total)
        qva_init = np.array(
            [
                [max(float(va_total_map.get((j, r), 1.0) or 1.0), _LB) for r in regions]
                for j in prod_secs
            ]
        )
        _q("qva", ("j", "r"), qva_init)

        # pva (j,r) — VA composite price, benchmark-normalized to 1.0.
        _price("pva", ("j", "r"), np.ones((nj, nr)))

        # qfe (f,j,r) — factor demand, seeded from VFM (oracle monolith
        # 875-882).
        vfm_map = _to_dict(bm.vfm)
        qfe_init = np.array(
            [
                [
                    [
                        max(float(vfm_map.get((f, j, r), 1.0) or 1.0), _LB)
                        for r in regions
                    ]
                    for j in prod_secs
                ]
                for f in factors
            ]
        )
        _q("qfe", ("f", "j", "r"), qfe_init)

        # pfe (f,j,r) — factor agent price, benchmark-normalized to 1.0.
        _price("pfe", ("f", "j", "r"), np.ones((nf, nj, nr)))

        # qf (=oracle's qf, production-nest angle; contract e_qf) (i,j,r)
        # — intermediate composite demand from the top production nest.
        # Distinct dict key from TradeArmingtonBlock's qfa (see module
        # docstring: the oracle has one qf Var, the contract names it
        # twice under two block-owned Vars).
        vdfm_map = _to_dict(bm.vdfm) if hasattr(bm, "vdfm") else {}
        vifm_map = _to_dict(bm.vifm) if hasattr(bm, "vifm") else {}
        qf_init = np.array(
            [
                [
                    [
                        max(
                            float(vdfm_map.get((i, j, r), 0.0) or 0.0)
                            + float(vifm_map.get((i, j, r), 0.0) or 0.0),
                            _LB,
                        )
                        for r in regions
                    ]
                    for j in prod_secs
                ]
                for i in comms
            ]
        )
        _q("qf", ("i", "j", "r"), qf_init)

        # pf (=oracle's pf_int, production-nest angle; contract e_pf)
        # (i,j,r) — Armington composite price dual to qf.
        pf_int0_map = _to_dict(d.pf_int_0)
        pf_init = np.array(
            [
                [
                    [
                        max(float(pf_int0_map.get((i, j, r), 1.0) or 1.0), _LB)
                        for r in regions
                    ]
                    for j in prod_secs
                ]
                for i in comms
            ]
        )
        _price("pf", ("i", "j", "r"), pf_init)

        # ------------------------------------------------------------------
        # STUB variables — owned by other blocks (not yet built, or
        # already built by TradeArmingtonBlock). Following the leaf-block
        # "first registration wins" dedup pattern.
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

        # pfd/pfm — agent prices for firm intermediates. This block OWNS
        # these per Task 6's own stub comment ("PRODUCTION block (Task
        # 7): domestic/imported agent prices for firm intermediates"),
        # but the oracle's eq_pfd/eq_pfm Constraints have NO corresponding
        # ID in _GTAP6_PRODUCTION (or anywhere in the contract) — only
        # eq_qfd/eq_qfm's READS of pfd/pfm are contracted (as e_qfd_arm/
        # e_qfm_arm, already ported by TradeArmingtonBlock). Declaring the
        # real values here (rather than leaving TradeArmingtonBlock's
        # placeholder ones(...) stub in place) is the economically correct
        # seed even though no e_pfd/e_pfm equation is wired in this task;
        # a later task should wire eq_pfd/eq_pfm's Constraints once the
        # contract grows an ID for them.
        tfd_map = _to_dict(d.tfd)
        tfi_map = _to_dict(d.tfi)
        pim0_map = _to_dict(d.pim_0) if hasattr(d, "pim_0") else {}
        pfd_init = np.array(
            [
                [
                    [
                        max(
                            (1.0 + float(to_arr[comms.index(i), regions.index(r)]))
                            * (1.0 + float(tfd_map.get((i, j, r), 0.0) or 0.0)),
                            _LB,
                        )
                        for r in regions
                    ]
                    for j in prod_secs
                ]
                for i in comms
            ]
        )
        pfm_init = np.array(
            [
                [
                    [
                        max(
                            float(pim0_map.get((i, r), 1.0) or 1.0)
                            * (1.0 + float(tfi_map.get((i, j, r), 0.0) or 0.0)),
                            _LB,
                        )
                        for r in regions
                    ]
                    for j in prod_secs
                ]
                for i in comms
            ]
        )
        variables["pfd"] = Variable(
            name="pfd",
            value=pfd_init,
            domains=("i", "j", "r"),
            domain="NonNegativeReals",
            lower=_LB,
            upper=float("inf"),
        )
        variables["pfm"] = Variable(
            name="pfm",
            value=pfm_init,
            domains=("i", "j", "r"),
            domain="NonNegativeReals",
            lower=_LB,
            upper=float("inf"),
        )

        # pfactor (f,r) — regional factor wage (oracle's model.pf(f,r)).
        # Renamed to avoid colliding with this block's OWN pf(i,j,r)
        # (the Armington composite intermediate price, contract e_pf).
        # Owned eventually by the FACTOR_MARKETS block (Task 9a) — see
        # gtap6_contract.py's e_pe_endw "pe(f,r) factor wage (mobile
        # factors)". Declared here as a genuinely-needed stub since
        # e_pfe's RHS reads it directly (oracle eq_pfe_rule: pfe[f,j,r]
        # == pf[f,r] * (1+tf[f,j,r])).
        _stub("pfactor", ("f", "r"), np.ones((nf, nr)))

        # qfd/qxs/qst — needed by eq_market's uses-side identity (e_qo).
        # qfd is already owned by TradeArmingtonBlock; qxs likewise;
        # qst (margin sale supply) likewise. Guarded by the dedup check
        # so this block does not clobber them if TradeArmingtonBlock ran
        # first in composition order.
        qfd_init = np.array(
            [
                [
                    [
                        max(float(vdfm_map.get((i, j, r), 0.0) or 0.0), _LB)
                        for r in regions
                    ]
                    for j in prod_secs
                ]
                for i in comms
            ]
        )
        _stub("qfd", ("i", "j", "r"), qfd_init)

        qpd_map = _to_dict(bm.vdpm) if hasattr(bm, "vdpm") else {}
        qgd_map = _to_dict(bm.vdgm) if hasattr(bm, "vdgm") else {}
        qpd_init = np.array(
            [
                [max(float(qpd_map.get((i, r), 0.0) or 0.0), _LB) for r in regions]
                for i in comms
            ]
        )
        qgd_init = np.array(
            [
                [max(float(qgd_map.get((i, r), 0.0) or 0.0), _LB) for r in regions]
                for i in comms
            ]
        )
        _stub("qpd", ("i", "r"), qpd_init)
        _stub("qgd", ("i", "r"), qgd_init)

        vxwd_map = _to_dict(bm.vxwd) if hasattr(bm, "vxwd") else {}
        qxs_init = np.zeros((ni, nr, len(dests)))
        for ii, i in enumerate(comms):
            for ri, r in enumerate(regions):
                for di, dt in enumerate(dests):
                    qxs_init[ii, ri, di] = max(
                        float(vxwd_map.get((i, r, dt), 0.0) or 0.0), _LB
                    )
        _stub("qxs", ("i", "r", "rp"), qxs_init)

        vst_map = _to_dict(bm.vst) if hasattr(bm, "vst") else {}
        qst_init = np.array(
            [
                [max(float(vst_map.get((mg, r), 0.0) or 0.0), _LB) for r in regions]
                for mg in margins
            ]
        )
        _stub("qst", ("marg", "r"), qst_init)

        equations: list[SymbolicEquation] = []

        # ---------------- e_qo (oracle eq_market, monolith 2264) ----------
        class EqQo(SymbolicEquation):
            name: str = "e_qo"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, r = indices
                if float(pyo_value(m.vom[i, r])) <= 1e-8:
                    return None
                uses = sum(m.qfd[i, j, r] for j in m.j)
                uses = uses + m.qpd[i, r] + m.qgd[i, r]
                uses = uses + sum(m.qxs[i, r, dst] for dst in m.rp)
                if i in set(m.marg):
                    uses = uses + m.qst[i, r]
                return m.qo[i, r] * (1.0 + m.to[i, r]) == uses

        equations.append(EqQo())

        # ---------------- e_ps (oracle eq_qo, monolith 1524) --------------
        class EqPs(SymbolicEquation):
            name: str = "e_ps"
            domains: tuple = ("j", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                j, r = indices
                if hasattr(m, "vop"):
                    if float(pyo_value(m.vop[j, r])) <= 1e-8:
                        return None
                elif float(pyo_value(m.vom[j, r])) <= 1e-8:
                    return None
                sva = float(pyo_value(m.share_va[j, r]))
                sigma_top = float(pyo_value(m.esubt[j]))

                if _eps_sigma(sigma_top):
                    int_sum = sum(
                        float(pyo_value(m.share_int[i, j, r])) * m.pf[i, j, r]
                        for i in m.i
                    )
                    return m.ps[j, r] == sva * m.pva[j, r] + int_sum
                sigma_top = _ces_cd_sigma(sigma_top)
                exp = 1.0 - sigma_top
                int_sum = sum(
                    float(pyo_value(m.share_int[i, j, r])) * m.pf[i, j, r] ** exp
                    for i in m.i
                )
                return m.ps[j, r] ** exp == sva * m.pva[j, r] ** exp + int_sum

        equations.append(EqPs())

        # ---------------- e_qf (oracle eq_qf, monolith 1583) --------------
        class EqQf(SymbolicEquation):
            name: str = "e_qf"
            domains: tuple = ("i", "j", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                i, j, r = indices
                sint = float(pyo_value(m.share_int[i, j, r]))
                if sint <= 0.0:
                    return None
                sigma_top = float(pyo_value(m.esubt[j]))
                if _eps_sigma(sigma_top):
                    return m.qf[i, j, r] == sint * m.qo[j, r]
                sigma_top = _ces_cd_sigma(sigma_top)
                return m.qf[i, j, r] == (
                    sint * m.qo[j, r] * (m.ps[j, r] / m.pf[i, j, r]) ** sigma_top
                )

        equations.append(EqQf())

        # ---------------- e_pf (oracle eq_pf_int, monolith 1660) ----------
        class EqPf(SymbolicEquation):
            name: str = "e_pf"
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
                return m.pf[i, j, r] ** expo == (
                    ad * m.pfd[i, j, r] ** expo + ai * m.pfm[i, j, r] ** expo
                )

        equations.append(EqPf())

        # ---------------- e_qva (oracle eq_va, monolith 1565) -------------
        class EqQva(SymbolicEquation):
            name: str = "e_qva"
            domains: tuple = ("j", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                j, r = indices
                sva = float(pyo_value(m.share_va[j, r]))
                if sva <= 0.0:
                    return None
                sigma_top = float(pyo_value(m.esubt[j]))
                if _eps_sigma(sigma_top):
                    return m.qva[j, r] == sva * m.qo[j, r]
                sigma_top = _ces_cd_sigma(sigma_top)
                return m.qva[j, r] == (
                    sva * m.qo[j, r] * (m.ps[j, r] / m.pva[j, r]) ** sigma_top
                )

        equations.append(EqQva())

        # ---------------- e_pva (oracle eq_pva, monolith 1601) ------------
        class EqPva(SymbolicEquation):
            name: str = "e_pva"
            domains: tuple = ("j", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                j, r = indices
                if float(pyo_value(m.va_total[j, r])) <= 1e-8:
                    return None
                sigma_va = _ces_cd_sigma(float(pyo_value(m.esubva[j])))
                exp = 1.0 - sigma_va
                terms = [
                    (f, float(pyo_value(m.share_fac[f, j, r])))
                    for f in m.f
                    if pyo_value(m.share_fac[f, j, r]) > 0.0
                ]
                if not terms:
                    return None
                rhs = sum(sfac * m.pfe[f, j, r] ** exp for f, sfac in terms)
                return m.pva[j, r] ** exp == rhs

        equations.append(EqPva())

        # ---------------- e_qfe (oracle eq_qfe, monolith 1619) ------------
        class EqQfe(SymbolicEquation):
            name: str = "e_qfe"
            domains: tuple = ("f", "j", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                f, j, r = indices
                sfac = float(pyo_value(m.share_fac[f, j, r]))
                if sfac <= 0.0:
                    return None
                sigma_va = _ces_cd_sigma(float(pyo_value(m.esubva[j])))
                return m.qfe[f, j, r] == (
                    sfac * m.qva[j, r] * (m.pva[j, r] / m.pfe[f, j, r]) ** sigma_va
                )

        equations.append(EqQfe())

        # ---------------- e_pfe (oracle eq_pfe, monolith 1643) ------------
        class EqPfe(SymbolicEquation):
            name: str = "e_pfe"
            domains: tuple = ("f", "j", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                f, j, r = indices
                sfac = float(pyo_value(m.share_fac[f, j, r]))
                if sfac <= 0.0:
                    return None
                return m.pfe[f, j, r] == m.pfactor[f, r] * (1.0 + m.tf[f, j, r])

        equations.append(EqPfe())

        return equations
