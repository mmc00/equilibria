"""GTAP PRODUCTION + SUPPLY block (units merged — circular via px/ps/x).

Ports the monolith's production nest and domestic-supply CET/CES equations
VERBATIM from ``gtap_model_equations.py`` (5575-5901):

  PRODUCTION: eq_nd (5590), eq_va (5609), eq_pxeq (5632), eq_pndeq (5682),
              eq_pvaeq (5733), eq_x (5766), eq_po (5790), eq_pp_rai (5821),
              eq_peq (5839)
  SUPPLY:     eq_xs (5868)

The two monolith constraints declared then ``.deactivate()``d in this range
(``prf_y`` 5581, ``eq_ps`` 5897) are NOT ported as active constraints.

ifSUB (global switch): the monolith substitution macros ``_m_pp`` (5497),
``_m_pfa`` (5560) return the PLAIN VAR under ``if_sub=False`` (the comp-stat
oracle form these gates compare against) and the substituted expression under
``if_sub=True``. The Task-4 blocks port the ``if_sub=False`` form — ``_m_pp`` →
``pp_rai``, ``_m_pfa`` → ``pfa`` — and the composer (Task 5) applies the ifSUB
mode by deactivating the 9 defining eqs and fixing their paired vars to the
macro value (monolith post-block 7970-8036). See ``blocks/gtap/__init__.py``.

FIDELITY: equation bodies are the monolith's, transcribed (same Skip guards,
same ``value(model.<param>[...])`` inlining, same ``**omega``/``**expo``,
same ``exp(log(pva))`` CD tautology). Params are built via ``_derived_params``
(a verbatim transcription of the monolith's ``_add_parameters`` recipes) so
every ``model.<param>[...]`` reference and every inlined float matches the
oracle exactly. Elasticity/shift accessors are captured as closures over
``self.params`` exactly as the monolith reads them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyomo.environ import exp, log, value

from equilibria.blocks.base import Block
from equilibria.blocks.gtap import _derived_params as dp  # noqa: F401 (submodule)
from equilibria.blocks.gtap import _ifsub_macros as mac
from equilibria.core.parameters import Parameter
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable

_FLOOR = 1e-8
_REL = 1e-3


def _price_floor(init: float) -> float:
    """Monolith two-pass price floor: max(1e-8, 1e-3*init) for init>0 (5295-5356)."""
    if init is None or init <= 0.0:
        return _FLOOR
    return max(_FLOOR, _REL * float(init))


class ProductionSupplyBlock(Block):
    """GTAP production nest + domestic supply (merged circular unit)."""

    name: str = "GTAP_PRODUCTION_SUPPLY"
    description: str = "GTAP production CES nest + domestic supply CES/CET"
    sets: Any = None
    params: Any = None
    # ifSUB toggle: under if_sub the eq_pp_rai report eq is deactivated by the
    # composer and M_PP (pp_rai = p_rai·(1+prdtx)) is substituted INLINE in the
    # consuming supply/cost eqs (eq_xseq/eq_pxeq), so pp_rai stays coupled to p_rai.
    if_sub: bool = False

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["r", "a", "i"]

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        regions = list(set_manager.get("r"))
        acts = list(set_manager.get("a"))
        comms = list(set_manager.get("i"))
        facs = list(set_manager.get("f"))
        aa = list(set_manager.get("aa"))
        p = self.params
        s = self.sets
        _byname = {"r": regions, "a": acts, "i": comms, "f": facs, "aa": aa}

        # ------------------------------------------------------------------
        # Parameters (monolith _add_parameters recipes) — same names/values so
        # every model.<param>[...] reference and inlined value matches the oracle.
        # ------------------------------------------------------------------
        def _param(name, data, doms, default=0.0, mutable=False):
            parameters[name] = Parameter(
                name=name,
                value=dp.to_array(data, [_byname[d] for d in doms], default),
                domains=tuple(doms),
                mutable=mutable,
            )

        _param("and_param", dp.and_param_data(p, s), ("r", "a"))
        _param("ava_param", dp.ava_param_data(p, s), ("r", "a"))
        _param("nd_share", dp.nd_share_data(p, s), ("r", "a"))
        _param("p_io", dp.p_io_data(p, s), ("r", "i", "a"))
        _param("io_param", dp.io_param_data(p, s), ("r", "i", "a"))
        _param("lambdaio", dp.lambdaio_data(p, s), ("r", "i", "a"), default=1.0)
        _param("af_param", dp.af_param_data(p, s), ("r", "f", "a"))
        _param("gx_param", dp.gx_param_data(p, s), ("r", "a", "i"))
        _param("xscale", dp.xscale_data(p, s), ("r", "aa"), default=1.0)
        _param("p_ax", dp.p_ax_data(p, s), ("r", "a", "i"))
        _param("prdtx_rai", dp.prdtx_rai_data(p, s), ("r", "a", "i"))
        _param("xflag", dp.xflag_data(p, s), ("r", "a", "i"))
        # gd_share/ge_share shared with TRADE_CET (dedup by name — first wins);
        # declared here too so a standalone build resolves them. CARRY: drift.
        # mutable=True: the monolith (2001-2005) declares these mutable and references
        # them unwrapped, so they stay symbolic (not folded to a literal) — matches
        # the oracle under finite-omegax and survives the composer's share recompute.
        _param("gd_share", dp.gd_share_data(p, s), ("r", "i"), mutable=True)
        _param("ge_share", dp.ge_share_data(p, s), ("r", "i"), mutable=True)

        # ------------------------------------------------------------------
        # Variables OWNED by this unit (monolith 3509-3579). Prices carry the
        # runtime relative floor; quantities stay NonNegativeReals (0, None).
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

        na, ni, nr = len(acts), len(comms), len(regions)
        bm = p.benchmark
        # xp init = get_vom_init (benchmark VA+ND); x = makb; xs/xds benchmark.
        # Quantity inits do not affect the form/domain gates (bounds are (0,None));
        # a faithful benchmark seed is used (deep fallback chains omitted — they
        # only refine INIT LEVELS, which Task-5's composer re-settles).
        xp_init = np.array(
            [[max(self._vom_init(r, a), _FLOOR) for a in acts] for r in regions]
        )
        # x/xs use the monolith get_make_init (make-CET allocation for finite
        # omegas), NOT raw makb — see _make_init. xds stays the makb-based domestic
        # supply (the scaling re-settles it); xs = Σ_a make.
        x_init = np.array(
            [
                [[max(self._make_init(r, a, i), 0.0) for i in comms] for a in acts]
                for r in regions
            ]
        )
        xs_init = np.array(
            [
                [
                    max(sum(self._make_init(r, a, i) for a in acts), _FLOOR)
                    for i in comms
                ]
                for r in regions
            ]
        )
        xds_init = np.array(
            [
                [
                    max(
                        sum(float(bm.makb.get((r, a, i), 0.0) or 0.0) for a in acts),
                        _FLOOR,
                    )
                    for i in comms
                ]
                for r in regions
            ]
        )
        p_rai_init = np.array(
            [
                [[self._p_rai_init(r, a, i) for i in comms] for a in acts]
                for r in regions
            ]
        )
        pp_rai_init = np.array(
            [
                [
                    [
                        max(
                            (1.0 + self._prdtx(r, a, i)) * self._p_rai_init(r, a, i),
                            _FLOOR,
                        )
                        for i in comms
                    ]
                    for a in acts
                ]
                for r in regions
            ]
        )
        ones_ra = np.ones((nr, na))
        ones_ri = np.ones((nr, ni))

        _q("xp", ("r", "a"), xp_init)
        _q("x", ("r", "a", "i"), x_init)
        _price("px", ("r", "a"), ones_ra)
        _price("p_rai", ("r", "a", "i"), p_rai_init)
        _price("pp", ("r", "a"), ones_ra)
        _price("pp_rai", ("r", "a", "i"), pp_rai_init)
        _q("xs", ("r", "i"), xs_init)
        _q("xds", ("r", "i"), xds_init)
        _price("ps", ("r", "i"), ones_ri)
        _price("pd", ("r", "i"), ones_ri)
        # va/nd/pva/pnd (production nest bundles) — monolith 3758-3794.
        _q("va", ("r", "a"), ones_ra)
        _q("nd", ("r", "a"), ones_ra)
        pva0 = getattr(p.calibrated, "pva_bench", {})
        pnd0 = getattr(p.calibrated, "pnd_bench", {})
        pva_init = np.array(
            [[float(pva0.get((r, a), 1.0)) for a in acts] for r in regions]
        )
        pnd_init = np.array(
            [[float(pnd0.get((r, a), 1.0)) for a in acts] for r in regions]
        )
        _price("pva", ("r", "a"), pva_init)
        _price("pnd", ("r", "a"), pnd_init)

        # closures for the inline-python accessors (read exactly as the monolith)
        el = p.elasticities
        sh = p.shifts

        def _get_sigmap(r, a):
            return el.sigmap.get((r, a), 1.0)

        def _get_sigmand(r, a):
            return el.sigmand.get((r, a), 1.0)

        def _get_sigmav(r, a):
            return el.sigmav.get((r, a), 1.0)

        def _get_omegas(r, a):
            return el.omegas.get((r, a), 1.0)

        def _get_sigmas(r, i):
            return el.sigmas.get((r, i), 2.0)

        def _axp_shift(r, a):
            return sh.axp.get((r, a), 1.0)

        def _lambdand(r, a):
            return sh.lambdand.get((r, a), 1.0)

        def _lambdava(r, a):
            return sh.lambdava.get((r, a), 1.0)

        def _lambdaf(r, f, a):
            return sh.lambdaf.get((r, f, a), 1.0)

        acts_out = s.activity_commodities
        comm_acts = s.commodity_activities
        makb = bm.makb
        shifts = sh
        pshifts_axp = sh.axp
        pshifts_lambdaio = sh.lambdaio
        if_sub = self.if_sub

        def _m_pp(model, r, a, i):
            """GAMS $macro M_PP (8013): p_rai·(1+prdtx_rai) inline under ifSUB (so
            pp_rai stays coupled to p_rai when eq_pp_rai is deactivated), else the
            plain report var pp_rai. Mirrors this block's own eq_pp_rai body."""
            if if_sub:
                return (1.0 + model.prdtx_rai[r, a, i]) * model.p_rai[r, a, i]
            return model.pp_rai[r, a, i]

        equations: list[SymbolicEquation] = []

        # ---------------- eq_nd (monolith 5590) ----------------
        class EqNd(SymbolicEquation):
            name: str = "eq_nd"
            domains: tuple = ("r", "a")

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                r, a = indices
                and_val = value(m.and_param[r, a])
                if and_val <= 0.0:
                    return None
                sigmap = _get_sigmap(r, a)
                px = m.px[r, a]
                pnd = m.pnd[r, a]
                if value(pnd) <= 0:
                    return None
                ratio = px / pnd
                shift = _axp_shift(r, a) * _lambdand(r, a)
                return m.nd[r, a] == and_val * m.xp[r, a] * ratio**sigmap * shift ** (
                    sigmap - 1
                )

        equations.append(EqNd())

        # ---------------- eq_va (monolith 5609) ----------------
        class EqVa(SymbolicEquation):
            name: str = "eq_va"
            domains: tuple = ("r", "a")

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                r, a = indices
                ava_val = value(m.ava_param[r, a])
                if ava_val <= 0.0:
                    return None
                sigmap = _get_sigmap(r, a)
                px = m.px[r, a]
                pva = m.pva[r, a]
                if value(pva) <= 0:
                    return None
                ratio = px / pva
                shift = _axp_shift(r, a) * _lambdava(r, a)
                return m.va[r, a] == ava_val * m.xp[r, a] * ratio**sigmap * shift ** (
                    sigmap - 1
                )

        equations.append(EqVa())

        # ---------------- eq_pxeq (monolith 5632) ----------------
        class EqPxeq(SymbolicEquation):
            name: str = "eq_pxeq"
            domains: tuple = ("r", "a")

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                r, a = indices
                and_val = value(m.and_param[r, a])
                ava_val = value(m.ava_param[r, a])
                if not shifts.axp and not shifts.lambdand and not shifts.lambdava:
                    and_val = value(m.nd_share[r, a])
                    ava_val = 1.0 - and_val
                if and_val <= 0.0 and ava_val <= 0.0:
                    return None
                sigmap = _get_sigmap(r, a)
                expo = 1.0 - sigmap
                if abs(expo) < 1e-8:
                    nd_term = (
                        (m.pnd[r, a] / max(_lambdand(r, a), 1e-8)) ** and_val
                        if and_val > 0.0
                        else 1.0
                    )
                    va_term = (
                        (m.pva[r, a] / max(_lambdava(r, a), 1e-8)) ** ava_val
                        if ava_val > 0.0
                        else 1.0
                    )
                    return m.px[r, a] == nd_term * va_term
                shift = _axp_shift(r, a) ** (sigmap - 1.0)
                lambdand = max(_lambdand(r, a), 1e-8)
                lambdava = max(_lambdava(r, a), 1e-8)
                term_nd = (
                    and_val * (m.pnd[r, a] / lambdand) ** expo if and_val > 0.0 else 0.0
                )
                term_va = (
                    ava_val * (m.pva[r, a] / lambdava) ** expo if ava_val > 0.0 else 0.0
                )
                return m.px[r, a] ** expo == shift * (term_nd + term_va)

        equations.append(EqPxeq())

        # ---------------- eq_pndeq (monolith 5682) ----------------
        class EqPndeq(SymbolicEquation):
            name: str = "eq_pndeq"
            domains: tuple = ("r", "a")

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                r, a = indices
                sigmand = _get_sigmand(r, a)
                expo = 1.0 - sigmand
                if abs(expo) < 1e-8:
                    terms = []
                    for i in m.i:
                        io_val = (
                            value(m.io_param[r, i, a])
                            if hasattr(m, "io_param")
                            else value(m.p_io[r, i, a])
                        )
                        if not pshifts_lambdaio:
                            io_val = value(m.p_io[r, i, a])
                        if io_val <= 0.0:
                            continue
                        lambdaio = max(value(m.lambdaio[r, i, a]), 1e-8)
                        terms.append((m.pa[r, i, a] / lambdaio) ** io_val)
                    if not terms:
                        return None
                    prod = 1.0
                    for term in terms:
                        prod *= term
                    return m.pnd[r, a] == prod
                terms = []
                for i in m.i:
                    io_val = (
                        value(m.io_param[r, i, a])
                        if hasattr(m, "io_param")
                        else value(m.p_io[r, i, a])
                    )
                    if not pshifts_lambdaio:
                        io_val = value(m.p_io[r, i, a])
                    if io_val <= 0.0:
                        continue
                    lambdaio = max(value(m.lambdaio[r, i, a]), 1e-8)
                    terms.append(io_val * (m.pa[r, i, a] / lambdaio) ** expo)
                if not terms:
                    return None
                return m.pnd[r, a] ** expo == sum(terms)

        equations.append(EqPndeq())

        # ---------------- eq_pvaeq (monolith 5733) ----------------
        class EqPvaeq(SymbolicEquation):
            name: str = "eq_pvaeq"
            domains: tuple = ("r", "a")

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                r, a = indices
                sigmav = _get_sigmav(r, a)
                expo = 1.0 - sigmav
                if abs(expo) < 1e-8:
                    return exp(log(m.pva[r, a])) == m.pva[r, a]
                terms = []
                for f in m.f:
                    af_val = (
                        value(m.af_param[r, f, a])
                        if hasattr(m, "af_param")
                        else value(m.af_share[r, f, a])
                    )
                    if af_val <= 0.0:
                        continue
                    lambdaf = max(_lambdaf(r, f, a), 1e-8)
                    # M_PFA inline under if_sub=True (pfa coupled to pf; eq_pfaeq off).
                    pfa_t = mac.m_pfa(m, p, r, f, a) if if_sub else m.pfa[r, f, a]
                    terms.append(af_val * (pfa_t / lambdaf) ** expo)
                if not terms:
                    return None
                return m.pva[r, a] ** expo == sum(terms)

        equations.append(EqPvaeq())

        # ---------------- eq_x (monolith 5766) ----------------
        class EqX(SymbolicEquation):
            name: str = "eq_x"
            domains: tuple = ("r", "a", "i")

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                r, a, i = indices
                outputs = acts_out.get(a, [])
                if outputs and i not in outputs:
                    return None
                if value(m.xflag[r, a, i]) <= 0.0:
                    return None
                share = value(m.gx_param[r, a, i])
                make_base = makb.get((r, a, i), 0.0)
                if share <= 0.0 and make_base <= 0.0:
                    return None
                omega = _get_omegas(r, a)
                if omega == float("inf"):
                    return m.p_rai[r, a, i] == m.px[r, a]
                return (
                    m.x[r, a, i]
                    == share
                    * (m.xp[r, a] / m.xscale[r, a])
                    * (m.p_rai[r, a, i] / m.px[r, a]) ** omega
                )

        equations.append(EqX())

        # ---------------- eq_po (monolith 5790) ----------------
        class EqPo(SymbolicEquation):
            name: str = "eq_po"
            domains: tuple = ("r", "a")

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                r, a = indices
                outputs = acts_out.get(a)
                if not outputs:
                    return None
                active_outputs = [
                    i
                    for i in outputs
                    if value(m.gx_param[r, a, i]) > 0.0
                    or makb.get((r, a, i), 0.0) > 0.0
                ]
                if not active_outputs:
                    return None
                omega = _get_omegas(r, a)
                if omega == float("inf"):
                    return (m.xp[r, a] / m.xscale[r, a]) == sum(
                        m.x[r, a, i] for i in active_outputs
                    )
                exponent = 1.0 + omega
                return m.px[r, a] ** exponent == sum(
                    value(m.gx_param[r, a, i]) * m.p_rai[r, a, i] ** exponent
                    for i in active_outputs
                )

        equations.append(EqPo())

        # ---------------- eq_pp_rai (monolith 5821) ----------------
        class EqPpRai(SymbolicEquation):
            name: str = "eq_pp_rai"
            domains: tuple = ("r", "a", "i")

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                r, a, i = indices
                outputs = acts_out.get(a, [])
                if outputs and i not in outputs:
                    return None
                if value(m.xflag[r, a, i]) <= 0.0:
                    return None
                share = value(m.gx_param[r, a, i])
                make_base = makb.get((r, a, i), 0.0)
                if share <= 0.0 and make_base <= 0.0:
                    return None
                return (
                    m.pp_rai[r, a, i]
                    == (1.0 + value(m.prdtx_rai[r, a, i])) * m.p_rai[r, a, i]
                )

        equations.append(EqPpRai())

        # ---------------- eq_peq (monolith 5839) ----------------
        class EqPeq(SymbolicEquation):
            name: str = "eq_peq"
            domains: tuple = ("r", "a", "i")

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                r, a, i = indices
                outputs = acts_out.get(a, [])
                if outputs and i not in outputs:
                    return None
                if value(m.xflag[r, a, i]) <= 0.0:
                    return None
                share = value(m.gx_param[r, a, i])
                make_base = makb.get((r, a, i), 0.0)
                ax_val = value(m.p_ax[r, a, i])
                if share <= 0.0 and make_base <= 0.0 and ax_val <= 0.0:
                    return None
                sigma = _get_sigmas(r, i)
                # M_PP inline under if_sub=True (pp_rai coupled to p_rai here).
                if sigma == float("inf"):
                    return _m_pp(m, r, a, i) == m.ps[r, i]
                return (
                    m.x[r, a, i]
                    == ax_val * m.xs[r, i] * (m.ps[r, i] / _m_pp(m, r, a, i)) ** sigma
                )

        equations.append(EqPeq())

        # ---------------- eq_xs (monolith 5868) ----------------
        class EqXs(SymbolicEquation):
            name: str = "eq_xs"
            domains: tuple = ("r", "i")

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                r, i = indices
                producing_activities = comm_acts.get(i, list(m.a))
                active_activities = [
                    a
                    for a in producing_activities
                    if value(m.gx_param[r, a, i]) > 0.0
                    or makb.get((r, a, i), 0.0) > 0.0
                ]
                if not active_activities:
                    return None
                sigma = _get_sigmas(r, i)
                if sigma == float("inf"):
                    return m.xs[r, i] == sum(m.x[r, a, i] for a in active_activities)
                exponent = 1.0 - sigma
                if abs(exponent) < 1e-8:
                    return None
                # M_PP inline under if_sub=True (pp_rai coupled to p_rai here).
                return m.ps[r, i] ** exponent == sum(
                    value(m.p_ax[r, a, i]) * _m_pp(m, r, a, i) ** exponent
                    for a in active_activities
                )

        equations.append(EqXs())

        return equations

    # ------------------------------------------------------------------
    # init helpers (monolith get_*_init logic — the parts that set price floors)
    # ------------------------------------------------------------------
    def _vom_init(self, r: str, a: str) -> float:
        """xp init = get_vom_init: ND (vdfp+vmfp) + VA (evfb) (monolith 2799-2830)."""
        bm = self.params.benchmark
        nd = sum(
            float(bm.vdfp.get((r, i, a), 0.0) or 0.0)
            + float(bm.vmfp.get((r, i, a), 0.0) or 0.0)
            for i in self.sets.i
        )
        va = sum(
            float(bm.evfb.get((r, f, a), bm.vfm.get((r, f, a), 0.0)) or 0.0)
            for f in self.sets.f
        )
        total = nd + va
        if total <= 0.0:
            total = float(bm.vom.get((r, a), 0.0) or 0.0)
        return total

    def _prdtx(self, r: str, a: str, i: str) -> float:
        """prdtx per commodity = makb/maks - 1 (else rto) (monolith 3487-3495)."""
        bm = self.params.benchmark
        makb_val = float(bm.makb.get((r, a, i), 0.0) or 0.0)
        maks_val = float(bm.maks.get((r, a, i), 0.0) or 0.0)
        if maks_val > 0 and makb_val > 0:
            return makb_val / maks_val - 1.0
        return float(self.params.taxes.rto.get((r, a), 0.0) or 0.0)

    def _p_rai_init(self, r: str, a: str, i: str) -> float:
        """p_rai init = 1/(1+prdtx), floored 1e-8 (monolith get_p_rai_init 3486)."""
        prdtx = self._prdtx(r, a, i)
        return max(1.0 / max(1.0 + prdtx, 1e-12), _FLOOR)

    def _make_init(self, r: str, a: str, i: str) -> float:
        """Output by activity-commodity = monolith get_make_init (3147-3174).

        omega=inf → raw makb; FINITE omegas → the make-CET allocation
        gx·xp·(p_rai)^omega (px=1 at benchmark). The raw-makb shortcut was correct
        only for omega=inf commodities; for finite omegas (e.g. Food omegas=5) it
        overstated xs, drifting the post-scaling gd_share/ge_share recompute and the
        ifSUB CET quantities (found via the xmodel-parity cascade). Restrict to the
        activity's output commodities."""
        p = self.params
        outputs = self.sets.activity_commodities.get(a, [])
        if outputs and i not in outputs:
            return 0.0
        gx_val = float(p.calibrated.gx_param.get((r, a, i), 0.0) or 0.0)
        if gx_val <= 0.0:
            return 0.0
        omega = p.elasticities.omegas.get((r, a), float("inf"))
        makb = float(p.benchmark.makb.get((r, a, i), 0.0) or 0.0)
        if omega == float("inf"):
            return (
                max(makb, 1e-8)
                if makb > 0
                else max(gx_val * self._vom_init(r, a), 1e-8)
            )
        xp_phys = self._vom_init(r, a)
        p_rai_v = self._p_rai_init(r, a, i)
        return max(gx_val * xp_phys * (p_rai_v**omega), 1e-8)
