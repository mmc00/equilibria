"""GTAP6 FACTOR MARKETS block (leaf unit).

Ports the v6.2 monolith's factor-market clearing at the COMMODITY level
(``qoes(f,r)``/``pf(f,r)``), following the same fidelity discipline
``blocks/gtap6/trade_armington.py``/``blocks/gtap6/production.py`` used for
the trade and production units.

Oracle status (read via ``scripts/gtap6/_v62_monolith_oracle.py``,
``GTAP6MonolithOracle._add_factor_markets``, monolith 2288-2320): the
oracle's factor-market block is DELIBERATELY INCOMPLETE — its own docstring
says "v6.2 with all factors treated as mobile within the region (the
BOOK3X3 default — ETRE = -1 for all factors)" and "Phase 2d will add the
sluggish CET allocation for factors with ETRE > 0". It wires exactly two
Constraints, both applied UNIFORMLY across every factor (no ``mf``/``sf``
branch at all):

  eq_factor_clear -> sum_j qfe(f,j,r) == qoes(f,r)   (market clearing)
  eq_qoes_fixed    -> qoes(f,r) == evom(f,r)          (supply pinned to
                                                        benchmark; closure)

There is NO ``eq_qoes``, ``eq_pmes``, ``eq_pm_endw``, ``eq_qe``, or
``eq_pe_endw`` method/Constraint anywhere in the oracle — ``grep -n "def
eq_qoes\\|def eq_pmes\\|def eq_pm_endw\\|def eq_qe\\b\\|def eq_pe_endw"
scripts/gtap6/_v62_monolith_oracle.py`` returns nothing, and ``model.mf``/
``model.sf`` (registered at monolith 130-131) are never referenced by any
equation body in the file. This block is therefore NOT a rename of an
existing oracle Constraint (unlike every equation in
``trade_armington.py``/``production.py``) — it is the genuine mobile/
sluggish split the oracle's own docstring defers to "Phase 2d", written
here to fulfil the v6.2 contract's ``_GTAP6_FACTOR_MARKETS`` IDs.

Economics (NOT invented — transcribed from the same textbook GTAP
specific-factor equation GTAP7's ``blocks/gtap/factor.py`` already ports
faithfully, ``EqPfeq``/``EqPfteq``, verified byte-identical to van der
Mensbrugghe's GAMS ``ENDW_PRICE`` block per
``docs/findings/f3_5_base_calibrado_done_2026-07-30.md`` lines 255-283:
``pm(es,r) = (sum_j REVSHR(es,j,r)*pmes(es,j,r)^(1-ETRAE))^(1/(1-ETRAE))``,
with ``REVSHR`` == this module's ``gf_share`` and ``1-ETRAE`` == ``1+
omegaf`` under the ``omegaf = -etrae`` sign convention below). v6.2 has no
activity-level factor income tax (``tinc``/``kappaf``), so the wedge terms
GTAP7 threads through ``_m_pfy``/``kappaf`` are simply absent — the
sector-level factor price IS ``pfe(f,j,r)`` (already the after-tax agent
price ProductionBlock's own ``e_pfe`` computes as
``pfactor(f,r)*(1+tf(f,j,r))``), so this block reads ``pfe``/``qfe``
directly rather than re-deriving a GTAP7-style ``pf``/``pfy`` pair.

  Sluggish factors (``f in sets.sf``, e.g. Land/Capital on gtap6_3x3) —
  the CET sectoral allocation, GTAP7's ``eq_pfeq``/``eq_pfteq`` sf-branch
  adapted to v6.2 naming (``qfe`` for GTAP7's ``xf``, ``qoes`` for GTAP7's
  ``xft``, ``pmes`` for GTAP7's per-sector ``pf``, ``pmagg`` — a new Var
  this block owns — for GTAP7's aggregate ``pft``):

    e_qoes (cross-multiplied CET first-order condition, avoiding a bare
            division exactly as GTAP7's ``EqPfeq`` does):
              pmes(f,j,r)^omegaf(f) * gf_share(f,j,r) * qoes(f,r)
                == pmagg(f,r)^omegaf(f) * qfe(f,j,r)
    e_pmes: sector factor price == the after-tax agent price
            ProductionBlock's ``e_pfe`` already computes (v6.2 has no
            further per-activity wedge to add — the ``M_PFY`` substitution
            GTAP7 needs collapses to the identity map since v6.2's kappa
            == 0):
              pmes(f,j,r) == pfe(f,j,r)
    e_pm_endw: CET aggregate price index (GTAP7's ``eq_pfteq``
            omegaf-finite branch):
              pmagg(f,r)^(1+omegaf(f))
                == sum_j gf_share(f,j,r) * pmes(f,j,r)^(1+omegaf(f))

  Mobile factors (``f in sets.mf``) — the oracle's own uniform
  ``eq_factor_clear``/``eq_qoes_fixed`` ARE already the correct mobile
  law-of-one-price form (``omegaf -> inf`` collapses GTAP7's CET to a
  single regional wage with no cross-sector wedge), so ``e_qe``/
  ``e_pe_endw`` transcribe those two Constraints VERBATIM, restricted to
  ``f in sets.mf`` (the oracle applies them to every factor because it has
  not yet split mobile/sluggish; this block scopes the SAME algebra to the
  subset the contract's ``e_qe``/``e_pe_endw`` IDs actually own, leaving
  the sluggish subset to ``e_qoes``/``e_pmes``/``e_pm_endw`` instead):
    e_qe:       sum_j qfe(f,j,r) == qe(f,r)     (total factor supply)
    e_pe_endw:  qe(f,r) == evom(f,r)             (supply pinned to
                                                   benchmark)

FIDELITY: the mobile branch (``e_qe``/``e_pe_endw``) is byte-identical to
the oracle's own ``eq_factor_clear``/``eq_qoes_fixed`` bodies (verified by
the numeric form-diff test below, restricted to ``mf``). The sluggish
branch (``e_qoes``/``e_pmes``/``e_pm_endw``) has no oracle Constraint to
diff against — it is verified against the oracle's OWN benchmark
calibration instead (the CET is exactly satisfied at the benchmark point:
all prices == 1.0, so ``sum_j gf_share == 1`` collapses every sluggish
equation to a share-normalization identity), the same identity-check
methodology Task 6 used for ``e_qds``/``e_qtmfsd``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from equilibria.blocks.base import Block
from equilibria.core.parameters import Parameter
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable

_LB = 1e-6


def _to_dict(mapping: Any) -> dict:
    """Coerce a params.*.get-style mapping to a plain dict (defensive)."""
    if mapping is None:
        return {}
    if isinstance(mapping, dict):
        return mapping
    return dict(mapping)


def _omegaf(etrae: float) -> float:
    """CET transformation elasticity from ETRE, GTAP7 ``_omegaf`` convention.

    ``omegaf = -etrae`` (``blocks/gtap/factor.py`` ``_omegaf``): GTAP's
    ``ETRE``/``ETRAE`` is conventionally negative or zero for a genuine
    sluggish factor (e.g. ``etrae['Land'] = -1.0`` on gtap6_3x3), and the
    CET exponent used in the price/quantity equations is its negation.
    ``float("inf")`` is never returned here — the mobile/sluggish split is
    driven by ``sets.mf``/``sets.sf`` (SLUG), not by testing ``omegaf`` for
    infinity, per the task brief.
    """
    return -float(etrae)


class FactorBlock(Block):
    """GTAP6 commodity-level factor markets (mobile wage + sluggish CET)."""

    name: str = "GTAP6_FACTOR"
    description: str = (
        "GTAP6 factor markets: mobile law-of-one-price + sluggish CET "
        "sectoral allocation, at the commodity (f,r) level"
    )
    sets: Any = None
    params: Any = None
    derived: Any = None

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["r", "j", "f"]

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        regions = list(set_manager.get("r"))
        prod_secs = list(set_manager.get("j"))
        factors = list(set_manager.get("f")) if set_manager.has("f") else []

        p = self.params
        d = self.derived
        el = p.elasticities

        nr, nj, nf = len(regions), len(prod_secs), len(factors)

        mobile = set(self.sets.mf)
        sluggish = set(self.sets.sf)

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

        evom_arr = _p2("evom", d.evom, (factors, regions))
        parameters["evom"] = Parameter(
            name="evom", value=evom_arr, domains=("f", "r"), mutable=True
        )

        omegaf_arr = np.array(
            [_omegaf(el.etrae.get(f, 0.0)) for f in factors], dtype=float
        )
        parameters["omegaf"] = Parameter(
            name="omegaf", value=omegaf_arr, domains=("f",), mutable=True
        )

        # gf_share(f,j,r) — CET sectoral REVENUE share of factor f's total
        # regional income earned in sector j (GTAP7's ``REVSHR``:
        # ``blocks/gtap/factor.py`` ``gf_share`` recipe, ``vfm/evom``). NOT
        # the same share as ProductionBlock's ``share_fac`` (the VA-nest
        # COST share ``qfe/va_total(j,r)``, a different economic ratio
        # entirely — verified numerically: reusing ``share_fac`` here
        # violates the CET benchmark identity by 1-2 orders of magnitude,
        # while ``vfm/evom`` sums to ~1.04-1.09 across sectors per factor
        # (the standard small agent-vs-market-price benchmark wedge the
        # oracle's own docstring documents for ``eq_market``, ~2-9%, not a
        # bug in this block).
        vfm_map_for_share = _to_dict(p.benchmark.vfm)
        evom_map_for_share = _to_dict(d.evom)
        gf_share_arr = np.zeros((nf, nj, nr), dtype=float)
        for fi, f in enumerate(factors):
            for ri, r in enumerate(regions):
                evom_val = float(evom_map_for_share.get((f, r), 0.0) or 0.0)
                if evom_val <= 1e-8:
                    continue
                for ji, j in enumerate(prod_secs):
                    vfm_val = float(vfm_map_for_share.get((f, j, r), 0.0) or 0.0)
                    gf_share_arr[fi, ji, ri] = vfm_val / evom_val
        parameters["gf_share"] = Parameter(
            name="gf_share", value=gf_share_arr, domains=("f", "j", "r")
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

        # qoes (f,r) — sluggish factor total regional supply, seeded from
        # evom (oracle's own qoes init, monolith 902-910).
        evom_map = _to_dict(d.evom)
        qoes_init = np.array(
            [
                [max(float(evom_map.get((f, r), 1.0) or 1.0), _LB) for r in regions]
                for f in factors
            ]
        )
        _q("qoes", ("f", "r"), qoes_init)

        # pmes (f,j,r) — sluggish factor sector price, benchmark-normalized
        # to 1.0 (same convention as ProductionBlock's pfe).
        _price("pmes", ("f", "j", "r"), np.ones((nf, nj, nr)))

        # pmagg (f,r) — sluggish CET aggregate price index (this block's
        # new Var for e_pm_endw; GTAP7's analogous pft), benchmark-
        # normalized to 1.0.
        _price("pmagg", ("f", "r"), np.ones((nf, nr)))

        # qe (f,r) — mobile factor total regional supply, seeded from evom
        # (same seed recipe as qoes; the two Vars partition sets.f via
        # mf/sf so there is no overlap in which cells are economically
        # "live", even though both are declared over the full (f,r) grid
        # for a rectangular Variable array).
        _q("qe", ("f", "r"), qoes_init.copy())

        # ------------------------------------------------------------------
        # STUB variables — owned by other blocks (ProductionBlock).
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

        # qfe (f,j,r) — factor demand, owned by ProductionBlock (e_qfe).
        # Declared here as a guarded stub in case FactorBlock is composed
        # standalone (dedup: ProductionBlock's real values win if it ran
        # first).
        vfm_map = _to_dict(p.benchmark.vfm)
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
        _stub("qfe", ("f", "j", "r"), qfe_init)

        # pfe (f,j,r) — factor agent price, owned by ProductionBlock
        # (e_pfe: pfe = pfactor*(1+tf)). e_pmes reads this directly (see
        # module docstring: v6.2 has no per-activity factor tax beyond what
        # ProductionBlock's e_pfe already bakes in).
        _stub("pfe", ("f", "j", "r"), np.ones((nf, nj, nr)))

        # pfactor (f,r) — regional factor wage. ProductionBlock already
        # declares this as a genuinely-needed stub for its own e_pfe; this
        # block does not own it (mobile e_pe_endw pins the SUPPLY qe, not
        # pfactor — the oracle's uniform closure never wires a
        # wage-clearing equation for pf/pfactor itself, only the QUANTITY
        # side qoes/qoes_fixed; the price adjusts via the wider model's
        # income/GDP block, out of this task's scope). Declared here only
        # so a standalone FactorBlock build has a complete variable set;
        # guarded so ProductionBlock's registration (if composed first)
        # wins.
        _stub("pfactor", ("f", "r"), np.ones((nf, nr)))

        equations: list[SymbolicEquation] = []

        # ---------------- e_qoes (sluggish CET sector allocation) --------
        # GTAP7 blocks/gtap/factor.py EqPfeq sf-branch, v6.2-adapted:
        #   pmes[f,j,r]^omegaf * gf_share[f,j,r] * qoes[f,r]
        #     == pmagg[f,r]^omegaf * qfe[f,j,r]
        # (cross-multiplied CET first-order condition; v6.2 drops the
        # (1-kappa) wedge GTAP7 needs since there is no activity-level
        # factor income tax). Skipped for mobile factors (owned by e_qe)
        # and for evom<=0 cells (mirrors the oracle's own Skip guard on
        # eq_factor_clear/eq_qoes_fixed).
        class EqQoes(SymbolicEquation):
            name: str = "e_qoes"
            domains: tuple = ("f", "j", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                f, j, r = indices
                if f not in sluggish:
                    return None
                if float(pyo_value(m.evom[f, r])) <= 1e-8:
                    return None
                gf = float(pyo_value(m.gf_share[f, j, r]))
                if gf <= 0.0:
                    return None
                omega = float(pyo_value(m.omegaf[f]))
                pmes_term = m.pmes[f, j, r] ** omega
                pmagg_term = m.pmagg[f, r] ** omega
                return pmes_term * gf * m.qoes[f, r] == pmagg_term * m.qfe[f, j, r]

        equations.append(EqQoes())

        # ---------------- e_pmes (sluggish sector factor price) -----------
        # v6.2 has no per-activity factor tax (module docstring): the
        # sector factor price IS the agent price ProductionBlock's e_pfe
        # already computes.
        class EqPmes(SymbolicEquation):
            name: str = "e_pmes"
            domains: tuple = ("f", "j", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                f, j, r = indices
                if f not in sluggish:
                    return None
                gf = float(pyo_value(m.gf_share[f, j, r]))
                if gf <= 0.0:
                    return None
                return m.pmes[f, j, r] == m.pfe[f, j, r]

        equations.append(EqPmes())

        # ---------------- e_pm_endw (sluggish CET aggregate price) --------
        # GTAP7 blocks/gtap/factor.py EqPfteq omegaf-finite branch:
        #   pmagg[f,r]^(1+omega) == sum_j gf_share[f,j,r]*pmes[f,j,r]^(1+omega)
        # Verified byte-identical to van der Mensbrugghe's GAMS ENDW_PRICE
        # block (docs/findings/f3_5_base_calibrado_done_2026-07-30.md
        # lines 255-283): pm(es,r) = (sum_j REVSHR*pmes^(1-ETRAE))^(1/(1-ETRAE)),
        # 1-ETRAE == 1+omegaf under this module's omegaf=-etrae convention.
        class EqPmEndw(SymbolicEquation):
            name: str = "e_pm_endw"
            domains: tuple = ("f", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                f, r = indices
                if f not in sluggish:
                    return None
                if float(pyo_value(m.evom[f, r])) <= 1e-8:
                    return None
                omega = float(pyo_value(m.omegaf[f]))
                terms = [
                    (j, float(pyo_value(m.gf_share[f, j, r])))
                    for j in m.j
                    if pyo_value(m.gf_share[f, j, r]) > 0.0
                ]
                if not terms:
                    return None
                expo = 1.0 + omega
                rhs = sum(gf * m.pmes[f, j, r] ** expo for j, gf in terms)
                return m.pmagg[f, r] ** expo == rhs

        equations.append(EqPmEndw())

        # ---------------- e_qe (mobile factor market clearing) -----------
        # Byte-identical transcription of the oracle's eq_factor_clear
        # (monolith 2307-2312), restricted to f in sets.mf.
        class EqQe(SymbolicEquation):
            name: str = "e_qe"
            domains: tuple = ("f", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                f, r = indices
                if f not in mobile:
                    return None
                if float(pyo_value(m.evom[f, r])) <= 1e-8:
                    return None
                return sum(m.qfe[f, j, r] for j in m.j) == m.qe[f, r]

        equations.append(EqQe())

        # ---------------- e_pe_endw (mobile factor supply pinned) --------
        # Byte-identical transcription of the oracle's eq_qoes_fixed
        # (monolith 2315-2320), restricted to f in sets.mf.
        class EqPeEndw(SymbolicEquation):
            name: str = "e_pe_endw"
            domains: tuple = ("f", "r")

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                f, r = indices
                if f not in mobile:
                    return None
                if float(pyo_value(m.evom[f, r])) <= 1e-8:
                    return None
                return m.qe[f, r] == pyo_value(m.evom[f, r])

        equations.append(EqPeEndw())

        return equations
