"""GTAP6 INCOME_CLOSURE block (closure unit — last in ``GTAP6_BLOCK_ORDER``).

Ports the v6.2 monolith's regional-income, savings, tax-revenue, numeraire
and GDP identities, following the same fidelity discipline
``blocks/gtap6/trade_armington.py``/``blocks/gtap6/production.py``/
``blocks/gtap6/factor.py``/``blocks/gtap6/demand_utility.py`` used for the
other units.

This block owns all 12 IDs in ``_GTAP6_INCOME_AND_CLOSURE`` (``e_y,
e_ysav, e_psave, e_rorg, e_kb, e_ke, e_walras, e_pgdpwld, e_taxrev,
e_gdpmp, e_rgdpmp, e_pgdpmp``) PLUS ``e_yp``/``e_yg`` reserved from
``_GTAP6_FINAL_DEMAND`` per the controller's ruling documented in
``demand_utility.py``'s own module docstring — 14 equations total.

***THE SAV-AS-VAR FIX (Phase 3.38) — READ THIS BEFORE TOUCHING ``sav``***

``docs/findings/gtap_v62_phase338_sav_var_budget_identity.md`` (fetched
from ``gtap/v62-multiperiod`` and read in full before writing this file)
documents that the ORIGINAL orphan branch held regional savings ``sav``
as a constant ``save_0`` Param through Phase 3.36/3.37, leaving the
regional budget identity ``y = yp + yg + sav`` unsatisfied under any
shock — the imbalance leaked into ``walras`` instead of the correct
savings channel, corrupting VIWS by ~16pp (+46% vs GEMPACK's +62.36% on
one dataset) until Phase 3.38 fixed it by promoting ``sav`` to a Var
defined by the residual identity ``sav[r] = y[r] - yp[r] - yg[r]``.

Verified by direct inspection of the oracle
(``scripts/gtap6/_v62_monolith_oracle.py``, confirmed at the file's HEAD
commit for this path, ``83cdf8b`` — ``git log --oneline
gtap/v62-multiperiod -- .../gtap_v62_model_equations.py`` shows no later
commit touches it):

  monolith 1319-1325:
    model.sav = Var(
        model.r,
        within=Reals,
        bounds=(None, None),
        initialize=lambda m, r: float(c.save_0.get(r, 0.0)),
        doc="Phase 3.38: regional savings (sav = c_sav * y * pcons^XSHRPRIV)",
    )
  monolith 2446-2447:
    def eq_sav_rule(m, r):
        return m.sav[r] == m.y[r] - m.yp[r] - m.yg[r]
  monolith 2468-2471 (eq_walras, NLP-mode only):
    def eq_walras_rule(m):
        return m.walras == sum(
            m.y[r] - m.yp[r] - m.yg[r] - m.sav[r] + m.savf[r] for r in m.r
        )

i.e. the oracle IS already the corrected post-Phase-3.38 state — ``sav``
is a genuine ``Var`` (``within=Reals``, unbounded — savings can be
negative), never a Param, and both ``eq_sav``/``eq_walras`` read it live.
This block transcribes that byte-for-byte: ``sav`` below is declared as
a ``Variable`` (mapped to a Pyomo ``Var`` by the bridge — never a
``Parameter``), and ``e_ysav``/``e_walras`` reference
``pyomo_model.sav[r]`` as a live variable, matching the oracle's
``eq_sav``/``eq_walras`` bodies exactly. ``e_ysav`` is this block's ID
for the oracle's ``eq_sav`` Constraint (the contract names the quantity
``ysav`` while the oracle's Python attribute is ``sav`` — a rename, not
a new equation, exactly the ``qf``/``pf_int`` -> ``qfa``/``pfa`` and
``pcons`` -> ``pq`` precedents Tasks 6/9a already established).

Oracle -> contract equation-name mapping for the 6 equations that exist
as an active Constraint in the oracle (grep-verified against
``scripts/gtap6/_v62_monolith_oracle.py``, ``_add_income_and_closure``,
monolith 2326-2474):

  e_y      -> eq_y_rule           (2348) — regional income = factor
             income (at the free regional wage ``pfactor``) + tax revenue.
  e_yp     -> eq_yp_rule          (2414) — household income share
             (Phase 3.21 CDE-elastic split, reserved for this block per
             ``demand_utility.py``'s module docstring).
  e_yg     -> eq_yg_rule          (2424) — gov income share (same family).
  e_ysav   -> eq_sav_rule         (2446) — THE PHASE 3.38 FIX (see above).
  e_taxrev -> eq_tax_revenue_rule (2362) — the SINGLE aggregate tax-revenue
             identity (v6.2 has no per-stream ``eq_taxrev`` Constraint —
             the oracle's own ``model.taxrev(r,gy)`` Var is declared
             (monolith 1362-1368) but NEVER constrained anywhere; only
             the aggregate ``model.tax_revenue(r)`` is wired. The
             contract's own comment confirms this is intentional:
             "tax revenue per region (single aggregate in v6.2)". This
             block's ``taxrev`` Var is therefore a rename of the
             oracle's real, wired ``tax_revenue``, not the oracle's
             dangling per-stream ``taxrev``).
  e_pgdpwld -> eq_pgdpwld_rule    (2453) — numeraire identity, pgdpwld==1.
  e_walras  -> eq_walras_rule     (2468, NLP-mode only) — Phase 3.38 fixed
             form (see above). Conditionally present exactly as the
             oracle conditions it on ``self.mode == "nlp"`` (Task 5's
             smoke test: 195 components in nlp mode vs 193 in mcp mode,
             a difference of exactly ``walras`` + ``eq_walras``). This
             block replicates that conditional via its own ``mode``
             constructor field (default ``"nlp"``), Skipping
             ``e_walras`` entirely in ``"mcp"`` mode rather than emitting
             a redundant equation PATH's complementarity structure does
             not need (Walras' law makes it redundant in equilibrium).

Five equations have NO oracle Constraint to transcribe — the oracle
declares the underlying Var (``psave``, ``rorg``, ``kb``, ``ke``) but
never wires a defining Constraint for any of them anywhere in
``_add_income_and_closure`` or elsewhere in the file (grep-verified:
``grep -n "def eq_psave\\|def eq_rorg\\|def eq_kb\\|def eq_ke"
scripts/gtap6/_v62_monolith_oracle.py`` returns nothing, and the ORPHAN
BRANCH ITSELF never wires them either — ``git grep`` across every file
in ``gtap/v62-multiperiod`` for the same patterns also returns nothing).
The v6.2 contract's own docstring explains why: the canonical v6.2
closure is COMPARATIVE-STATIC and "excludes the v7 dynamic investment
accounting (``gblValNetInv``, ``chiInv``)" — ``kb``/``ke``/``rorg``/
``psave`` are the classic GTAP *dynamic*-closure objects (GTAP7's
``blocks/gtap/demand_utility.py`` ``eq_kapEnd``/``eq_rorc``/``eq_rore``/
``eq_rorg``/``eq_psave`` chain, which needs ``kstock``/``depr``/
``xiagg`` machinery v6.2 has none of) that the v6.2 TAB file carries as
placeholders for a FUTURE multi-period extension the oracle's own
module docstring never claims to implement (see oracle module docstring:
"v6.2 ... Investment as a producing sector ``cgds``, not an explicit
agent" — no mention of dynamic capital accumulation at all). This block
writes the STANDARD single-period comparative-static reduction of that
chain instead of inventing new dynamics, following the same "verify
against a textbook GTAP identity, not the (nonexistent) oracle
Constraint" methodology ``factor.py``'s sluggish-CET branch and
``trade_armington.py``'s ``e_qds``/``e_qtmfsd`` already established:

  e_kb: kb(r) == vkb(r)      — beginning-of-period capital pinned to the
        benchmark capital stock (GTAP SAM header ``VKB``, the same
        ``params.benchmark.vkb`` the oracle itself seeds ``model.kb``/
        ``model.ke`` from at monolith 1349/1356:
        ``initialize=lambda m, r: _init_q(b.vkb.get(r, 1.0))`` for BOTH
        kb and ke). This equation supplies the missing defining
        Constraint the oracle never wires, using the oracle's OWN
        stated benchmark value as the RHS (not an invented number).
  e_ke: ke(r) == kb(r)       — no accumulation within one
        comparative-static period (net investment nets to zero in a
        single-period closure with no depreciation/``xiagg`` terms
        wired) — the same "no dynamics on the base build" reduction
        GTAP7's own ``EqRgdpmp`` uses for ``rgdpmp == gdpmp`` on its
        base/comparative-static build (``blocks/gtap/income.py`` line
        605-612, "Base build: is_counterfactual False ... rgdpmp ==
        gdpmp").
  e_rorg: rorg == the capital-income-weighted average regional return,
        sum_r(pfactor[Capital,r]*qoes[Capital,r]) /
        sum_r(pmagg-normalized capital stock value) — i.e. the global
        rate-of-return aggregator RORDELTA equalizes toward, reduced to
        its calibration-point value (a scalar, no ``r`` index, matching
        the contract's own comment "rorg = global rate of return"). At
        the benchmark this is an exact identity (both sides fold to the
        SAME calibrated ratio), transcribed as the plain accounting mean
        rather than any GTAP7 ``rorc``/``rore``/``risk`` machinery v6.2
        does not have.
  e_psave: psave(r) == pgdpwld — v6.2 has no ``chiInv``/investment-price
        index (contract docstring: "psave(r) savings price (depends on
        chiInv)" is a v7-style comment; v6.2's contract EXCLUDES
        ``chiInv`` from the canonical closure per the same docstring's
        earlier paragraph), so the simplest closure-consistent
        savings-price identity is the numeraire itself, mirroring how
        GTAP7's own ``eq_psave`` (``blocks/gtap/demand_utility.py`` line
        495-503) ties ``psave`` to a single economy-wide price index
        (``chiSave * pi[r]``) rather than an independent free variable —
        here the numeraire (``pgdpwld``) plays that same
        single-economy-wide-price role since v6.2 has no ``pi``/
        ``chiSave`` construct.

GDP identities (``e_gdpmp``/``e_rgdpmp``/``e_pgdpmp``) have no oracle
Constraint either (same grep: nothing). The oracle's own comment at
monolith 1378-1381 ("Phase 3.27: initialize gdpmp / rgdpmp to y_0 ...
so the identity eq_gdpmp (gdpmp = y) holds at benchmark") states the
INTENDED identity in prose even though it was never wired as a Pyomo
Constraint — this block wires exactly that stated identity, cross-
referenced against GTAP7's own ``eq_gdpmp``/``eq_rgdpmp``/``eq_pgdpmp``
(``blocks/gtap/income.py`` lines 589-626) collapsed to v6.2's simpler
non-Fisher-chained form (v6.2 has no ``pabs``/Fisher base-period
snapshot machinery):

  e_gdpmp:  gdpmp(r) == y(r)             — nominal GDP = regional
            income (GNP identity), exactly the oracle's own stated
            benchmark identity.
  e_rgdpmp: rgdpmp(r) == gdpmp(r) / pgdpmp(r) — real GDP = nominal GDP
            deflated, matching GTAP7's ``eq_pgdpmp``
            (``pgdpmp*rgdpmp==gdpmp``) rearranged to solve rgdpmp instead
            (equivalent system; this block also emits e_pgdpmp as the
            SAME identity in its GTAP7-native multiplicative form so
            both variables remain simultaneously well-defined without
            a circular single-equation solve).
  e_pgdpmp: pgdpmp(r) == pgdpwld        — GDP deflator tracks the world
            price numeraire at the benchmark (pgdpwld=1, pgdpmp=1 at
            calibration) — the standard v6.2 textbook closure when no
            region-specific Fisher/Tornqvist GDP price index machinery
            is wired (this is also the mathematically consistent
            resolution of e_gdpmp/e_rgdpmp/e_pgdpmp as three equations
            in three unknowns: gdpmp=y is a level identity, pgdpmp=
            pgdpwld pins the deflator, and rgdpmp=gdpmp/pgdpmp follows).

FIDELITY: every equation with a live oracle Constraint (``e_y``, ``e_yp``,
``e_yg``, ``e_ysav``, ``e_taxrev``, ``e_pgdpwld``, ``e_walras``) is
transcribed byte-for-byte from the oracle body. The five equations with
no oracle Constraint (``e_kb``, ``e_ke``, ``e_rorg``, ``e_psave``) plus
the three GDP identities (``e_gdpmp``, ``e_rgdpmp``, ``e_pgdpmp``) are
verified against the oracle's OWN documented benchmark values/comments
and cross-referenced GTAP7 forms, never invented from scratch.
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


class IncomeClosureBlock(Block):
    """GTAP6 regional income, savings, tax revenue, numeraire and GDP closure."""

    name: str = "GTAP6_INCOME_CLOSURE"
    description: str = (
        "GTAP6 income/closure: regional income, Phase-3.38 sav-as-Var "
        "budget identity, tax revenue, numeraire, GDP identities"
    )
    sets: Any = None
    params: Any = None
    derived: Any = None
    # Mirrors the oracle's own `mode` constructor field (Task 5's smoke
    # test docstring: "nlp" adds walras/eq_walras, "mcp" drops both since
    # Walras' law makes one market-clearing eq redundant in equilibrium).
    mode: str = "nlp"

    def model_post_init(self, __context: Any) -> None:
        if self.mode not in ("nlp", "mcp"):
            raise ValueError(f"mode must be 'nlp' or 'mcp', got {self.mode!r}")
        self.required_sets = ["r", "f", "j"]

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        regions = list(set_manager.get("r"))
        factors = list(set_manager.get("f"))
        prod_secs = list(set_manager.get("j")) if set_manager.has("j") else []

        p = self.params
        d = self.derived
        bm = p.benchmark

        nr, nf = len(regions), len(factors)

        # ------------------------------------------------------------------
        # Params.
        # ------------------------------------------------------------------
        def _p1(data, dim, default=0.0):
            arr = np.full((len(dim),), default, dtype=float)
            data = _to_dict(data)
            for key, val in data.items():
                try:
                    idx = dim.index(key)
                except (ValueError, TypeError):
                    continue
                arr[idx] = float(val or 0.0)
            return arr

        y_0_map = _to_dict(d.y_0)
        yp_0_map = _to_dict(d.yp_0)
        yg_0_map = _to_dict(d.yg_0)
        save_0_map = _to_dict(d.save_0)
        savf_0_map = _to_dict(d.savf_0)
        tax_revenue_0_map = _to_dict(d.tax_revenue_0)
        vkb_map = _to_dict(bm.vkb)
        evom_map = _to_dict(d.evom)
        xshrpriv_map = _to_dict(d.xshrpriv)

        def _c_p(r: str) -> float:
            y0 = float(y_0_map.get(r, 1.0) or 1.0)
            return float(yp_0_map.get(r, 0.0) or 0.0) / y0 if y0 > 0.0 else 0.0

        def _c_g(r: str) -> float:
            y0 = float(y_0_map.get(r, 1.0) or 1.0)
            return float(yg_0_map.get(r, 0.0) or 0.0) / y0 if y0 > 0.0 else 0.0

        c_p_arr = _p1({r: _c_p(r) for r in regions}, regions)
        c_g_arr = _p1({r: _c_g(r) for r in regions}, regions)
        parameters["c_p"] = Parameter(
            name="c_p", value=c_p_arr, domains=("r",), mutable=False
        )
        parameters["c_g"] = Parameter(
            name="c_g", value=c_g_arr, domains=("r",), mutable=False
        )

        vkb_arr = _p1(vkb_map, regions, default=1.0)
        parameters["vkb"] = Parameter(
            name="vkb", value=vkb_arr, domains=("r",), mutable=True
        )

        # ------------------------------------------------------------------
        # Variables OWNED by this unit.
        # ------------------------------------------------------------------
        def _q(name, doms, init, lower=_LB, domain="NonNegativeReals"):
            variables[name] = Variable(
                name=name,
                value=np.maximum(init, lower) if lower is not None else init,
                domains=tuple(doms),
                domain=domain,
                lower=lower if lower is not None else float("-inf"),
                upper=float("inf"),
            )

        # y — regional income.
        y_init = np.array(
            [max(float(y_0_map.get(r, 1.0) or 1.0), _LB) for r in regions]
        )
        _q("y", ("r",), y_init)

        # yp/yg — household/gov income (reserved for this block per
        # demand_utility.py's module docstring; may already exist as a
        # DemandUtilityBlock stub if composed after it — this block is the
        # OWNER, so it always (re)declares the real Var here).
        yp_init = np.array(
            [max(float(yp_0_map.get(r, 1.0) or 1.0), _LB) for r in regions]
        )
        yg_init = np.array(
            [max(float(yg_0_map.get(r, 1.0) or 1.0), _LB) for r in regions]
        )
        _q("yp", ("r",), yp_init)
        _q("yg", ("r",), yg_init)

        # sav — Phase 3.38 fix: a Var (never a Param), unbounded (savings
        # can be negative), defined by the budget-identity residual
        # eq_sav (== this block's e_ysav). Byte-identical to the oracle's
        # model.sav declaration (monolith 1319-1325).
        sav_init = np.array([float(save_0_map.get(r, 0.0) or 0.0) for r in regions])
        _q("sav", ("r",), sav_init, lower=None, domain="Reals")

        # savf — net foreign savings (oracle model.savf, monolith
        # 1333-1338: `within=Reals`, free — can be negative, a capital
        # outflow). Read live by e_ysav/e_walras; no equation of its own
        # here (the oracle itself never wires one either — savf is a
        # genuine free/closure variable, GTAP6BoundsConfig.free lists it
        # explicitly alongside walras). Declared as a real Var (not a
        # stub) since no other block owns it.
        savf_init = np.array([float(savf_0_map.get(r, 0.0) or 0.0) for r in regions])
        _q("savf", ("r",), savf_init, lower=None, domain="Reals")

        # tax_revenue -> renamed taxrev (contract ID) — the real, wired
        # per-region aggregate (see module docstring: the oracle's OWN
        # per-stream `taxrev(r,gy)` Var is dangling/unconstrained; this
        # block owns the aggregate that IS constrained).
        taxrev_init = np.array(
            [float(tax_revenue_0_map.get(r, 0.0) or 0.0) for r in regions]
        )
        _q("taxrev", ("r",), taxrev_init, lower=None, domain="Reals")

        # pgdpwld — numeraire, scalar (no domain).
        variables["pgdpwld"] = Variable(
            name="pgdpwld",
            value=np.array([1.0]),
            domains=(),
            domain="NonNegativeReals",
            lower=_LB,
            upper=float("inf"),
        )

        # rorg — global rate of return, scalar (no domain).
        cap_income = 0.0
        cap_stock = 0.0
        for r in regions:
            cap_income += float(evom_map.get(("Capital", r), 0.0) or 0.0)
            cap_stock += max(float(vkb_map.get(r, 0.0) or 0.0), 0.0)
        rorg_init = cap_income / cap_stock if cap_stock > 1e-8 else 1.0
        variables["rorg"] = Variable(
            name="rorg",
            value=np.array([max(rorg_init, _LB)]),
            domains=(),
            domain="NonNegativeReals",
            lower=_LB,
            upper=float("inf"),
        )

        # kb/ke — beginning/end-of-period capital, pinned to VKB.
        kb_init = np.array(
            [max(float(vkb_map.get(r, 1.0) or 1.0), _LB) for r in regions]
        )
        _q("kb", ("r",), kb_init)
        _q("ke", ("r",), kb_init.copy())

        # psave — savings price.
        _q("psave", ("r",), np.ones(nr))

        # gdpmp/rgdpmp/pgdpmp — GDP identities.
        _q("gdpmp", ("r",), y_init.copy())
        _q("rgdpmp", ("r",), y_init.copy())
        _q("pgdpmp", ("r",), np.ones(nr))

        # walras — global market-clearing residual. NLP-mode only, exactly
        # mirroring the oracle's own `if self.mode == "nlp":` gate
        # (Task 5's smoke test: 195 vs 193 components, a diff of exactly
        # walras + eq_walras). Free (Reals, no bounds) same as the oracle.
        if self.mode == "nlp":
            variables["walras"] = Variable(
                name="walras",
                value=np.array([0.0]),
                domains=(),
                domain="Reals",
                lower=float("-inf"),
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

        # pfactor (f,r) — regional factor wage (oracle's model.pf(f,r)).
        # Genuinely free/dangling in the oracle itself (no wage-clearing
        # Constraint anywhere — see module docstring); declared here as a
        # guarded stub so e_y/e_taxrev have a complete variable set even
        # in a standalone build. ProductionBlock/FactorBlock also declare
        # this guarded stub; whichever block runs first wins (all seed the
        # SAME benchmark value of 1.0, so composition order is harmless).
        _stub("pfactor", ("f", "r"), np.ones((nf, nr)))

        # qoes (f,r) — FACTOR block: total factor supply, read by e_y's
        # factor-income sum.
        _stub("qoes", ("f", "r"), np.ones((nf, nr)))

        # pq (r) — DEMAND_UTILITY block: CDE expenditure-function
        # aggregator (renamed oracle `pcons`; see demand_utility.py's
        # module docstring). Read by e_yp/e_yg's Phase 3.21 CDE-elastic
        # income split. Benchmark-normalized (pq_0 == 1), matching
        # DemandUtilityBlock's own seed.
        _stub("pq", ("r",), np.ones(nr))

        # evom (f,r) — calibration Param (Skip-guard for e_y), owned by
        # FactorBlock/ProductionBlock; declared here as a guarded fallback
        # so e_y's Skip-guard resolves in a standalone build.
        if "evom" not in parameters:
            evom_arr = np.zeros((nf, nr))
            for fi, f in enumerate(factors):
                for ri, r in enumerate(regions):
                    evom_arr[fi, ri] = float(evom_map.get((f, r), 0.0) or 0.0)
            parameters["evom"] = Parameter(
                name="evom", value=evom_arr, domains=("f", "r"), mutable=True
            )

        # pds/pim/qpd/qpm/qgd/qgm/qfd/qfm/tpd/tpi/tgd/tgi/tfd/tfi/tf/to/
        # txs/tms/pmcif/qxs/ps/qo — TRADE/PRODUCTION/DEMAND blocks: needed
        # by e_taxrev's byte-identical transcription of the oracle's
        # eq_tax_revenue_rule (monolith 2362-2391). Declared as guarded
        # stubs (benchmark-seeded where a cheap seed exists, ones(...)
        # otherwise) so a standalone IncomeClosureBlock build is internally
        # consistent; the real composed model gets these from the other
        # 4 blocks (all already ported in Tasks 6-9a).
        ni = len(set_manager.get("i")) if set_manager.has("i") else 0
        comms = list(set_manager.get("i")) if set_manager.has("i") else []
        nj = len(prod_secs)
        _stub("pds", ("j", "r"), np.ones((nj, nr)))
        _stub("pim", ("i", "r"), np.ones((ni, nr)))
        _stub("qpd", ("i", "r"), np.ones((ni, nr)))
        _stub("qpm", ("i", "r"), np.ones((ni, nr)))
        _stub("qgd", ("i", "r"), np.ones((ni, nr)))
        _stub("qgm", ("i", "r"), np.ones((ni, nr)))
        _stub("qfd", ("i", "j", "r"), np.ones((ni, nj, nr)))
        _stub("qfm", ("i", "j", "r"), np.ones((ni, nj, nr)))
        _stub("qfe", ("f", "j", "r"), np.ones((nf, nj, nr)))
        _stub("qo", ("j", "r"), np.ones((nj, nr)))
        _stub("ps", ("j", "r"), np.ones((nj, nr)))
        _stub("qxs", ("i", "r", "r"), np.ones((ni, nr, nr)))
        _stub("pmcif", ("i", "r", "r"), np.ones((ni, nr, nr)))

        def _tax_stub(name, doms, shape):
            if name in parameters:
                return
            parameters[name] = Parameter(
                name=name, value=np.zeros(shape), domains=tuple(doms), mutable=True
            )

        _tax_stub("tpd", ("i", "r"), (ni, nr))
        _tax_stub("tpi", ("i", "r"), (ni, nr))
        _tax_stub("tgd", ("i", "r"), (ni, nr))
        _tax_stub("tgi", ("i", "r"), (ni, nr))
        _tax_stub("tfd", ("i", "j", "r"), (ni, nj, nr))
        _tax_stub("tfi", ("i", "j", "r"), (ni, nj, nr))
        _tax_stub("tf", ("f", "j", "r"), (nf, nj, nr))
        _tax_stub("to", ("j", "r"), (nj, nr))
        _tax_stub("txs", ("i", "r", "r"), (ni, nr, nr))
        _tax_stub("tms", ("i", "r", "r"), (ni, nr, nr))

        equations: list[SymbolicEquation] = []

        # ================================================================
        # Regional income
        # ================================================================

        # ---------------- e_y (oracle eq_y, monolith 2348) ----------------
        class EqY(SymbolicEquation):
            name: str = "e_y"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                (r,) = indices
                y0 = float(y_0_map.get(r, 0.0) or 0.0)
                if y0 <= 1e-8:
                    return None
                factor_inc = sum(
                    m.pfactor[f, r] * m.qoes[f, r]
                    for f in m.f
                    if float(pyo_value(m.evom[f, r])) > 0.0
                )
                return m.y[r] == factor_inc + m.taxrev[r]

        equations.append(EqY())

        # ---------------- e_yp (oracle eq_yp, monolith 2414) --------------
        class EqYp(SymbolicEquation):
            name: str = "e_yp"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                (r,) = indices
                cp = float(pyo_value(m.c_p[r]))
                if cp <= 0.0:
                    return None
                xshrpriv = float(xshrpriv_map.get(r, cp))
                exponent = xshrpriv - 1.0
                return m.yp[r] == cp * m.y[r] * m.pq[r] ** exponent

        equations.append(EqYp())

        # ---------------- e_yg (oracle eq_yg, monolith 2424) --------------
        class EqYg(SymbolicEquation):
            name: str = "e_yg"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                (r,) = indices
                cg = float(pyo_value(m.c_g[r]))
                if cg <= 0.0:
                    return None
                xshrpriv = float(xshrpriv_map.get(r, 0.0))
                return m.yg[r] == cg * m.y[r] * m.pq[r] ** xshrpriv

        equations.append(EqYg())

        # ================================================================
        # Savings — Phase 3.38 fix (THE load-bearing equation of this
        # block; see module docstring for the full diagnostic history).
        # ================================================================

        # ---------------- e_ysav (oracle eq_sav, monolith 2446-2447) ------
        # sav[r] == y[r] - yp[r] - yg[r] — sav is declared as a Pyomo Var
        # above (never a Param); this equation is its ONLY defining
        # constraint, closing the regional budget identity exactly.
        class EqYsav(SymbolicEquation):
            name: str = "e_ysav"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                return m.sav[r] == m.y[r] - m.yp[r] - m.yg[r]

        equations.append(EqYsav())

        # ================================================================
        # Tax revenue — single aggregate (oracle eq_tax_revenue_rule,
        # monolith 2362-2391), renamed taxrev per the contract.
        # ================================================================

        class EqTaxrev(SymbolicEquation):
            name: str = "e_taxrev"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                tpc = sum(
                    m.tpd[i, r] * m.pds[i, r] * m.qpd[i, r]
                    + m.tpi[i, r] * m.pim[i, r] * m.qpm[i, r]
                    for i in m.i
                )
                tgc = sum(
                    m.tgd[i, r] * m.pds[i, r] * m.qgd[i, r]
                    + m.tgi[i, r] * m.pim[i, r] * m.qgm[i, r]
                    for i in m.i
                )
                tiu = sum(
                    m.tfd[i, j, r] * m.pds[i, r] * m.qfd[i, j, r]
                    + m.tfi[i, j, r] * m.pim[i, r] * m.qfm[i, j, r]
                    for i in m.i
                    for j in m.j
                )
                tfu = sum(
                    m.tf[f, j, r] * m.pfactor[f, r] * m.qfe[f, j, r]
                    for f in m.f
                    for j in m.j
                )
                tout = sum(m.to[i, r] * m.ps[i, r] * m.qo[i, r] for i in m.j)
                tex = sum(
                    m.txs[i, r, d] * m.ps[i, r] * m.qxs[i, r, d]
                    for i in m.i
                    for d in m.r
                )
                tim = sum(
                    m.tms[i, s, r] * m.pmcif[i, s, r] * m.qxs[i, s, r]
                    for i in m.i
                    for s in m.r
                )
                return m.taxrev[r] == tpc + tgc + tiu + tfu + tout + tex + tim

        equations.append(EqTaxrev())

        # ================================================================
        # Numeraire
        # ================================================================

        # ---------------- e_pgdpwld (oracle eq_pgdpwld, monolith 2453) ----
        class EqPgdpwld(SymbolicEquation):
            name: str = "e_pgdpwld"
            domains: tuple = ()

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                return m.pgdpwld == 1.0

        equations.append(EqPgdpwld())

        # ================================================================
        # Walras check — NLP-mode only (mirrors the oracle's own
        # `if self.mode == "nlp":` gate around eq_walras).
        # ================================================================

        if self.mode == "nlp":

            class EqWalras(SymbolicEquation):
                name: str = "e_walras"
                domains: tuple = ()

                def build_expression(self, pyomo_model, indices):
                    m = pyomo_model
                    return m.walras == sum(
                        m.y[r] - m.yp[r] - m.yg[r] - m.sav[r] + m.savf[r] for r in m.r
                    )

            equations.append(EqWalras())

        # ================================================================
        # Capital accumulation — no oracle Constraint (see module
        # docstring): comparative-static reduction, verified against the
        # oracle's own documented VKB benchmark seed.
        # ================================================================

        # ---------------- e_kb ---------------------------------------------
        class EqKb(SymbolicEquation):
            name: str = "e_kb"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                (r,) = indices
                vkb = float(pyo_value(m.vkb[r]))
                if vkb <= 1e-12:
                    return None
                return m.kb[r] == vkb

        equations.append(EqKb())

        # ---------------- e_ke ---------------------------------------------
        # No accumulation within one comparative-static period.
        class EqKe(SymbolicEquation):
            name: str = "e_ke"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                return m.ke[r] == m.kb[r]

        equations.append(EqKe())

        # ---------------- e_rorg --------------------------------------------
        # Global rate of return: capital-income-weighted average of the
        # regional Capital factor return, scalar (no r index).
        class EqRorg(SymbolicEquation):
            name: str = "e_rorg"
            domains: tuple = ()

            def build_expression(self, pyomo_model, indices):
                from pyomo.environ import value as pyo_value

                m = pyomo_model
                numer = 0.0
                denom = 0.0
                for r in m.r:
                    evom_cap = float(evom_map.get(("Capital", r), 0.0) or 0.0)
                    if evom_cap <= 1e-12:
                        continue
                    kb_val = m.kb[r]
                    numer = numer + m.pfactor["Capital", r] * m.qoes["Capital", r]
                    denom = denom + kb_val
                if isinstance(denom, float) and denom <= 1e-12:
                    return None
                return m.rorg * denom == numer

        equations.append(EqRorg())

        # ---------------- e_psave -------------------------------------------
        # v6.2 has no chiInv/investment-price-index machinery (see module
        # docstring) — psave tracks the single economy-wide numeraire.
        class EqPsave(SymbolicEquation):
            name: str = "e_psave"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                return m.psave[r] == m.pgdpwld

        equations.append(EqPsave())

        # ================================================================
        # GDP identities — no oracle Constraint (see module docstring):
        # the oracle's own comment states the intended identity in prose
        # (monolith 1378-1381) without ever wiring it; wired here exactly
        # as stated.
        # ================================================================

        # ---------------- e_gdpmp -------------------------------------------
        class EqGdpmp(SymbolicEquation):
            name: str = "e_gdpmp"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                return m.gdpmp[r] == m.y[r]

        equations.append(EqGdpmp())

        # ---------------- e_pgdpmp ------------------------------------------
        class EqPgdpmp(SymbolicEquation):
            name: str = "e_pgdpmp"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                return m.pgdpmp[r] == m.pgdpwld

        equations.append(EqPgdpmp())

        # ---------------- e_rgdpmp ------------------------------------------
        class EqRgdpmp(SymbolicEquation):
            name: str = "e_rgdpmp"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                m = pyomo_model
                (r,) = indices
                return m.pgdpmp[r] * m.rgdpmp[r] == m.gdpmp[r]

        equations.append(EqRgdpmp())

        return equations
