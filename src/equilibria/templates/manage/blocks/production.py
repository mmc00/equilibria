"""MANAGE's production group (design, group A): the CES tree and its links.

``CESTree`` writes the tree (``e_xfac``, ``e_pxfac``, ``e_pxfacInd``);
``ManageProduction`` writes the rows around it (``data/manage/gams/inc/model.inc``):

- ``e_xap`` (:1379) and ``e_pap`` (:1385): an intermediate leaf of the tree is
  the activity's Armington demand ``xa(i,a)``, priced at ``p_chiPA·paf``;
- ``e_lambdaxfac`` (:1428): the productivity shifter of a nest, the product of
  its shifters, for the cells GAMS leaves free (labor in the baseline, because
  ``chixFacL`` is endogenous; iterloop.inc:88-115 fixes the others);
- ``e_px`` (:1455) and ``e_pp`` (:1466): unit cost and producer price;
- ``e_xpNew`` (:2997): output is the top nest relative to its 2014 level;
- ``e_yieldLnd`` (:1480): output per unit of land (reporting).

GAMS indexes every tree cell by the vintage ``v``; with one vintage
(``ifVint = 0``, ``old = v = {""}``) Python drops it (``calibration``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field

from equilibria.blocks._levels import LevelBlock, eq
from equilibria.blocks.base import Block
from equilibria.blocks.production import CESTree
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.templates.manage.calibration import Calibration

# every column of MANAGE's 2015 solve is free (gha2015_scalar.gms: FX 0, no bounds)
FREE = -float("inf")

PRODUCTION_FAMILIES: tuple[str, ...] = (
    "e_xfac", "e_pxfac", "e_pxfacInd", "e_xap", "e_pap", "e_lambdaxfac",
    "e_px", "e_pp", "e_xpNew", "e_yieldLnd",
)  # fmt: skip


def production_tree(**inputs: Any) -> CESTree:
    """The production tree: ``p_alphaXfac(fpChl, fpPar, a)`` over the nests."""
    return CESTree(
        name="ProductionTree",
        quantity="xfac",
        price="pxfac",
        index_price="pxfacInd",
        shifter="lambdaxfac",
        node_set="fpNest",
        parent_set="fpPar",
        owner_set="a",
        alpha="p_alphaXfac",
        sigma="p_sigmaXFac",
        additive="isAdditiveCes",
        top="fpTop",
        lower=FREE,
        **inputs,
    )


class ManageProduction(LevelBlock):
    """The rows that tie MANAGE's production tree to its prices and output."""

    name: str = Field(default="ManageProduction")

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["fpNest", "fpPar", "a", "i", "aa", "em", "aagr"]
        self.uses = [
            "xfac", "pxfac", "lambdaxfac", "xp", "xs", "paf", "tPrd", "tpsb",
            "tPhn", "tEmi", "emical", "chiEmiXP", "chixFacL",
        ]  # fmt: skip

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        el = {s: set_manager.get(s).elements for s in self.required_sets}
        n = {s: len(v) for s, v in el.items()}

        def members(name: str, *sets: str) -> set[tuple[str, ...]]:
            """The cells where the 0/1 array ``name`` is 1."""
            arr = self.value(name)
            return {
                tuple(el[s][i] for s, i in zip(sets, pos, strict=True))
                for pos in zip(*np.nonzero(arr), strict=True)
            }

        flag = members("flagXfac", "fpNest", "a")
        aact = {a for (a,) in members("aact", "a")}
        top = {t for (t,) in members("fpTop", "fpPar")}
        nest_i = {c for (c,) in members("fpNestI", "fpNest")}
        lab = {c for (c,) in members("l", "fpNest")}
        cap = {c for (c,) in members("cap", "fpNest")}
        fac = {c for (c,) in members("fp", "fpNest")}
        nrg = {c for (c,) in members("nrgFac", "fpNest")}
        land = {c for (c,) in members("actHa", "fpNest")}
        agr: dict[str, list[str]] = {}
        for g, a in sorted(members("mapAagr", "aagr", "a")):
            agr.setdefault(a, []).append(g)
        emi = members("emiFlag", "em", "aa")
        sam_ia = members("sam", "i", "a")
        free_lambda = members("lambdaxfac_free", "fpNest", "a")
        rho = members("p_rhoEmiXP", "em", "a")
        p_as = members("p_as", "a", "i")

        self.param(parameters, "p_chiPA", ("i",))
        self.param(parameters, "p_as", ("a", "i"))
        self.param(parameters, "p_rhoEmiXP", ("em", "a"))
        self.param(parameters, "p_phiEmiXP", ("em", "a"))
        self.param(parameters, "xfac_t0", ("fpNest", "a"))
        self.param(parameters, "xp_t0", ("a",))
        self.var(variables, "px", ("a",), (n["a"],), lower=FREE)
        self.var(variables, "pp", ("a",), (n["a"],), lower=FREE)
        self.var(variables, "yieldLnd", ("a",), (n["a"],), lower=FREE)
        self.var(variables, "xa", ("i", "aa"), (n["i"], n["aa"]), lower=FREE)
        # the productivity shifters (instruments, fixed but chixFacL;
        # model.inc:553-562); chixFacV(v) is a scalar with one vintage
        self.var(variables, "chixFacF", ("fpNest",), (n["fpNest"],), lower=FREE)
        self.var(variables, "chixFacA", ("a",), (n["a"],), lower=FREE)
        self.var(
            variables, "chixFacFA", ("fpNest", "a"), (n["fpNest"], n["a"]), lower=FREE
        )
        self.var(
            variables,
            "chixFacAgr",
            ("aagr", "fpNest"),
            (n["aagr"], n["fpNest"]),
            lower=FREE,
        )
        for scalar in ("chixFacV", "chixFacK", "chixFacKL", "chixFacFP", "chixFacNRG"):
            self.var(variables, scalar, (), (), lower=FREE)

        def xap(m: Any, idx: tuple[str, ...]) -> Any:
            i, a = idx  # e_xap(i,a) $ sam(i,a)
            if (i, a) not in sam_ia:
                return None
            return m.xa[i, a] == (m.xfac[i, a] if (i, a) in flag else 0.0)

        def pap(m: Any, idx: tuple[str, ...]) -> Any:
            c, a = idx  # e_pap(flagXfac(fpNestI,a,old))
            if c not in nest_i or (c, a) not in flag:
                return None
            return m.pxfac[c, a] == m.p_chiPA[c] * m.paf[c, a]

        def lambdaxfac(m: Any, idx: tuple[str, ...]) -> Any:
            c, a = idx
            if (c, a) not in free_lambda:
                return None
            rhs = m.chixFacF[c] * m.chixFacA[a] * m.chixFacV * m.chixFacFA[c, a]
            if c in lab:
                rhs = rhs * (1 + m.chixFacL)
            if c in cap:
                rhs = rhs * (1 + m.chixFacK)
            if c in cap or c in lab:
                rhs = rhs * (1 + m.chixFacKL)
            if c in fac:
                rhs = rhs * (1 + m.chixFacFP)
            for g in agr.get(a, ()):
                rhs = rhs * m.chixFacAgr[g, c]
            if c in nrg:
                rhs = rhs * (1 + m.chixFacNRG)
            return m.lambdaxfac[c, a] == rhs

        def px(m: Any, idx: tuple[str, ...]) -> Any:
            (a,) = idx
            if a not in aact:
                return None
            cost = sum(
                m.xfac[t, a] * m.pxfac[t, a] for t in sorted(top) if (t, a) in flag
            )
            return m.px[a] * m.xp[a] == cost

        def pp(m: Any, idx: tuple[str, ...]) -> Any:
            (a,) = idx
            if a not in aact:
                return None
            rhs = m.px[a] * (1 + m.tPrd[a] + m.tpsb[a] + m.tPhn[a])
            for e in el["em"]:
                if (e, a) in emi and (e, a) in rho:
                    rhs = rhs + (
                        m.tEmi[e, a] * m.emical * m.p_phiEmiXP[e, a]
                        * m.chiEmiXP[e, a] * m.p_rhoEmiXP[e, a]
                    )  # fmt: skip
            return m.pp[a] == rhs

        def xp_new(m: Any, idx: tuple[str, ...]) -> Any:
            t, a = idx  # e_xpNew(fpTop,aact(a),v), old(v) and not ifVint
            if t not in top or a not in aact:
                return None
            rhs = m.xfac[t, a] / m.xfac_t0[t, a] * m.xp_t0[a] if (t, a) in flag else 0.0
            return m.xp[a] == rhs

        def yield_lnd(m: Any, idx: tuple[str, ...]) -> Any:
            (a,) = idx
            used = [c for c in el["fpNest"] if c in land and (c, a) in flag]
            if not used:
                return None
            supply = sum(m.xs[i] for i in el["i"] if (a, i) in p_as)
            return supply == m.yieldLnd[a] * sum(m.xfac[c, a] for c in used)

        return [
            eq("e_xap", ("i", "a"), xap),
            eq("e_pap", ("fpNest", "a"), pap),
            eq("e_lambdaxfac", ("fpNest", "a"), lambdaxfac),
            eq("e_px", ("a",), px),
            eq("e_pp", ("a",), pp),
            eq("e_xpNew", ("fpPar", "a"), xp_new),
            eq("e_yieldLnd", ("a",), yield_lnd),
        ]


def production_group(cal: Calibration) -> tuple[Block, ...]:
    """The blocks of group A, in order, from the calibration."""
    inputs = cal.block_inputs()
    values, start = inputs["values"], inputs["start"]
    return (
        production_tree(values=values, start=start),
        ManageProduction(values=values, start=start),
    )
