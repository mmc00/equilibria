"""Log-value GTAP domestic/export block (dom/export part of port group `_trade`).

Ports the two market-clearing families e_qds / e_pds from equations.py::_trade; the
import nest and margins live in ArmingtonBilateralLVBlock.
"""

from __future__ import annotations

from typing import Any

from pyomo.environ import log as Log

from equilibria.blocks.base import Block
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable

from ._lv_common import _has, seed_array


class TradeCETLVBlock(Block):
    name: str = "GTAP_LV_TRADE_CET"
    description: str = "Log-value domestic aggregate + market clearing (qds, qc)"
    sol: Any = None

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["r", "i", "a", "marg"]

    def setup(self, set_manager, parameters, variables):
        sol = self.sol
        regs = list(set_manager.get("r"))
        acts = list(set_manager.get("a"))
        comm = list(set_manager.get("i"))
        marg = list(set_manager.get("marg"))
        smap = {"r": regs, "i": comm}

        variables["qds"] = Variable(
            name="qds",
            value=seed_array(sol, "qds", ("i", "r"), smap),
            domains=("i", "r"),
            domain="NonNegativeReals",
            lower=1e-8,
            upper=float("inf"),
        )

        def qfa(c, a, r):
            return _has(sol, "α_qfa", (c, a, r))

        def qga(c, r):
            return _has(sol, "α_qga", (c, r))

        def qia(c, r):
            return _has(sol, "α_qia", (c, r))

        def qxs(c, s, d):
            return _has(sol, "α_qxs", (c, s, d))

        eqs: list[SymbolicEquation] = []

        # e_qds: domestic aggregate
        class EqQds(SymbolicEquation):
            name: str = "e_qds"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, r = indices
                rhs = sum(m.qfd[c, a, r] for a in acts if qfa(c, a, r)) + m.qpd[c, r]
                if qga(c, r):
                    rhs = rhs + m.qgd[c, r]
                if qia(c, r):
                    rhs = rhs + m.qid[c, r]
                return Log(m.qds[c, r]) == Log(rhs)

        eqs.append(EqQds())

        # e_pds: market clearing qc == qds + exports + margin supply
        class EqPds(SymbolicEquation):
            name: str = "e_pds"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, r = indices
                rhs = m.qds[c, r] + sum(m.qxs[c, r, d] for d in regs if qxs(c, r, d))
                if c in marg:
                    rhs = rhs + m.qst[c, r]
                return Log(m.qc[c, r]) == Log(rhs)

        eqs.append(EqPds())

        return eqs
