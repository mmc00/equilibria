"""Log-value GTAP world-closure block (world-scalar part of `_capital` + numeraire).

World price of capital goods, Walras supply/demand, the world factor price (rorg), and
the numeraire pin ppa[comm0, reg0]. All scalar equations (domains=()).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyomo.environ import log as Log

from equilibria.blocks.base import Block
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable

from ._lv_common import _get, _has


class ClosureLVBlock(Block):
    name: str = "GTAP_LV_CLOSURE"
    description: str = "Log-value world closure: pcgdswld, Walras, rorg, numeraire"
    sol: Any = None

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["r", "i", "a", "f"]

    def setup(self, set_manager, parameters, variables):
        sol = self.sol
        regs = list(set_manager.get("r"))
        acts = list(set_manager.get("a"))
        comm = list(set_manager.get("i"))
        endw = list(set_manager.get("f"))
        comm0, reg0 = comm[0], regs[0]

        def T(name, idx):
            return _get(sol, name, idx)

        def evfp(e, a, r):
            return _has(sol, "α_qfe", (e, a, r))

        # scalar vars this block owns
        for nm in ("walras_sup", "walras_dem", "pcgdswld", "pfactwld", "rorg"):
            v = T(nm, ())
            variables[nm] = Variable(
                name=nm,
                value=np.asarray([v if v else 1.0]),
                domains=(),
                domain="NonNegativeReals",
                lower=1e-8,
                upper=float("inf"),
            )

        allc = [(e, a, r) for e in endw for a in acts for r in regs if evfp(e, a, r)]
        ppa0 = T("ppa", (comm0, reg0)) or 1.0

        eqs: list[SymbolicEquation] = []

        # e_pcgdswld: pcgdswld·net_global == v_net_global
        class EqPcgdswld(SymbolicEquation):
            name: str = "e_pcgdswld"
            domains: tuple = ()

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                net_global = sum(m.qinv[r] - T("δ", (r,)) * m.kb[r] for r in regs)
                v_net_global = sum(
                    m.pinv[r] * (m.qinv[r] - T("δ", (r,)) * m.kb[r]) for r in regs
                )
                return Log(m.pcgdswld * net_global) == Log(v_net_global)

        eqs.append(EqPcgdswld())

        # e_walras_sup
        class EqWalrasSup(SymbolicEquation):
            name: str = "e_walras_sup"
            domains: tuple = ()

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                return Log(m.walras_sup) == Log(m.pcgdswld * m.globalcgds)

        eqs.append(EqWalrasSup())

        # e_walras_dem
        class EqWalrasDem(SymbolicEquation):
            name: str = "e_walras_dem"
            domains: tuple = ()

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                return Log(m.walras_dem) == Log(
                    sum(m.psave[r] * m.qsave[r] for r in regs)
                )

        eqs.append(EqWalrasDem())

        # e_rorg: world factor price
        class EqRorg(SymbolicEquation):
            name: str = "e_rorg"
            domains: tuple = ()

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                lhs = m.pfactwld * sum(m.qfe[e, a, r] for e, a, r in allc)
                rhs = sum(m.peb[e, a, r] * m.qfe[e, a, r] for e, a, r in allc)
                return Log(lhs) == Log(rhs)

        eqs.append(EqRorg())

        # e_numeraire: pin ppa[comm0, reg0]
        class EqNumeraire(SymbolicEquation):
            name: str = "e_numeraire"
            domains: tuple = ()

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                return Log(m.ppa[comm0, reg0]) == Log(ppa0)

        eqs.append(EqNumeraire())

        return eqs
