"""Log-value GTAP factor block (port group `_factors`).

Ports equations.py::_factors verbatim: factor use==supply, use/income tax wedges,
regional factor price index, and the mobile / sluggish(CET) / fixed endowment market
clearing. Factor subsets are fm (mobile), fs (sluggish), ff (fixed).
"""

from __future__ import annotations

from typing import Any

from pyomo.environ import log as Log

from equilibria.blocks.base import Block
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable

from ._lv_common import _ces_input, _get, _has, seed_array


class FactorLVBlock(Block):
    name: str = "GTAP_LV_FACTOR"
    description: str = (
        "Log-value factor market: use=supply, tax wedges, mobile/sluggish/fixed"
    )
    sol: Any = None

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["r", "a", "f", "fm", "fs", "ff"]

    def setup(self, set_manager, parameters, variables):
        sol = self.sol
        regs = list(set_manager.get("r"))
        acts = list(set_manager.get("a"))
        endw = list(set_manager.get("f"))
        endwm = list(set_manager.get("fm"))
        endws = list(set_manager.get("fs"))
        endwf = list(set_manager.get("ff"))
        smap = {"r": regs, "a": acts, "f": endw, "fm": endwm, "fs": endws, "ff": endwf}

        owned = [
            ("qes", ("f", "a", "r")),
            ("peb", ("f", "a", "r")),
            ("pes", ("f", "a", "r")),
            ("tfe", ("f", "a", "r")),
            ("tinc", ("f", "a", "r")),
            ("pfactor", ("r",)),
            ("pe", ("fms", "r")),
            ("qe", ("fms", "r")),
            ("qesf", ("ff", "a", "r")),
        ]
        # pe/qe are indexed over endwms (mobile+sluggish); use fms set
        fms = list(set_manager.get("fms"))
        smap["fms"] = fms
        for nm, dims in owned:
            variables[nm] = Variable(
                name=nm,
                value=seed_array(sol, nm, dims, smap),
                domains=dims,
                domain="NonNegativeReals",
                lower=1e-8,
                upper=float("inf"),
            )

        def evfp(e, a, r):
            return _has(sol, "α_qfe", (e, a, r))

        eqs: list[SymbolicEquation] = []

        # e_peb: qfe == qes
        class EqPeb(SymbolicEquation):
            name: str = "e_peb"
            domains: tuple = ("f", "a", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                e, a, r = indices
                if not evfp(e, a, r):
                    return None
                return Log(m.qfe[e, a, r]) == Log(m.qes[e, a, r])

        eqs.append(EqPeb())

        # e_pfe: pfe == peb·tfe
        class EqPfe(SymbolicEquation):
            name: str = "e_pfe"
            domains: tuple = ("f", "a", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                e, a, r = indices
                if not evfp(e, a, r):
                    return None
                return Log(m.pfe[e, a, r]) == Log(m.peb[e, a, r] * m.tfe[e, a, r])

        eqs.append(EqPfe())

        # e_pes: peb == pes·tinc
        class EqPes(SymbolicEquation):
            name: str = "e_pes"
            domains: tuple = ("f", "a", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                e, a, r = indices
                if not evfp(e, a, r):
                    return None
                return Log(m.peb[e, a, r]) == Log(m.pes[e, a, r] * m.tinc[e, a, r])

        eqs.append(EqPes())

        # e_pfactor: pfactor·Σqfe == Σ qfe·peb
        class EqPfactor(SymbolicEquation):
            name: str = "e_pfactor"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (r,) = indices
                cells = [(e, a) for e in endw for a in acts if evfp(e, a, r)]
                if not cells:
                    return None
                qsum = sum(m.qfe[e, a, r] for e, a in cells)
                vsum = sum(m.qfe[e, a, r] * m.peb[e, a, r] for e, a in cells)
                return Log(m.pfactor[r] * qsum) == Log(vsum)

        eqs.append(EqPfactor())

        # e_pe1: mobile market clearing qe == Σ_a qfe
        class EqPe1(SymbolicEquation):
            name: str = "e_pe1"
            domains: tuple = ("fm", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                e, r = indices
                cells = [a for a in acts if evfp(e, a, r)]
                if not cells:
                    return None
                return Log(m.qe[e, r]) == Log(sum(m.qfe[e, a, r] for a in cells))

        eqs.append(EqPe1())

        # e_qes1: mobile one-price pes == pe
        class EqQes1(SymbolicEquation):
            name: str = "e_qes1"
            domains: tuple = ("fm", "a", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                e, a, r = indices
                if not evfp(e, a, r):
                    return None
                return Log(m.pes[e, a, r]) == Log(m.pe[e, r])

        eqs.append(EqQes1())

        # e_qes2: sluggish CET supply across acts
        class EqQes2(SymbolicEquation):
            name: str = "e_qes2"
            domains: tuple = ("fs", "a", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                e, a, r = indices
                if not evfp(e, a, r):
                    return None
                members = [aa for aa in acts if evfp(e, aa, r)]
                prices = [m.pes[e, aa, r] for aa in members]
                alphas = [_get(sol, "α_qes2", (e, aa, r)) for aa in members]
                i = members.index(a)
                sigma = _get(sol, "etrae", (e, r))
                gamma = _get(sol, "γ_qes2", (e, r))
                return Log(m.qes[e, a, r]) == Log(
                    _ces_input(m.qe[e, r], prices, alphas, sigma, gamma, i)
                )

        eqs.append(EqQes2())

        # e_pe2: sluggish value balance pe·qe == Σ pes·qes
        class EqPe2(SymbolicEquation):
            name: str = "e_pe2"
            domains: tuple = ("fs", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                e, r = indices
                cells = [a for a in acts if evfp(e, a, r)]
                if not cells:
                    return None
                rhs = sum(m.pes[e, a, r] * m.qes[e, a, r] for a in cells)
                return Log(m.pe[e, r] * m.qe[e, r]) == Log(rhs)

        eqs.append(EqPe2())

        # e_qes3: fixed factor qes == qesf
        class EqQes3(SymbolicEquation):
            name: str = "e_qes3"
            domains: tuple = ("ff", "a", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                e, a, r = indices
                if not evfp(e, a, r):
                    return None
                return Log(m.qes[e, a, r]) == Log(m.qesf[e, a, r])

        eqs.append(EqQes3())

        return eqs
