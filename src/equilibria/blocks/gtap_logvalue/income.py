"""Log-value GTAP income block (port group `_income`).

Ports equations.py::_income verbatim: consumer prices (ppd/ppm/ppa), factor income,
regional income = fincome + Σ tax revenue (every instrument), the household/gov income
split, and the consumer price + utility aggregation. e_uepriv/e_uelas are non-log
(plain lhs==rhs), the rest are Log balances.
"""

from __future__ import annotations

from typing import Any

from pyomo.environ import log as Log

from equilibria.blocks.base import Block
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable

from ._lv_common import _get, _has, seed_array


class IncomeLVBlock(Block):
    name: str = "GTAP_LV_INCOME"
    description: str = (
        "Log-value income: consumer prices, tax revenue, utility aggregation"
    )
    sol: Any = None

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["r", "i", "a", "f"]

    def setup(self, set_manager, parameters, variables):
        sol = self.sol
        regs = list(set_manager.get("r"))
        acts = list(set_manager.get("a"))
        comm = list(set_manager.get("i"))
        endw = list(set_manager.get("f"))
        smap = {"r": regs, "a": acts, "i": comm, "f": endw}

        owned = [
            ("ppd", ("i", "r")),
            ("ppm", ("i", "r")),
            ("ppa", ("i", "r")),
            ("qpa", ("i", "r")),
            ("qpd", ("i", "r")),
            ("qpm", ("i", "r")),
            ("tpd", ("i", "r")),
            ("tpm", ("i", "r")),
            ("fincome", ("r",)),
            ("y", ("r",)),
            ("yp", ("r",)),
            ("yg", ("r",)),
            ("ug", ("r",)),
            ("u", ("r",)),
            ("p", ("r",)),
            ("uelas", ("r",)),
            ("uepriv", ("r",)),
            ("ppriv", ("r",)),
            ("pgov", ("r",)),
            ("qgd", ("i", "r")),
            ("qgm", ("i", "r")),
            ("tgd", ("i", "r")),
            ("tgm", ("i", "r")),
            ("qid", ("i", "r")),
            ("qim", ("i", "r")),
            ("tid", ("i", "r")),
            ("tim", ("i", "r")),
            ("tfd", ("i", "a", "r")),
            ("tfm", ("i", "a", "r")),
            ("txs", ("i", "r", "r")),
            ("tms", ("i", "r", "r")),
        ]
        for nm, dims in owned:
            variables[nm] = Variable(
                name=nm,
                value=seed_array(sol, nm, dims, smap),
                domains=dims,
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

        def qca(c, a, r):
            return _has(sol, "α_qca", (c, a, r))

        def evfp(e, a, r):
            return _has(sol, "α_qfe", (e, a, r))

        def T(name, idx):
            return _get(sol, name, idx)

        eqs: list[SymbolicEquation] = []

        # e_ppd / e_ppm
        class EqPpd(SymbolicEquation):
            name: str = "e_ppd"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, r = indices
                return Log(m.ppd[c, r]) == Log(m.pds[c, r] * m.tpd[c, r])

        class EqPpm(SymbolicEquation):
            name: str = "e_ppm"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, r = indices
                return Log(m.ppm[c, r]) == Log(m.pms[c, r] * m.tpm[c, r])

        eqs += [EqPpd(), EqPpm()]

        # e_ppa: qpa·ppa == ppd·qpd + ppm·qpm
        class EqPpa(SymbolicEquation):
            name: str = "e_ppa"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, r = indices
                lhs = m.qpa[c, r] * m.ppa[c, r]
                rhs = m.ppd[c, r] * m.qpd[c, r] + m.ppm[c, r] * m.qpm[c, r]
                return Log(lhs) == Log(rhs)

        eqs.append(EqPpa())

        # e_fincome: factor income net of depreciation
        class EqFincome(SymbolicEquation):
            name: str = "e_fincome"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (r,) = indices
                cells = [(e, a) for e in endw for a in acts if evfp(e, a, r)]
                fac = sum(m.peb[e, a, r] * m.qes[e, a, r] for e, a in cells)
                return Log(m.fincome[r]) == Log(
                    fac - T("δ", (r,)) * m.pinv[r] * m.kb[r]
                )

        eqs.append(EqFincome())

        # e_y: regional income = fincome + Σ tax revenue
        class EqY(SymbolicEquation):
            name: str = "e_y"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (r,) = indices
                rev = m.fincome[r]
                for c in comm:
                    rev = rev + m.qpd[c, r] * m.pds[c, r] * (m.tpd[c, r] - 1)
                    rev = rev + m.qpm[c, r] * m.pms[c, r] * (m.tpm[c, r] - 1)
                    if qga(c, r):
                        rev = rev + m.qgd[c, r] * m.pds[c, r] * (m.tgd[c, r] - 1)
                        rev = rev + m.qgm[c, r] * m.pms[c, r] * (m.tgm[c, r] - 1)
                    if qia(c, r):
                        rev = rev + m.qid[c, r] * m.pds[c, r] * (m.tid[c, r] - 1)
                        rev = rev + m.qim[c, r] * m.pms[c, r] * (m.tim[c, r] - 1)
                for c in comm:
                    for a in acts:
                        if qfa(c, a, r):
                            rev = rev + m.qfd[c, a, r] * m.pfd[c, a, r] / m.tfd[
                                c, a, r
                            ] * (m.tfd[c, a, r] - 1)
                            rev = rev + m.qfm[c, a, r] * m.pfm[c, a, r] / m.tfm[
                                c, a, r
                            ] * (m.tfm[c, a, r] - 1)
                        if qca(c, a, r):
                            rev = rev + m.qca[c, a, r] * m.ps[c, a, r] * (
                                m.to[c, a, r] - 1
                            )
                for e in endw:
                    for a in acts:
                        if evfp(e, a, r):
                            rev = rev + m.qfe[e, a, r] * m.peb[e, a, r] * (
                                m.tfe[e, a, r] - 1
                            )
                for c in comm:
                    for d in regs:
                        if qxs(c, r, d):
                            rev = rev + m.qxs[c, r, d] * m.pfob[c, r, d] / m.txs[
                                c, r, d
                            ] * (m.txs[c, r, d] - 1)
                    for s in regs:
                        if qxs(c, s, r):
                            rev = rev + m.qxs[c, s, r] * m.pcif[c, s, r] * (
                                m.tms[c, s, r] - 1
                            )
                return Log(m.y[r]) == Log(rev)

        eqs.append(EqY())

        # e_uepriv (non-log): uepriv = Σ qpa·ppa·incpar / yp
        class EqUepriv(SymbolicEquation):
            name: str = "e_uepriv"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (r,) = indices
                num = sum(m.qpa[c, r] * m.ppa[c, r] * T("incpar", (c, r)) for c in comm)
                return m.uepriv[r] == num / m.yp[r]

        eqs.append(EqUepriv())

        # e_uelas (non-log)
        class EqUelas(SymbolicEquation):
            name: str = "e_uelas"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (r,) = indices
                syp, syg = T("σyp", (r,)), T("σyg", (r,))
                return m.uelas[r] == 1.0 / (syp / m.uepriv[r] + syg + (1.0 - syp - syg))

        eqs.append(EqUelas())

        # e_yp
        class EqYp(SymbolicEquation):
            name: str = "e_yp"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (r,) = indices
                return Log(m.yp[r]) == Log(
                    m.y[r] * T("σyp", (r,)) * m.uelas[r] / m.uepriv[r]
                )

        eqs.append(EqYp())

        # e_yg
        class EqYg(SymbolicEquation):
            name: str = "e_yg"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (r,) = indices
                return Log(m.yg[r]) == Log(m.y[r] * T("σyg", (r,)) * m.uelas[r])

        eqs.append(EqYg())

        # e_ug
        class EqUg(SymbolicEquation):
            name: str = "e_ug"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (r,) = indices
                return Log(m.ug[r]) == Log(m.yg[r] / T("pop", (r,)) / m.pgov[r])

        eqs.append(EqUg())

        # e_p: consumer price index
        class EqP(SymbolicEquation):
            name: str = "e_p"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (r,) = indices
                syp, syg = T("σyp", (r,)), T("σyg", (r,))
                rhs = (
                    m.ppriv[r] * syp + m.pgov[r] * syg + m.psave[r] * (1.0 - syp - syg)
                )
                return Log(m.p[r]) == Log(rhs)

        eqs.append(EqP())

        # e_u
        class EqU(SymbolicEquation):
            name: str = "e_u"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (r,) = indices
                return Log(m.u[r]) == Log(m.y[r] / m.p[r] / T("pop", (r,)))

        eqs.append(EqU())

        return eqs
