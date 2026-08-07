"""Log-value GTAP demand/utility + capital block (port groups `_final_demand` and the
capFlex branch of `_capital`).

Ports private CDE demand, government and investment demand (with their Armington
splits and price indices), plus savings, capital stock, and the rate-of-return /
returns-equalizing capital closure (rordelta=1). The world-closure scalars
(pcgdswld, walras, rorg) live in ClosureLVBlock.
"""

from __future__ import annotations

from typing import Any

from pyomo.environ import log as Log

from equilibria.blocks.base import Block
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable

from ._lv_common import _ces_input, _get, _has, seed_array


def _cde_share(alpha, beta, e, u, prices, income, i):
    """CDE demand for good i — verbatim from the port's _cde_share."""
    w = [
        b * u ** (ej * (1.0 - a)) * (1.0 - a)
        for a, b, ej in zip(alpha, beta, e, strict=True)
    ]
    denom = sum(
        wj * (p / income) ** (1.0 - a)
        for wj, p, a in zip(w, prices, alpha, strict=True)
    )
    return w[i] * (prices[i] / income) ** (-alpha[i]) / denom


def _cde_sum(alpha, beta, e, u, prices, income):
    """CDE sum-of-shares term — verbatim from the port's _cde_sum."""
    return sum(
        b * u ** ((1.0 - a) * ej) * (p / income) ** (1.0 - a)
        for a, b, ej, p in zip(alpha, beta, e, prices, strict=True)
    )


class DemandUtilityLVBlock(Block):
    name: str = "GTAP_LV_DEMAND_UTILITY"
    description: str = "Log-value CDE/gov/investment demand + capital account (capFlex)"
    sol: Any = None
    rordelta: int = 1

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["r", "i", "a", "f", "fc"]

    def setup(self, set_manager, parameters, variables):
        sol = self.sol
        regs = list(set_manager.get("r"))
        acts = list(set_manager.get("a"))
        comm = list(set_manager.get("i"))
        endw = list(set_manager.get("f"))
        endwc = list(set_manager.get("fc"))
        smap = {"r": regs, "i": comm, "a": acts, "f": endw, "fc": endwc}
        rordelta = self.rordelta

        # owned: (name, dims, negative_ok)
        owned = [
            ("up", ("r",), False),
            ("qga", ("i", "r"), False),
            ("pga", ("i", "r"), False),
            ("qia", ("i", "r"), False),
            ("pia", ("i", "r"), False),
            ("pid", ("i", "r"), False),
            ("pim", ("i", "r"), False),
            ("pinv", ("r",), False),
            ("qinv", ("r",), False),
            ("qsave", ("r",), True),
            ("psave", ("r",), False),
            ("kb", ("r",), False),
            ("ke", ("r",), False),
            ("rore", ("r",), False),
            ("rorc", ("r",), False),
            ("rental", ("r",), False),
            ("globalcgds", (), False),
        ]
        for nm, dims, neg in owned:
            variables[nm] = Variable(
                name=nm,
                value=seed_array(sol, nm, dims, smap),
                domains=dims,
                domain="Reals" if neg else "NonNegativeReals",
                lower=-1e12 if neg else 1e-8,
                upper=float("inf"),
            )

        def qga(c, r):
            return _has(sol, "α_qga", (c, r))

        def qia(c, r):
            return _has(sol, "α_qia", (c, r))

        def evfp(e, a, r):
            return _has(sol, "α_qfe", (e, a, r))

        def T(name, idx):
            return _get(sol, name, idx)

        eqs: list[SymbolicEquation] = []

        # e_qpa: private CDE demand per good
        class EqQpa(SymbolicEquation):
            name: str = "e_qpa"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, r = indices
                alpha = [1.0 - T("subpar", (cc, r)) for cc in comm]
                beta = [T("β_qpa", (cc, r)) for cc in comm]
                e = [T("incpar", (cc, r)) for cc in comm]
                prices = [m.ppa[cc, r] for cc in comm]
                i = comm.index(c)
                income = m.yp[r] / T("pop", (r,))
                lhs = m.qpa[c, r] / T("pop", (r,))
                rhs = _cde_share(alpha, beta, e, m.up[r], prices, income, i)
                return Log(lhs) == Log(rhs)

        eqs.append(EqQpa())

        # e_up: CDE closure  1 == Σ shares
        class EqUp(SymbolicEquation):
            name: str = "e_up"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (r,) = indices
                alpha = [1.0 - T("subpar", (cc, r)) for cc in comm]
                beta = [T("β_qpa", (cc, r)) for cc in comm]
                e = [T("incpar", (cc, r)) for cc in comm]
                prices = [m.ppa[cc, r] for cc in comm]
                income = m.yp[r] / T("pop", (r,))
                return Log(1.0) == Log(
                    _cde_sum(alpha, beta, e, m.up[r], prices, income)
                )

        eqs.append(EqUp())

        # e_qpd / e_qpm: Armington split of qpa
        for slot, vname in ((0, "qpd"), (1, "qpm")):

            class EqQpdqpm(SymbolicEquation):
                name: str = f"e_{'qpd' if slot == 0 else 'qpm'}"
                domains: tuple = ("i", "r")
                _slot: int = slot
                _v: str = vname

                def build_expression(self, pyomo_model, indices):  # noqa: N805
                    m = pyomo_model
                    c, r = indices
                    prices = [m.ppd[c, r], m.ppm[c, r]]
                    alphas = [
                        T("α_qpdqpm", ("dom", c, r)),
                        T("α_qpdqpm", ("imp", c, r)),
                    ]
                    sigma = T("esubd", (c, r))
                    gamma = T("γ_qpdqpm", (c, r))
                    lhs = getattr(m, self._v)[c, r]
                    rhs = _ces_input(
                        m.qpa[c, r], prices, alphas, sigma, gamma, self._slot
                    )
                    return Log(lhs) == Log(rhs)

            eqs.append(EqQpdqpm())

        # e_ppriv: private consumption price index
        class EqPpriv(SymbolicEquation):
            name: str = "e_ppriv"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (r,) = indices
                qsum = sum(m.qpa[c, r] for c in comm)
                vsum = sum(m.ppa[c, r] * m.qpa[c, r] for c in comm)
                return Log(m.ppriv[r] * qsum) == Log(vsum)

        eqs.append(EqPpriv())

        # e_qga: gov demand  pga·qga == yg·α_qga
        class EqQga(SymbolicEquation):
            name: str = "e_qga"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, r = indices
                if not qga(c, r):
                    return None
                return Log(m.pga[c, r] * m.qga[c, r]) == Log(
                    m.yg[r] * T("α_qga", (c, r))
                )

        eqs.append(EqQga())

        # e_pgov: gov price index
        class EqPgov(SymbolicEquation):
            name: str = "e_pgov"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (r,) = indices
                cells = [c for c in comm if qga(c, r)]
                if not cells:
                    return None
                qsum = sum(m.qga[c, r] for c in cells)
                vsum = sum(m.qga[c, r] * m.pga[c, r] for c in cells)
                return Log(m.pgov[r] * qsum) == Log(vsum)

        eqs.append(EqPgov())

        # e_qgd / e_qgm: Armington split of qga
        for slot, vname in ((0, "qgd"), (1, "qgm")):

            class EqQgdqgm(SymbolicEquation):
                name: str = f"e_{'qgd' if slot == 0 else 'qgm'}"
                domains: tuple = ("i", "r")
                _slot: int = slot
                _v: str = vname

                def build_expression(self, pyomo_model, indices):  # noqa: N805
                    m = pyomo_model
                    c, r = indices
                    if not qga(c, r):
                        return None
                    prices = [m.pgd[c, r], m.pgm[c, r]]
                    alphas = [
                        T("α_qgdqgm", ("dom", c, r)),
                        T("α_qgdqgm", ("imp", c, r)),
                    ]
                    sigma = T("esubd", (c, r))
                    gamma = T("γ_qgdqgm", (c, r))
                    lhs = getattr(m, self._v)[c, r]
                    rhs = _ces_input(
                        m.qga[c, r], prices, alphas, sigma, gamma, self._slot
                    )
                    return Log(lhs) == Log(rhs)

            eqs.append(EqQgdqgm())

        # e_pga: gov value balance
        class EqPga(SymbolicEquation):
            name: str = "e_pga"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, r = indices
                if not qga(c, r):
                    return None
                lhs = m.qga[c, r] * m.pga[c, r]
                rhs = m.pgd[c, r] * m.qgd[c, r] + m.pgm[c, r] * m.qgm[c, r]
                return Log(lhs) == Log(rhs)

        eqs.append(EqPga())

        # e_qia: investment demand (Leontief σ=0)
        class EqQia(SymbolicEquation):
            name: str = "e_qia"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, r = indices
                if not qia(c, r):
                    return None
                members = [cc for cc in comm if qia(cc, r)]
                prices = [m.pia[cc, r] for cc in members]
                alphas = [T("α_qia", (cc, r)) for cc in members]
                i = members.index(c)
                gamma = T("γ_qia", (r,))
                return Log(m.qia[c, r]) == Log(
                    _ces_input(m.qinv[r], prices, alphas, 0.0, gamma, i)
                )

        eqs.append(EqQia())

        # e_pinv: investment price index
        class EqPinv(SymbolicEquation):
            name: str = "e_pinv"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (r,) = indices
                cells = [c for c in comm if qia(c, r)]
                qsum = sum(m.qia[c, r] for c in cells)
                vsum = sum(m.pia[c, r] * m.qia[c, r] for c in cells)
                return Log(m.pinv[r] * qsum) == Log(vsum)

        eqs.append(EqPinv())

        # e_qid / e_qim: Armington split of qia
        for slot, vname in ((0, "qid"), (1, "qim")):

            class EqQidqim(SymbolicEquation):
                name: str = f"e_{'qid' if slot == 0 else 'qim'}"
                domains: tuple = ("i", "r")
                _slot: int = slot
                _v: str = vname

                def build_expression(self, pyomo_model, indices):  # noqa: N805
                    m = pyomo_model
                    c, r = indices
                    if not qia(c, r):
                        return None
                    prices = [m.pid[c, r], m.pim[c, r]]
                    alphas = [
                        T("α_qidqim", ("dom", c, r)),
                        T("α_qidqim", ("imp", c, r)),
                    ]
                    sigma = T("esubd", (c, r))
                    gamma = T("γ_qidqim", (c, r))
                    lhs = getattr(m, self._v)[c, r]
                    rhs = _ces_input(
                        m.qia[c, r], prices, alphas, sigma, gamma, self._slot
                    )
                    return Log(lhs) == Log(rhs)

            eqs.append(EqQidqim())

        # e_pia: investment value balance
        class EqPia(SymbolicEquation):
            name: str = "e_pia"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, r = indices
                if not qia(c, r):
                    return None
                lhs = m.pia[c, r] * m.qia[c, r]
                rhs = m.pid[c, r] * m.qid[c, r] + m.pim[c, r] * m.qim[c, r]
                return Log(lhs) == Log(rhs)

        eqs.append(EqPia())

        # ---- capital account (capFlex, rordelta=1) ----

        # e_qsave: saving balance
        class EqQsave(SymbolicEquation):
            name: str = "e_qsave"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (r,) = indices
                syp, syg = T("σyp", (r,)), T("σyg", (r,))
                lhs = m.y[r] * m.uelas[r]
                rhs = (
                    syp * m.y[r] * m.uelas[r]
                    + syg * m.y[r] * m.uelas[r]
                    + m.psave[r] * m.qsave[r]
                )
                return Log(lhs) == Log(rhs)

        eqs.append(EqQsave())

        # e_psave: saving price (semi-log form; global sums). Non-log wrapper (_add_raw).
        class EqPsave(SymbolicEquation):
            name: str = "e_psave"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (r,) = indices
                net_global = sum(m.qinv[rr] - T("δ", (rr,)) * m.kb[rr] for rr in regs)
                rhs = (
                    Log(m.pinv[r])
                    + sum(
                        ((m.qinv[rr] - T("δ", (rr,)) * m.kb[rr]) - m.qsave[rr])
                        * Log(m.pinv[rr])
                        for rr in regs
                    )
                    / net_global
                )
                return Log(m.psave[r]) == rhs

        eqs.append(EqPsave())

        # e_kb: capital stock  ρ·kb == Σ qe[capital]
        class EqKb(SymbolicEquation):
            name: str = "e_kb"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (r,) = indices
                rhs = sum(m.qe[e, r] for e in endwc)
                return Log(T("ρ", (r,)) * m.kb[r]) == Log(rhs)

        eqs.append(EqKb())

        # e_ke: end-of-period capital
        class EqKe(SymbolicEquation):
            name: str = "e_ke"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (r,) = indices
                return Log(m.ke[r]) == Log(m.qinv[r] + (1.0 - T("δ", (r,))) * m.kb[r])

        eqs.append(EqKe())

        # e_rore: expected return
        class EqRore(SymbolicEquation):
            name: str = "e_rore"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (r,) = indices
                lhs = T("α_qinv", (r,)) * m.rore[r]
                rhs = m.rorc[r] / (m.ke[r] / m.kb[r]) ** T("rorflex", (r,))
                return Log(lhs) == Log(rhs)

        eqs.append(EqRore())

        # e_rorc: current return
        class EqRorc(SymbolicEquation):
            name: str = "e_rorc"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (r,) = indices
                cells = [(e, a) for e in endwc for a in acts if evfp(e, a, r)]
                qsum = sum(m.qes[e, a, r] for e, a in cells)
                lhs = m.rorc[r] * (qsum - T("δ", (r,)) * m.kb[r])
                rhs = qsum * (m.rental[r] / m.pinv[r])
                return Log(lhs) == Log(rhs)

        eqs.append(EqRorc())

        # e_rental: capital rental price
        class EqRental(SymbolicEquation):
            name: str = "e_rental"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (r,) = indices
                cells = [(e, a) for e in endwc for a in acts if evfp(e, a, r)]
                qsum = sum(m.qes[e, a, r] for e, a in cells)
                vsum = sum(m.qes[e, a, r] * m.pes[e, a, r] for e, a in cells)
                return Log(qsum * m.rental[r]) == Log(vsum)

        eqs.append(EqRental())

        if rordelta == 1:
            # e_qinv: returns equalize rore == rorg
            class EqQinv(SymbolicEquation):
                name: str = "e_qinv"
                domains: tuple = ("r",)

                def build_expression(self, pyomo_model, indices):  # noqa: N805
                    m = pyomo_model
                    (r,) = indices
                    return Log(m.rore[r]) == Log(m.rorg)

            eqs.append(EqQinv())

            # e_globalcgds: globalcgds == Σ net investment
            class EqGlobalcgds(SymbolicEquation):
                name: str = "e_globalcgds"
                domains: tuple = ()

                def build_expression(self, pyomo_model, indices):  # noqa: N805
                    m = pyomo_model
                    rhs = sum(m.qinv[r] - T("δ", (r,)) * m.kb[r] for r in regs)
                    return Log(m.globalcgds) == Log(rhs)

            eqs.append(EqGlobalcgds())

        return eqs
