"""Log-value GTAP bilateral Armington block (import-nest part of port group `_trade`).

Ports equations.py::_trade verbatim, EXCEPT the domestic/export market-clearing
(e_qds/e_pds), which live in TradeCETLVBlock. Covers the import aggregate, the
bilateral Armington CES sourcing, FOB/CIF/delivered prices, transport margins, and the
firm/gov/investment tax links. Bilateral index is (comm, src, dest) → domains ("i","rp","r").
"""

from __future__ import annotations

from typing import Any

from pyomo.environ import log as Log

from equilibria.blocks.base import Block
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable

from ._lv_common import _ces_input, _get, _has, seed_array


class ArmingtonBilateralLVBlock(Block):
    name: str = "GTAP_LV_ARMINGTON"
    description: str = "Log-value bilateral Armington import nest + margins + tax links"
    sol: Any = None

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["r", "i", "a", "marg", "rp"]

    def setup(self, set_manager, parameters, variables):
        sol = self.sol
        regs = list(set_manager.get("r"))
        acts = list(set_manager.get("a"))
        comm = list(set_manager.get("i"))
        marg = list(set_manager.get("marg"))
        smap = {"r": regs, "i": comm, "a": acts, "marg": marg, "rp": regs}

        owned = [
            ("qms", ("i", "r")),
            ("pms", ("i", "r")),
            ("qxs", ("i", "rp", "r")),
            ("pmds", ("i", "rp", "r")),
            ("pfob", ("i", "rp", "r")),
            ("pcif", ("i", "rp", "r")),
            ("ptrans", ("i", "rp", "r")),
            ("tx", ("i", "r")),
            ("tm", ("i", "r")),
            ("qtmfsd", ("marg", "i", "rp", "r")),
            ("qtm", ("marg",)),
            ("qst", ("marg", "r")),
            ("pt", ("marg",)),
            ("pfd", ("i", "a", "r")),
            ("pfm", ("i", "a", "r")),
            ("pgd", ("i", "r")),
            ("pgm", ("i", "r")),
            ("pid", ("i", "r")),
            ("pim", ("i", "r")),
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

        def vtwr(mm, c, s, d):
            return _has(sol, "α_qtmfsd", (mm, c, s, d))

        def vtwr_sum(c, s, d):
            return any(vtwr(mm, c, s, d) for mm in marg)

        eqs: list[SymbolicEquation] = []

        # e_qms: import aggregate
        class EqQms(SymbolicEquation):
            name: str = "e_qms"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, r = indices
                rhs = sum(m.qfm[c, a, r] for a in acts if qfa(c, a, r)) + m.qpm[c, r]
                if qga(c, r):
                    rhs = rhs + m.qgm[c, r]
                if qia(c, r):
                    rhs = rhs + m.qim[c, r]
                return Log(m.qms[c, r]) == Log(rhs)

        eqs.append(EqQms())

        # e_qxs: bilateral CES sourcing
        class EqQxs(SymbolicEquation):
            name: str = "e_qxs"
            domains: tuple = ("i", "rp", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, s, d = indices
                if not qxs(c, s, d):
                    return None
                origins = [ss for ss in regs if qxs(c, ss, d)]
                prices = [m.pmds[c, ss, d] for ss in origins]
                alphas = [_get(sol, "α_qxs", (c, ss, d)) for ss in origins]
                i = origins.index(s)
                sigma = _get(sol, "esubm", (c, d))
                gamma = _get(sol, "γ_qxs", (c, d))
                return Log(m.qxs[c, s, d]) == Log(
                    _ces_input(m.qms[c, d], prices, alphas, sigma, gamma, i)
                )

        eqs.append(EqQxs())

        # e_pms: pms·qms == Σ_s pmds·qxs
        class EqPms(SymbolicEquation):
            name: str = "e_pms"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, d = indices
                origins = [s for s in regs if qxs(c, s, d)]
                if not origins:
                    return None
                rhs = sum(m.pmds[c, s, d] * m.qxs[c, s, d] for s in origins)
                return Log(m.pms[c, d] * m.qms[c, d]) == Log(rhs)

        eqs.append(EqPms())

        # e_pfob: pfob == pds·tx·txs
        class EqPfob(SymbolicEquation):
            name: str = "e_pfob"
            domains: tuple = ("i", "rp", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, s, d = indices
                if not qxs(c, s, d):
                    return None
                return Log(m.pfob[c, s, d]) == Log(
                    m.pds[c, s] * m.tx[c, s] * m.txs[c, s, d]
                )

        eqs.append(EqPfob())

        # e_pcif: pcif·qxs == pfob·qxs + margins
        class EqPcif(SymbolicEquation):
            name: str = "e_pcif"
            domains: tuple = ("i", "rp", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, s, d = indices
                if not qxs(c, s, d):
                    return None
                rhs = m.pfob[c, s, d] * m.qxs[c, s, d]
                if vtwr_sum(c, s, d):
                    rhs = rhs + m.ptrans[c, s, d] * sum(
                        m.qtmfsd[mm, c, s, d] for mm in marg if vtwr(mm, c, s, d)
                    )
                return Log(m.pcif[c, s, d] * m.qxs[c, s, d]) == Log(rhs)

        eqs.append(EqPcif())

        # e_pmds: pmds == pcif·tm·tms
        class EqPmds(SymbolicEquation):
            name: str = "e_pmds"
            domains: tuple = ("i", "rp", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, s, d = indices
                if not qxs(c, s, d):
                    return None
                return Log(m.pmds[c, s, d]) == Log(
                    m.pcif[c, s, d] * m.tm[c, d] * m.tms[c, s, d]
                )

        eqs.append(EqPmds())

        # e_qtmfsd: margin demand ∝ trade flow
        class EqQtmfsd(SymbolicEquation):
            name: str = "e_qtmfsd"
            domains: tuple = ("marg", "i", "rp", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                mm, c, s, d = indices
                if not vtwr(mm, c, s, d):
                    return None
                return Log(m.qtmfsd[mm, c, s, d]) == Log(
                    _get(sol, "α_qtmfsd", (mm, c, s, d)) * m.qxs[c, s, d]
                )

        eqs.append(EqQtmfsd())

        # e_ptrans: transport price index per route
        class EqPtrans(SymbolicEquation):
            name: str = "e_ptrans"
            domains: tuple = ("i", "rp", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, s, d = indices
                if not vtwr_sum(c, s, d):
                    return None
                ms = [mm for mm in marg if vtwr(mm, c, s, d)]
                qsum = sum(m.qtmfsd[mm, c, s, d] for mm in ms)
                vsum = sum(m.qtmfsd[mm, c, s, d] * m.pt[mm] for mm in ms)
                return Log(m.ptrans[c, s, d] * qsum) == Log(vsum)

        eqs.append(EqPtrans())

        # e_qtm: total margin demand
        class EqQtm(SymbolicEquation):
            name: str = "e_qtm"
            domains: tuple = ("marg",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (mm,) = indices
                cells = [
                    (c, s, d)
                    for c in comm
                    for s in regs
                    for d in regs
                    if vtwr(mm, c, s, d)
                ]
                if not cells:
                    return None
                rhs = sum(m.qtmfsd[mm, c, s, d] for c, s, d in cells)
                return Log(m.qtm[mm]) == Log(rhs)

        eqs.append(EqQtm())

        # e_qst: CES supply of margin from regions
        class EqQst(SymbolicEquation):
            name: str = "e_qst"
            domains: tuple = ("marg", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                mm, r = indices
                if not _has(sol, "α_qst", (mm, r)):
                    return None
                members = [rr for rr in regs if _has(sol, "α_qst", (mm, rr))]
                prices = [m.pds[mm, rr] for rr in members]
                alphas = [_get(sol, "α_qst", (mm, rr)) for rr in members]
                i = members.index(r)
                sigma = _get(sol, "esubs", (mm,))
                gamma = _get(sol, "γ_qst", (mm,))
                return Log(m.qst[mm, r]) == Log(
                    _ces_input(m.qtm[mm], prices, alphas, sigma, gamma, i)
                )

        eqs.append(EqQst())

        # e_pt: margin value balance pt·qtm == Σ pds·qst
        class EqPt(SymbolicEquation):
            name: str = "e_pt"
            domains: tuple = ("marg",)

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                (mm,) = indices
                members = [r for r in regs if _has(sol, "α_qst", (mm, r))]
                if not members:
                    return None
                rhs = sum(m.pds[mm, r] * m.qst[mm, r] for r in members)
                return Log(m.pt[mm] * m.qtm[mm]) == Log(rhs)

        eqs.append(EqPt())

        # tax links pfd/pfm (firm, domains i,a,r) and pgd/pgm/pid/pim (i,r)
        class EqPfd(SymbolicEquation):
            name: str = "e_pfd"
            domains: tuple = ("i", "a", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, a, r = indices
                if not qfa(c, a, r):
                    return None
                return Log(m.pfd[c, a, r]) == Log(m.pds[c, r] * m.tfd[c, a, r])

        class EqPfm(SymbolicEquation):
            name: str = "e_pfm"
            domains: tuple = ("i", "a", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, a, r = indices
                if not qfa(c, a, r):
                    return None
                return Log(m.pfm[c, a, r]) == Log(m.pms[c, r] * m.tfm[c, a, r])

        class EqPgd(SymbolicEquation):
            name: str = "e_pgd"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, r = indices
                if not qga(c, r):
                    return None
                return Log(m.pgd[c, r]) == Log(m.pds[c, r] * m.tgd[c, r])

        class EqPgm(SymbolicEquation):
            name: str = "e_pgm"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, r = indices
                if not qga(c, r):
                    return None
                return Log(m.pgm[c, r]) == Log(m.pms[c, r] * m.tgm[c, r])

        class EqPid(SymbolicEquation):
            name: str = "e_pid"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, r = indices
                if not qia(c, r):
                    return None
                return Log(m.pid[c, r]) == Log(m.pds[c, r] * m.tid[c, r])

        class EqPim(SymbolicEquation):
            name: str = "e_pim"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, r = indices
                if not qia(c, r):
                    return None
                return Log(m.pim[c, r]) == Log(m.pms[c, r] * m.tim[c, r])

        eqs += [EqPfd(), EqPfm(), EqPgd(), EqPgm(), EqPid(), EqPim()]

        return eqs
