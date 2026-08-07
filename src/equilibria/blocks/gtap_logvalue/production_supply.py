"""Log-value GTAP production/supply block (port group `_production`).

Ports src/equilibria/templates/gtap_julia/equations.py::_production verbatim into
composable SymbolicEquations. Value/CES balances are wrapped Log(lhs)==Log(rhs).
Each build_expression re-derives the masked member list + α/σ/γ for its index exactly
as the port's loop body does, reading them from the calibrated point `self.sol`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyomo.environ import log as Log

from equilibria.blocks.base import Block
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable

# ---- calibrated-point access (same semantics as the port's _get/_has) ----


def _get(sol: dict[str, Any], name: str, idx: tuple[str, ...]) -> float:
    d = sol.get(name)
    if d is None:
        return 0.0
    if isinstance(d, dict):
        return float(d.get(idx, 0.0) or 0.0)
    return float(d)


def _has(sol: dict[str, Any], name: str, idx: tuple[str, ...]) -> bool:
    d = sol.get(name)
    if not isinstance(d, dict) or idx not in d:
        return False
    v = d[idx]
    return v == v and v != 0.0  # finite (not NaN) and non-zero


def _ces_input(y, prices, alphas, sigma, gamma, i):
    """CES demand for input i — verbatim from the port's _ces_input."""
    if sigma == 1:
        prod_term = 1.0
        for a, p in zip(alphas, prices, strict=True):
            prod_term = prod_term * (a / p) ** a
        return y / (gamma * prod_term) * (alphas[i] / prices[i])
    if sigma == 0:
        return y * alphas[i] / gamma
    c = (1.0 / gamma) * sum(
        (a**sigma) * (p ** (1.0 - sigma)) for a, p in zip(alphas, prices, strict=True)
    ) ** (1.0 / (1.0 - sigma))
    return (y / gamma) * ((alphas[i] * gamma * c) / prices[i]) ** sigma


def _seed(sol: dict[str, Any], name: str, dims: tuple[str, ...], setmap) -> np.ndarray:
    """np.ndarray seed for `name` over its (block-set) domain, from sol[name]."""
    axes = [setmap[d] for d in dims]
    arr = np.ones([len(ax) for ax in axes], dtype=float)
    d = sol.get(name, {})
    if isinstance(d, dict):
        idx_of = [{m: k for k, m in enumerate(ax)} for ax in axes]
        for key, val in d.items():
            if len(key) != len(axes):
                continue
            try:
                pos = tuple(idx_of[j][key[j]] for j in range(len(axes)))
            except KeyError:
                continue
            if val == val:  # skip NaN
                arr[pos] = val
    return arr


class ProductionSupplyLVBlock(Block):
    name: str = "GTAP_LV_PRODUCTION"
    description: str = "Log-value firm CES/CET production nest + make"
    sol: Any = None

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["r", "i", "a"]

    def setup(self, set_manager, parameters, variables):
        sol = self.sol
        allv = sol  # flat namespace: sol["qo"], sol["α_qintva"], ...
        regs = list(set_manager.get("r"))
        acts = list(set_manager.get("a"))
        comm = list(set_manager.get("i"))
        setmap = {"r": regs, "a": acts, "i": comm}

        # port set key -> block set key for reading sol (port keys α as acts/reg/comm)
        # sol indices are already lower-cased port members == our block members.

        # Vars this block owns (positive lower bound = log domain). (name, dims)
        owned = [
            ("qo", ("a", "r")),
            ("po", ("a", "r")),
            ("qva", ("a", "r")),
            ("pva", ("a", "r")),
            ("qint", ("a", "r")),
            ("pint", ("a", "r")),
            ("qfa", ("i", "a", "r")),
            ("pfa", ("i", "a", "r")),
            ("qfe", ("f", "a", "r")),
            ("pfe", ("f", "a", "r")),
            ("qfd", ("i", "a", "r")),
            ("pfd", ("i", "a", "r")),
            ("qfm", ("i", "a", "r")),
            ("pfm", ("i", "a", "r")),
            ("qca", ("i", "a", "r")),
            ("pca", ("i", "a", "r")),
            ("ps", ("i", "a", "r")),
            ("to", ("i", "a", "r")),
            ("qc", ("i", "r")),
            ("pds", ("i", "r")),
        ]
        seed_setmap = {"r": regs, "a": acts, "i": comm, "f": list(set_manager.get("f"))}
        for nm, dims in owned:
            variables[nm] = Variable(
                name=nm,
                value=_seed(allv, nm, dims, seed_setmap),
                domains=dims,
                domain="NonNegativeReals",
                lower=1e-8,
                upper=float("inf"),
            )

        eqs: list[SymbolicEquation] = []

        # e_qo: qo·po == qva·pva + qint·pint  (γ_qca mask)
        class EqQo(SymbolicEquation):
            name: str = "e_qo"
            domains: tuple = ("a", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                a, r = indices
                if not _has(sol, "γ_qca", (a, r)):
                    return None
                lhs = m.qo[a, r] * m.po[a, r]
                rhs = m.qva[a, r] * m.pva[a, r] + m.qint[a, r] * m.pint[a, r]
                return Log(lhs) == Log(rhs)

        eqs.append(EqQo())

        # e_qintva: CES split of qo into {qint(i=0), qva(i=1)}
        for slot, vname in ((0, "qint"), (1, "qva")):

            class EqQintva(SymbolicEquation):
                name: str = f"e_qintva_{'int' if slot == 0 else 'va'}"
                domains: tuple = ("a", "r")
                _slot: int = slot
                _v: str = vname

                def build_expression(self, pyomo_model, indices):  # noqa: N805
                    m = pyomo_model
                    a, r = indices
                    if not _has(sol, "γ_qca", (a, r)):
                        return None
                    sigma = _get(sol, "esubt", (a, r))
                    gamma = _get(sol, "γ_qintva", (a, r))
                    prices = [m.pint[a, r], m.pva[a, r]]
                    alphas = [
                        _get(sol, "α_qintva", ("int", a, r)),
                        _get(sol, "α_qintva", ("va", a, r)),
                    ]
                    lhs = getattr(m, self._v)[a, r]
                    rhs = _ces_input(
                        m.qo[a, r], prices, alphas, sigma, gamma, self._slot
                    )
                    return Log(lhs) == Log(rhs)

            eqs.append(EqQintva())

        # e_qfa: CES demand of intermediates over comm (one eq per comm, port splits by c)
        class EqQfa(SymbolicEquation):
            name: str = "e_qfa"
            domains: tuple = ("i", "a", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, a, r = indices
                if not _has(sol, "α_qfa", (c, a, r)):
                    return None
                members = [cc for cc in comm if _has(sol, "α_qfa", (cc, a, r))]
                prices = [m.pfa[cc, a, r] for cc in members]
                alphas = [_get(sol, "α_qfa", (cc, a, r)) for cc in members]
                i = members.index(c)
                sigma = _get(sol, "esubc", (a, r))
                gamma = _get(sol, "γ_qfa", (a, r))
                return Log(m.qfa[c, a, r]) == Log(
                    _ces_input(m.qint[a, r], prices, alphas, sigma, gamma, i)
                )

        eqs.append(EqQfa())

        # e_pint: qint·pint == Σ pfa·qfa
        class EqPint(SymbolicEquation):
            name: str = "e_pint"
            domains: tuple = ("a", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                a, r = indices
                if not _has(sol, "γ_qca", (a, r)):
                    return None
                members = [cc for cc in comm if _has(sol, "α_qfa", (cc, a, r))]
                rhs = sum(m.pfa[cc, a, r] * m.qfa[cc, a, r] for cc in members)
                return Log(m.qint[a, r] * m.pint[a, r]) == Log(rhs)

        eqs.append(EqPint())

        # e_qfe: CES demand of factors over endw
        class EqQfe(SymbolicEquation):
            name: str = "e_qfe"
            domains: tuple = ("f", "a", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                e, a, r = indices
                if not _has(sol, "α_qfe", (e, a, r)):
                    return None
                endw = list(set_manager.get("f"))
                members = [ee for ee in endw if _has(sol, "α_qfe", (ee, a, r))]
                prices = [m.pfe[ee, a, r] for ee in members]
                alphas = [_get(sol, "α_qfe", (ee, a, r)) for ee in members]
                i = members.index(e)
                sigma = _get(sol, "esubva", (a, r))
                gamma = _get(sol, "γ_qfe", (a, r))
                return Log(m.qfe[e, a, r]) == Log(
                    _ces_input(m.qva[a, r], prices, alphas, sigma, gamma, i)
                )

        eqs.append(EqQfe())

        # e_pva: qva·pva == Σ pfe·qfe
        class EqPva(SymbolicEquation):
            name: str = "e_pva"
            domains: tuple = ("a", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                a, r = indices
                if not _has(sol, "γ_qca", (a, r)):
                    return None
                endw = list(set_manager.get("f"))
                members = [ee for ee in endw if _has(sol, "α_qfe", (ee, a, r))]
                rhs = sum(m.pfe[ee, a, r] * m.qfe[ee, a, r] for ee in members)
                return Log(m.qva[a, r] * m.pva[a, r]) == Log(rhs)

        eqs.append(EqPva())

        # e_qfd / e_qfm: Armington split of qfa into {qfd(0), qfm(1)}
        for slot, vname in ((0, "qfd"), (1, "qfm")):

            class EqQfdqfm(SymbolicEquation):
                name: str = f"e_{'qfd' if slot == 0 else 'qfm'}"
                domains: tuple = ("i", "a", "r")
                _slot: int = slot
                _v: str = vname

                def build_expression(self, pyomo_model, indices):  # noqa: N805
                    m = pyomo_model
                    c, a, r = indices
                    if not _has(sol, "α_qfdqfm", ("dom", c, a, r)):
                        return None
                    prices = [m.pfd[c, a, r], m.pfm[c, a, r]]
                    alphas = [
                        _get(sol, "α_qfdqfm", ("dom", c, a, r)),
                        _get(sol, "α_qfdqfm", ("imp", c, a, r)),
                    ]
                    sigma = _get(sol, "esubd", (c, r))
                    gamma = _get(sol, "γ_qfdqfm", (c, a, r))
                    lhs = getattr(m, self._v)[c, a, r]
                    rhs = _ces_input(
                        m.qfa[c, a, r], prices, alphas, sigma, gamma, self._slot
                    )
                    return Log(lhs) == Log(rhs)

            eqs.append(EqQfdqfm())

        # e_pfa: pfa·qfa == qfd·pfd + qfm·pfm
        class EqPfa(SymbolicEquation):
            name: str = "e_pfa"
            domains: tuple = ("i", "a", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, a, r = indices
                if not _has(sol, "α_qfdqfm", ("dom", c, a, r)):
                    return None
                lhs = m.pfa[c, a, r] * m.qfa[c, a, r]
                rhs = m.qfd[c, a, r] * m.pfd[c, a, r] + m.qfm[c, a, r] * m.pfm[c, a, r]
                return Log(lhs) == Log(rhs)

        eqs.append(EqPfa())

        # e_qca: CET make of qo into {qca[c]} over comm
        class EqQca(SymbolicEquation):
            name: str = "e_qca"
            domains: tuple = ("i", "a", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, a, r = indices
                if not _has(sol, "α_qca", (c, a, r)):
                    return None
                members = [cc for cc in comm if _has(sol, "α_qca", (cc, a, r))]
                prices = [m.ps[cc, a, r] for cc in members]
                alphas = [_get(sol, "α_qca", (cc, a, r)) for cc in members]
                i = members.index(c)
                sigma = _get(sol, "etraq", (a, r))
                gamma = _get(sol, "γ_qca", (a, r))
                return Log(m.qca[c, a, r]) == Log(
                    _ces_input(m.qo[a, r], prices, alphas, sigma, gamma, i)
                )

        eqs.append(EqQca())

        # e_po: po·qo == Σ qca·ps
        class EqPo(SymbolicEquation):
            name: str = "e_po"
            domains: tuple = ("a", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                a, r = indices
                if not _has(sol, "γ_qca", (a, r)):
                    return None
                members = [cc for cc in comm if _has(sol, "α_qca", (cc, a, r))]
                rhs = sum(m.qca[cc, a, r] * m.ps[cc, a, r] for cc in members)
                return Log(m.po[a, r] * m.qo[a, r]) == Log(rhs)

        eqs.append(EqPo())

        # e_ps: pca == ps·to (output tax, multiplicative power)
        class EqPs(SymbolicEquation):
            name: str = "e_ps"
            domains: tuple = ("i", "a", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, a, r = indices
                if not _has(sol, "α_qca", (c, a, r)):
                    return None
                return Log(m.pca[c, a, r]) == Log(m.ps[c, a, r] * m.to[c, a, r])

        eqs.append(EqPs())

        # e_qc: pds·qc == Σ pca·qca (make aggregation over acts)
        class EqQc(SymbolicEquation):
            name: str = "e_qc"
            domains: tuple = ("i", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, r = indices
                members = [a for a in acts if _has(sol, "α_qca", (c, a, r))]
                if not members:
                    return None
                rhs = sum(m.pca[c, a, r] * m.qca[c, a, r] for a in members)
                return Log(m.pds[c, r] * m.qc[c, r]) == Log(rhs)

        eqs.append(EqQc())

        # e_pca: esubq==0 → pca == pds; else CES (this dataset: esubq==0)
        class EqPca(SymbolicEquation):
            name: str = "e_pca"
            domains: tuple = ("i", "a", "r")

            def build_expression(self, pyomo_model, indices):  # noqa: N805
                m = pyomo_model
                c, a, r = indices
                if not _has(sol, "α_qca", (c, a, r)):
                    return None
                if _get(sol, "esubq", (c, r)) == 0:
                    return Log(m.pca[c, a, r]) == Log(m.pds[c, r])
                members = [aa for aa in acts if _has(sol, "α_qca", (c, aa, r))]
                prices = [m.pca[c, aa, r] for aa in members]
                alphas = [_get(sol, "α_pca", (c, aa, r)) for aa in members]
                i = members.index(a)
                sigma = 1.0 / _get(sol, "esubq", (c, r))
                gamma = _get(sol, "γ_pca", (c, r))
                return Log(m.qca[c, a, r]) == Log(
                    _ces_input(m.qc[c, r], prices, alphas, sigma, gamma, i)
                )

        eqs.append(EqPca())

        return eqs
