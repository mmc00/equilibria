"""gtap_julia equations — log-value-balance constraints ported from Julia
build_model!.jl, grouped thematically (Tasks 6-11).

Every constraint is ``log(lhs) == log(rhs)`` (value balances) or
``log(qty_i) == log(ces(...)_i)`` (CES demand, per input i). Taxes enter as
multiplicative powers. Existence masks δ derive from the calibrated shares
(δ = share is finite & non-zero). Elasticities and shares come from the loaded
Julia solution dict (``sol["all"]`` flat namespace).
"""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo

Log = pyo.log


def _get(sol: dict[str, Any], name: str, idx: tuple[str, ...] | None = None) -> float:
    """Scalar param/share lookup from the flat solution namespace."""
    d = sol["all"].get(name)
    if d is None:
        raise KeyError(f"param {name} not in solution")
    if idx is None:
        return float(d) if not isinstance(d, dict) else float(next(iter(d.values())))
    skey = tuple(str(k) for k in idx)
    return float(d[skey])


def _has(sol: dict[str, Any], name: str, idx: tuple[str, ...]) -> bool:
    """δ mask: the calibrated share for this cell is present & non-zero."""
    d = sol["all"].get(name)
    if not isinstance(d, dict):
        return False
    v = d.get(tuple(str(k) for k in idx))
    return v is not None and v == v and v != 0.0  # finite (not NaN) and non-zero


def _ces_input(y, prices, alphas, sigma, gamma, i):
    """Symbolic CES demand for input i (Pyomo expression). Mirrors Julia ces().

    y, prices[·], alphas[·] are Pyomo expressions/params; sigma, gamma floats.
    Returns the demand expression for the i-th input among the given (masked) list.
    """
    if sigma == 1:
        prod_term = 1.0
        for a, p in zip(alphas, prices, strict=True):
            prod_term = prod_term * (a / p) ** a
        return y / (gamma * prod_term) * (alphas[i] / prices[i])
    if sigma == 0:
        return y * alphas[i] / gamma
    # sigma > 0 (and the sigma<0 branch has the same closed form for non-zero α)
    c = (1.0 / gamma) * sum(
        (a**sigma) * (p ** (1.0 - sigma)) for a, p in zip(alphas, prices, strict=True)
    ) ** (1.0 / (1.0 - sigma))
    return (y / gamma) * ((alphas[i] * gamma * c) / prices[i]) ** sigma


def _add(model, name, rule_pairs):
    """Attach a scalar Constraint per (key, lhs, rhs) and return (name, comp)."""
    idxset = pyo.Set(initialize=[k for k, _, _ in rule_pairs])
    model.add_component(f"_{name}_idx", idxset)
    data = {k: (lhs, rhs) for k, lhs, rhs in rule_pairs}

    def _rule(m, *key):
        k = key if len(key) > 1 else key[0]
        lhs, rhs = data[k]
        return Log(lhs) == Log(rhs)

    con = pyo.Constraint(idxset, rule=_rule)
    model.add_component(name, con)
    return name, con


def _production(model, sol):
    regs = sol["sets"]["reg"]
    acts = sol["sets"]["acts"]
    comm = sol["sets"]["comm"]
    endw = sol["sets"]["endw"]
    out = []

    # e_qo: value balance  qo·po == qva·pva + qint·pint
    pairs = []
    for a in acts:
        for r in regs:
            if not _has(sol, "γ_qca", (a, r)):
                continue
            pairs.append(
                (
                    (a, r),
                    model.qo[a, r] * model.po[a, r],
                    model.qva[a, r] * model.pva[a, r]
                    + model.qint[a, r] * model.pint[a, r],
                )
            )
    if pairs:
        out.append(_add(model, "e_qo", pairs))

    # e_qintva: CES split of qo into {qint, qva}
    pairs_qint, pairs_qva = [], []
    for a in acts:
        for r in regs:
            if not _has(sol, "γ_qca", (a, r)):
                continue
            sigma = _get(sol, "esubt", (a, r))
            gamma = _get(sol, "γ_qintva", (a, r))
            prices = [model.pint[a, r], model.pva[a, r]]
            alphas = [
                _get(sol, "α_qintva", ("int", a, r)),
                _get(sol, "α_qintva", ("va", a, r)),
            ]
            pairs_qint.append(
                (
                    (a, r),
                    model.qint[a, r],
                    _ces_input(model.qo[a, r], prices, alphas, sigma, gamma, 0),
                )
            )
            pairs_qva.append(
                (
                    (a, r),
                    model.qva[a, r],
                    _ces_input(model.qo[a, r], prices, alphas, sigma, gamma, 1),
                )
            )
    if pairs_qint:
        out.append(_add(model, "e_qintva_int", pairs_qint))
        out.append(_add(model, "e_qintva_va", pairs_qva))

    # e_qfa: CES demand of intermediates {qfa[c]} from qint (nest over comm)
    for c in comm:
        pairs = []
        for a in acts:
            for r in regs:
                if not _has(sol, "α_qfa", (c, a, r)):
                    continue
                members = [cc for cc in comm if _has(sol, "α_qfa", (cc, a, r))]
                prices = [model.pfa[cc, a, r] for cc in members]
                alphas = [_get(sol, "α_qfa", (cc, a, r)) for cc in members]
                i = members.index(c)
                sigma = _get(sol, "esubc", (a, r))
                gamma = _get(sol, "γ_qfa", (a, r))
                pairs.append(
                    (
                        (c, a, r),
                        model.qfa[c, a, r],
                        _ces_input(model.qint[a, r], prices, alphas, sigma, gamma, i),
                    )
                )
        if pairs:
            out.append(_add(model, f"e_qfa_{c}", pairs))

    # e_pint: value balance qint·pint == Σ pfa·qfa
    pairs = []
    for a in acts:
        for r in regs:
            if not _has(sol, "γ_qca", (a, r)):
                continue
            members = [cc for cc in comm if _has(sol, "α_qfa", (cc, a, r))]
            rhs = sum(model.pfa[cc, a, r] * model.qfa[cc, a, r] for cc in members)
            pairs.append(((a, r), model.qint[a, r] * model.pint[a, r], rhs))
    if pairs:
        out.append(_add(model, "e_pint", pairs))

    # e_qfe: CES demand of factors {qfe[e]} from qva (nest over endw)
    for e in endw:
        pairs = []
        for a in acts:
            for r in regs:
                if not _has(sol, "α_qfe", (e, a, r)):
                    continue
                members = [ee for ee in endw if _has(sol, "α_qfe", (ee, a, r))]
                prices = [model.pfe[ee, a, r] for ee in members]
                alphas = [_get(sol, "α_qfe", (ee, a, r)) for ee in members]
                i = members.index(e)
                sigma = _get(sol, "esubva", (a, r))
                gamma = _get(sol, "γ_qfe", (a, r))
                pairs.append(
                    (
                        (e, a, r),
                        model.qfe[e, a, r],
                        _ces_input(model.qva[a, r], prices, alphas, sigma, gamma, i),
                    )
                )
        if pairs:
            out.append(_add(model, f"e_qfe_{e}", pairs))

    # e_pva: value balance qva·pva == Σ pfe·qfe
    pairs = []
    for a in acts:
        for r in regs:
            if not _has(sol, "γ_qca", (a, r)):
                continue
            members = [ee for ee in endw if _has(sol, "α_qfe", (ee, a, r))]
            rhs = sum(model.pfe[ee, a, r] * model.qfe[ee, a, r] for ee in members)
            pairs.append(((a, r), model.qva[a, r] * model.pva[a, r], rhs))
    if pairs:
        out.append(_add(model, "e_pva", pairs))

    # e_qfdqfm: Armington split of qfa[c] into {qfd, qfm} (nest dom/imp)
    pairs_d, pairs_m = [], []
    for c in comm:
        for a in acts:
            for r in regs:
                if not _has(sol, "α_qfdqfm", ("dom", c, a, r)):
                    continue
                prices = [model.pfd[c, a, r], model.pfm[c, a, r]]
                alphas = [
                    _get(sol, "α_qfdqfm", ("dom", c, a, r)),
                    _get(sol, "α_qfdqfm", ("imp", c, a, r)),
                ]
                sigma = _get(sol, "esubd", (c, r))
                gamma = _get(sol, "γ_qfdqfm", (c, a, r))
                pairs_d.append(
                    (
                        (c, a, r),
                        model.qfd[c, a, r],
                        _ces_input(model.qfa[c, a, r], prices, alphas, sigma, gamma, 0),
                    )
                )
                pairs_m.append(
                    (
                        (c, a, r),
                        model.qfm[c, a, r],
                        _ces_input(model.qfa[c, a, r], prices, alphas, sigma, gamma, 1),
                    )
                )
    if pairs_d:
        out.append(_add(model, "e_qfd", pairs_d))
        out.append(_add(model, "e_qfm", pairs_m))

    # e_pfa: value balance pfa·qfa == qfd·pfd + qfm·pfm
    pairs = []
    for c in comm:
        for a in acts:
            for r in regs:
                if not _has(sol, "α_qfdqfm", ("dom", c, a, r)):
                    continue
                pairs.append(
                    (
                        (c, a, r),
                        model.pfa[c, a, r] * model.qfa[c, a, r],
                        model.qfd[c, a, r] * model.pfd[c, a, r]
                        + model.qfm[c, a, r] * model.pfm[c, a, r],
                    )
                )
    if pairs:
        out.append(_add(model, "e_pfa", pairs))

    # e_qca: CET make of qo into {qca[c]} (nest over comm, etraq<0)
    for c in comm:
        pairs = []
        for a in acts:
            for r in regs:
                if not _has(sol, "α_qca", (c, a, r)):
                    continue
                members = [cc for cc in comm if _has(sol, "α_qca", (cc, a, r))]
                prices = [model.ps[cc, a, r] for cc in members]
                alphas = [_get(sol, "α_qca", (cc, a, r)) for cc in members]
                i = members.index(c)
                sigma = _get(sol, "etraq", (a, r))
                gamma = _get(sol, "γ_qca", (a, r))
                pairs.append(
                    (
                        (c, a, r),
                        model.qca[c, a, r],
                        _ces_input(model.qo[a, r], prices, alphas, sigma, gamma, i),
                    )
                )
        if pairs:
            out.append(_add(model, f"e_qca_{c}", pairs))

    # e_po: value balance po·qo == Σ qca·ps
    pairs = []
    for a in acts:
        for r in regs:
            if not _has(sol, "γ_qca", (a, r)):
                continue
            members = [cc for cc in comm if _has(sol, "α_qca", (cc, a, r))]
            rhs = sum(model.qca[cc, a, r] * model.ps[cc, a, r] for cc in members)
            pairs.append(((a, r), model.po[a, r] * model.qo[a, r], rhs))
    if pairs:
        out.append(_add(model, "e_po", pairs))

    # e_ps: output tax link  pca == ps·to (multiplicative power)
    pairs = []
    for c in comm:
        for a in acts:
            for r in regs:
                if not _has(sol, "α_qca", (c, a, r)):
                    continue
                pairs.append(
                    (
                        (c, a, r),
                        model.pca[c, a, r],
                        model.ps[c, a, r] * _get(sol, "to", (c, a, r)),
                    )
                )
    if pairs:
        out.append(_add(model, "e_ps", pairs))

    # e_qc: value balance pds·qc == Σ pca·qca (make aggregation over acts)
    pairs = []
    for c in comm:
        for r in regs:
            members = [a for a in acts if _has(sol, "α_qca", (c, a, r))]
            if not members:
                continue
            rhs = sum(model.pca[c, a, r] * model.qca[c, a, r] for a in members)
            pairs.append(((c, r), model.pds[c, r] * model.qc[c, r], rhs))
    if pairs:
        out.append(_add(model, "e_qc", pairs))

    return out


_GROUPS = {
    "production": _production,
}


def build_group(model, sol: dict[str, Any], group: str):
    """Build one thematic equation group, return list of (name, Constraint)."""
    return _GROUPS[group](model, sol)
