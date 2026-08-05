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
    """Attach a scalar Constraint per (key, lhs, rhs) and return (name, comp).

    Keys are normalized so a 1-tuple ('r',) and Pyomo's bare-element indexing
    ('r') resolve to the same entry.
    """
    idxset = pyo.Set(initialize=[k for k, _, _ in rule_pairs])
    model.add_component(f"_{name}_idx", idxset)
    data = {tuple(k): (lhs, rhs) for k, lhs, rhs in rule_pairs}

    def _rule(m, *key):
        lhs, rhs = data[tuple(key)]
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


def _factors(model, sol):
    regs = sol["sets"]["reg"]
    acts = sol["sets"]["acts"]
    endw = sol["sets"]["endw"]
    endwm = sol["sets"].get("endwm", [])
    endws = sol["sets"].get("endws", [])
    endwf = sol["sets"].get("endwf", [])
    out = []

    def evfp(e, a, r):
        return _has(sol, "α_qfe", (e, a, r))

    # e_peb: factor use == endowment supply  qfe == qes
    pairs = []
    for e in endw:
        for a in acts:
            for r in regs:
                if evfp(e, a, r):
                    pairs.append(((e, a, r), model.qfe[e, a, r], model.qes[e, a, r]))
    if pairs:
        out.append(_add(model, "e_peb", pairs))

    # e_pfe: factor price w/ use tax  pfe == peb·tfe
    pairs = []
    for e in endw:
        for a in acts:
            for r in regs:
                if evfp(e, a, r):
                    pairs.append(
                        (
                            (e, a, r),
                            model.pfe[e, a, r],
                            model.peb[e, a, r] * _get(sol, "tfe", (e, a, r)),
                        )
                    )
    if pairs:
        out.append(_add(model, "e_pfe", pairs))

    # e_pes: net-of-income-tax factor price  peb == pes·tinc
    pairs = []
    for e in endw:
        for a in acts:
            for r in regs:
                if evfp(e, a, r):
                    pairs.append(
                        (
                            (e, a, r),
                            model.peb[e, a, r],
                            model.pes[e, a, r] * _get(sol, "tinc", (e, a, r)),
                        )
                    )
    if pairs:
        out.append(_add(model, "e_pes", pairs))

    # e_pfactor: regional factor price index
    pairs = []
    for r in regs:
        cells = [(e, a) for e in endw for a in acts if evfp(e, a, r)]
        qsum = sum(model.qfe[e, a, r] for e, a in cells)
        vsum = sum(model.qfe[e, a, r] * model.peb[e, a, r] for e, a in cells)
        pairs.append(((r,), model.pfactor[r] * qsum, vsum))
    if pairs:
        out.append(_add(model, "e_pfactor", pairs))

    # e_pe1: mobile-factor market clearing  qe == Σ_a qfe
    pairs = []
    for e in endwm:
        for r in regs:
            cells = [a for a in acts if evfp(e, a, r)]
            if not cells:
                continue
            pairs.append(
                ((e, r), model.qe[e, r], sum(model.qfe[e, a, r] for a in cells))
            )
    if pairs:
        out.append(_add(model, "e_pe1", pairs))

    # e_qes1: mobile factor — one price  pes == pe
    pairs = []
    for e in endwm:
        for a in acts:
            for r in regs:
                if evfp(e, a, r):
                    pairs.append(((e, a, r), model.pes[e, a, r], model.pe[e, r]))
    if pairs:
        out.append(_add(model, "e_qes1", pairs))

    # e_qes2: sluggish factor — CET supply across acts (etrae<0)
    for a in acts:
        pairs = []
        for e in endws:
            for r in regs:
                if not evfp(e, a, r):
                    continue
                members = [aa for aa in acts if evfp(e, aa, r)]
                prices = [model.pes[e, aa, r] for aa in members]
                alphas = [_get(sol, "α_qes2", (e, aa, r)) for aa in members]
                i = members.index(a)
                sigma = _get(sol, "etrae", (e, r))
                gamma = _get(sol, "γ_qes2", (e, r))
                pairs.append(
                    (
                        (e, a, r),
                        model.qes[e, a, r],
                        _ces_input(model.qe[e, r], prices, alphas, sigma, gamma, i),
                    )
                )
        if pairs:
            out.append(_add(model, f"e_qes2_{a}", pairs))

    # e_pe2: sluggish factor value balance  pe·qe == Σ pes·qes
    pairs = []
    for e in endws:
        for r in regs:
            cells = [a for a in acts if evfp(e, a, r)]
            if not cells:
                continue
            rhs = sum(model.pes[e, a, r] * model.qes[e, a, r] for a in cells)
            pairs.append(((e, r), model.pe[e, r] * model.qe[e, r], rhs))
    if pairs:
        out.append(_add(model, "e_pe2", pairs))

    # e_qes3: fixed factor  qes == qesf
    pairs = []
    for e in endwf:
        for a in acts:
            for r in regs:
                if evfp(e, a, r):
                    pairs.append(((e, a, r), model.qes[e, a, r], model.qesf[e, a, r]))
    if pairs:
        out.append(_add(model, "e_qes3", pairs))

    return out


def _trade(model, sol):
    regs = sol["sets"]["reg"]
    acts = sol["sets"]["acts"]
    comm = sol["sets"]["comm"]
    marg = sol["sets"].get("marg", [])
    out = []

    def qfa(c, a, r):
        return _has(sol, "α_qfa", (c, a, r))

    def qga(c, r):
        return _has(sol, "α_qga", (c, r))

    def qia(c, r):
        return _has(sol, "α_qia", (c, r))

    def qxs(c, s, d):
        return _has(sol, "α_qxs", (c, s, d))

    def vtwr(m, c, s, d):
        return _has(sol, "α_qtmfsd", (m, c, s, d))

    def vtwr_sum(c, s, d):
        return any(vtwr(m, c, s, d) for m in marg)

    # e_qms: import aggregate = Σ firm imports + private + gov + invest imports
    pairs = []
    for c in comm:
        for r in regs:
            rhs = (
                sum(model.qfm[c, a, r] for a in acts if qfa(c, a, r)) + model.qpm[c, r]
            )
            if qga(c, r):
                rhs = rhs + model.qgm[c, r]
            if qia(c, r):
                rhs = rhs + model.qim[c, r]
            pairs.append(((c, r), model.qms[c, r], rhs))
    if pairs:
        out.append(_add(model, "e_qms", pairs))

    # e_qxs: CES sourcing of qms[c,d] across origins s (Armington, esubm)
    for s in regs:
        pairs = []
        for c in comm:
            for d in regs:
                if not qxs(c, s, d):
                    continue
                origins = [ss for ss in regs if qxs(c, ss, d)]
                prices = [model.pmds[c, ss, d] for ss in origins]
                alphas = [_get(sol, "α_qxs", (c, ss, d)) for ss in origins]
                i = origins.index(s)
                sigma = _get(sol, "esubm", (c, d))
                gamma = _get(sol, "γ_qxs", (c, d))
                pairs.append(
                    (
                        (c, s, d),
                        model.qxs[c, s, d],
                        _ces_input(model.qms[c, d], prices, alphas, sigma, gamma, i),
                    )
                )
        if pairs:
            out.append(_add(model, f"e_qxs_{s}", pairs))

    # e_pms: import value balance  pms·qms == Σ_s pmds·qxs
    pairs = []
    for c in comm:
        for d in regs:
            origins = [s for s in regs if qxs(c, s, d)]
            if not origins:
                continue
            rhs = sum(model.pmds[c, s, d] * model.qxs[c, s, d] for s in origins)
            pairs.append(((c, d), model.pms[c, d] * model.qms[c, d], rhs))
    if pairs:
        out.append(_add(model, "e_pms", pairs))

    # e_pfob: FOB price  pfob == pds·tx·txs (export taxes, multiplicative)
    pairs = []
    for c in comm:
        for s in regs:
            for d in regs:
                if qxs(c, s, d):
                    pairs.append(
                        (
                            (c, s, d),
                            model.pfob[c, s, d],
                            model.pds[c, s]
                            * _get(sol, "tx", (c, s))
                            * _get(sol, "txs", (c, s, d)),
                        )
                    )
    if pairs:
        out.append(_add(model, "e_pfob", pairs))

    # e_pcif: CIF value  pcif·qxs == pfob·qxs + margins
    pairs = []
    for c in comm:
        for s in regs:
            for d in regs:
                if not qxs(c, s, d):
                    continue
                rhs = model.pfob[c, s, d] * model.qxs[c, s, d]
                if vtwr_sum(c, s, d):
                    rhs = rhs + model.ptrans[c, s, d] * sum(
                        model.qtmfsd[m, c, s, d] for m in marg if vtwr(m, c, s, d)
                    )
                pairs.append(((c, s, d), model.pcif[c, s, d] * model.qxs[c, s, d], rhs))
    if pairs:
        out.append(_add(model, "e_pcif", pairs))

    # e_pmds: delivered import price  pmds == pcif·tm·tms (import tariff — the shock)
    pairs = []
    for c in comm:
        for s in regs:
            for d in regs:
                if qxs(c, s, d):
                    pairs.append(
                        (
                            (c, s, d),
                            model.pmds[c, s, d],
                            model.pcif[c, s, d]
                            * _get(sol, "tm", (c, d))
                            * _get(sol, "tms", (c, s, d)),
                        )
                    )
    if pairs:
        out.append(_add(model, "e_pmds", pairs))

    # e_qtmfsd: margin demand proportional to trade flow
    pairs = []
    for m in marg:
        for c in comm:
            for s in regs:
                for d in regs:
                    if vtwr(m, c, s, d):
                        pairs.append(
                            (
                                (m, c, s, d),
                                model.qtmfsd[m, c, s, d],
                                _get(sol, "α_qtmfsd", (m, c, s, d))
                                * model.qxs[c, s, d],
                            )
                        )
    if pairs:
        out.append(_add(model, "e_qtmfsd", pairs))

    # e_ptrans: transport price index per route
    pairs = []
    for c in comm:
        for s in regs:
            for d in regs:
                if not vtwr_sum(c, s, d):
                    continue
                ms = [m for m in marg if vtwr(m, c, s, d)]
                qsum = sum(model.qtmfsd[m, c, s, d] for m in ms)
                vsum = sum(model.qtmfsd[m, c, s, d] * model.pt[m] for m in ms)
                pairs.append(((c, s, d), model.ptrans[c, s, d] * qsum, vsum))
    if pairs:
        out.append(_add(model, "e_ptrans", pairs))

    # e_qtm: total margin m demand
    pairs = []
    for m in marg:
        cells = [
            (c, s, d) for c in comm for s in regs for d in regs if vtwr(m, c, s, d)
        ]
        rhs = sum(model.qtmfsd[m, c, s, d] for c, s, d in cells)
        pairs.append(((m,), model.qtm[m], rhs))
    if pairs:
        out.append(_add(model, "e_qtm", pairs))

    # e_qst: CES supply of margin m from regions (pds)
    for r in regs:
        pairs = []
        for m in marg:
            if not _has(sol, "α_qst", (m, r)):
                continue
            members = [rr for rr in regs if _has(sol, "α_qst", (m, rr))]
            prices = [model.pds[m, rr] for rr in members]
            alphas = [_get(sol, "α_qst", (m, rr)) for rr in members]
            i = members.index(r)
            sigma = _get(sol, "esubs", (m,))
            gamma = _get(sol, "γ_qst", (m,))
            pairs.append(
                (
                    (m, r),
                    model.qst[m, r],
                    _ces_input(model.qtm[m], prices, alphas, sigma, gamma, i),
                )
            )
        if pairs:
            out.append(_add(model, f"e_qst_{r}", pairs))

    # e_pt: margin value balance  pt·qtm == Σ pds·qst
    pairs = []
    for m in marg:
        members = [r for r in regs if _has(sol, "α_qst", (m, r))]
        rhs = sum(model.pds[m, r] * model.qst[m, r] for r in members)
        pairs.append(((m,), model.pt[m] * model.qtm[m], rhs))
    if pairs:
        out.append(_add(model, "e_pt", pairs))

    # e_qds: domestic aggregate = Σ firm dom + private + gov + invest domestic
    pairs = []
    for c in comm:
        for r in regs:
            rhs = (
                sum(model.qfd[c, a, r] for a in acts if qfa(c, a, r)) + model.qpd[c, r]
            )
            if qga(c, r):
                rhs = rhs + model.qgd[c, r]
            if qia(c, r):
                rhs = rhs + model.qid[c, r]
            pairs.append(((c, r), model.qds[c, r], rhs))
    if pairs:
        out.append(_add(model, "e_qds", pairs))

    # e_pds: market clearing  qc == qds + exports + margin supply
    pairs = []
    for c in comm:
        for r in regs:
            rhs = model.qds[c, r] + sum(
                model.qxs[c, r, d] for d in regs if qxs(c, r, d)
            )
            if c in marg:
                rhs = rhs + model.qst[c, r]
            pairs.append(((c, r), model.qc[c, r], rhs))
    if pairs:
        out.append(_add(model, "e_pds", pairs))

    # tax links: pfd/pfm (firm), pgd/pgm (gov), pid/pim (invest) — multiplicative
    def _taxlink(name, pvar, base, tax, mask):
        pairs = []
        for c in comm:
            if name in ("e_pfd", "e_pfm"):
                for a in acts:
                    for r in regs:
                        if qfa(c, a, r):
                            pairs.append(
                                (
                                    (c, a, r),
                                    pvar[c, a, r],
                                    base(c, r) * _get(sol, tax, (c, a, r)),
                                )
                            )
            else:
                for r in regs:
                    if mask(c, r):
                        pairs.append(
                            ((c, r), pvar[c, r], base(c, r) * _get(sol, tax, (c, r)))
                        )
        if pairs:
            out.append(_add(model, name, pairs))

    _taxlink("e_pfd", model.pfd, lambda c, r: model.pds[c, r], "tfd", qfa)
    _taxlink("e_pfm", model.pfm, lambda c, r: model.pms[c, r], "tfm", qfa)
    _taxlink("e_pgd", model.pgd, lambda c, r: model.pds[c, r], "tgd", qga)
    _taxlink("e_pgm", model.pgm, lambda c, r: model.pms[c, r], "tgm", qga)
    _taxlink("e_pid", model.pid, lambda c, r: model.pds[c, r], "tid", qia)
    _taxlink("e_pim", model.pim, lambda c, r: model.pms[c, r], "tim", qia)

    return out


_GROUPS = {
    "production": _production,
    "factors": _factors,
    "trade": _trade,
}


def build_group(model, sol: dict[str, Any], group: str):
    """Build one thematic equation group, return list of (name, Constraint)."""
    return _GROUPS[group](model, sol)
