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


def _add_raw(model, name, rule_pairs):
    """Attach a NON-log Constraint (lhs == rhs) per key. For eqs Julia writes
    without log() (uepriv, uelas)."""
    idxset = pyo.Set(initialize=[k for k, _, _ in rule_pairs])
    model.add_component(f"_{name}_idx", idxset)
    data = {tuple(k): (lhs, rhs) for k, lhs, rhs in rule_pairs}

    def _rule(m, *key):
        lhs, rhs = data[tuple(key)]
        return lhs == rhs

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

    # e_pca: make price aggregation. esubq==0 → pca == pds (per producing activity);
    # esubq>0 → CES. This dataset (and GTAP standard) has esubq==0.
    pairs = []
    for c in comm:
        for a in acts:
            for r in regs:
                if not _has(sol, "α_qca", (c, a, r)):
                    continue
                if _get(sol, "esubq", (c, r)) == 0:
                    pairs.append(((c, a, r), model.pca[c, a, r], model.pds[c, r]))
                else:
                    members = [aa for aa in acts if _has(sol, "α_qca", (c, aa, r))]
                    prices = [model.pca[c, aa, r] for aa in members]
                    alphas = [_get(sol, "α_pca", (c, aa, r)) for aa in members]
                    i = members.index(a)
                    sigma = 1.0 / _get(sol, "esubq", (c, r))
                    gamma = _get(sol, "γ_pca", (c, r))
                    pairs.append(
                        (
                            (c, a, r),
                            model.qca[c, a, r],
                            _ces_input(model.qc[c, r], prices, alphas, sigma, gamma, i),
                        )
                    )
    if pairs:
        out.append(_add(model, "e_pca", pairs))

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


def _cde_share(alpha, beta, e, u, prices, income, i):
    """Symbolic CDE demand for good i (Pyomo expression). Mirrors Julia cde() x[i].

    x_i = w_i·(p_i/c)^(-α_i) / Σ_j w_j·(p_j/c)^(1-α_j),  w_j = β_j·u^(e_j(1-α_j))(1-α_j)
    """
    w = [
        b * u ** (ej * (1.0 - a)) * (1.0 - a)
        for a, b, ej in zip(alpha, beta, e, strict=True)
    ]
    denom = sum(
        wj * (p / income) ** (1.0 - a)
        for wj, p, a in zip(w, prices, alpha, strict=True)
    )
    return w[i] * (prices[i] / income) ** (-alpha[i]) / denom


def _final_demand(model, sol):
    regs = sol["sets"]["reg"]
    comm = sol["sets"]["comm"]
    out = []

    def qga(c, r):
        return _has(sol, "α_qga", (c, r))

    def qia(c, r):
        return _has(sol, "α_qia", (c, r))

    # e_qpa: private demand (CDE) per good  qpa[c]/pop == cde(...)[c]
    for c in comm:
        pairs = []
        for r in regs:
            alpha = [1.0 - _get(sol, "subpar", (cc, r)) for cc in comm]
            beta = [_get(sol, "β_qpa", (cc, r)) for cc in comm]
            e = [_get(sol, "incpar", (cc, r)) for cc in comm]
            prices = [model.ppa[cc, r] for cc in comm]
            i = comm.index(c)
            income = model.yp[r] / _get(sol, "pop", (r,))
            lhs = model.qpa[c, r] / _get(sol, "pop", (r,))
            rhs = _cde_share(alpha, beta, e, model.up[r], prices, income, i)
            pairs.append(((c, r), lhs, rhs))
        if pairs:
            out.append(_add(model, f"e_qpa_{c}", pairs))

    # e_qpdqpm: Armington split of qpa into {qpd, qpm}
    pairs_d, pairs_m = [], []
    for c in comm:
        for r in regs:
            prices = [model.ppd[c, r], model.ppm[c, r]]
            alphas = [
                _get(sol, "α_qpdqpm", ("dom", c, r)),
                _get(sol, "α_qpdqpm", ("imp", c, r)),
            ]
            sigma = _get(sol, "esubd", (c, r))
            gamma = _get(sol, "γ_qpdqpm", (c, r))
            pairs_d.append(
                (
                    (c, r),
                    model.qpd[c, r],
                    _ces_input(model.qpa[c, r], prices, alphas, sigma, gamma, 0),
                )
            )
            pairs_m.append(
                (
                    (c, r),
                    model.qpm[c, r],
                    _ces_input(model.qpa[c, r], prices, alphas, sigma, gamma, 1),
                )
            )
    out.append(_add(model, "e_qpd", pairs_d))
    out.append(_add(model, "e_qpm", pairs_m))

    # e_ppa: private consumption price index  ppriv·Σqpa == Σ ppa·qpa
    pairs = []
    for r in regs:
        qsum = sum(model.qpa[c, r] for c in comm)
        vsum = sum(model.ppa[c, r] * model.qpa[c, r] for c in comm)
        pairs.append(((r,), model.ppriv[r] * qsum, vsum))
    out.append(_add(model, "e_ppriv", pairs))

    # e_qga: government demand  pga·qga == yg·α_qga
    pairs = []
    for c in comm:
        for r in regs:
            if qga(c, r):
                pairs.append(
                    (
                        (c, r),
                        model.pga[c, r] * model.qga[c, r],
                        model.yg[r] * _get(sol, "α_qga", (c, r)),
                    )
                )
    out.append(_add(model, "e_qga", pairs))

    # e_pgov: gov price index
    pairs = []
    for r in regs:
        cells = [c for c in comm if qga(c, r)]
        qsum = sum(model.qga[c, r] for c in cells)
        vsum = sum(model.qga[c, r] * model.pga[c, r] for c in cells)
        pairs.append(((r,), model.pgov[r] * qsum, vsum))
    out.append(_add(model, "e_pgov", pairs))

    # e_qgdqgm: Armington split of qga into {qgd, qgm}
    pairs_d, pairs_m = [], []
    for c in comm:
        for r in regs:
            if not qga(c, r):
                continue
            prices = [model.pgd[c, r], model.pgm[c, r]]
            alphas = [
                _get(sol, "α_qgdqgm", ("dom", c, r)),
                _get(sol, "α_qgdqgm", ("imp", c, r)),
            ]
            sigma = _get(sol, "esubd", (c, r))
            gamma = _get(sol, "γ_qgdqgm", (c, r))
            pairs_d.append(
                (
                    (c, r),
                    model.qgd[c, r],
                    _ces_input(model.qga[c, r], prices, alphas, sigma, gamma, 0),
                )
            )
            pairs_m.append(
                (
                    (c, r),
                    model.qgm[c, r],
                    _ces_input(model.qga[c, r], prices, alphas, sigma, gamma, 1),
                )
            )
    out.append(_add(model, "e_qgd", pairs_d))
    out.append(_add(model, "e_qgm", pairs_m))

    # e_pga: gov value balance  qga·pga == qgd·pgd + qgm·pgm
    pairs = []
    for c in comm:
        for r in regs:
            if qga(c, r):
                pairs.append(
                    (
                        (c, r),
                        model.qga[c, r] * model.pga[c, r],
                        model.pgd[c, r] * model.qgd[c, r]
                        + model.pgm[c, r] * model.qgm[c, r],
                    )
                )
    out.append(_add(model, "e_pga", pairs))

    # e_qia: investment demand (Leontief, σ=0)  qia[c] == ces(qinv, ..., 0)
    for c in comm:
        pairs = []
        for r in regs:
            if not qia(c, r):
                continue
            members = [cc for cc in comm if qia(cc, r)]
            prices = [model.pia[cc, r] for cc in members]
            alphas = [_get(sol, "α_qia", (cc, r)) for cc in members]
            i = members.index(c)
            gamma = _get(sol, "γ_qia", (r,))
            pairs.append(
                (
                    (c, r),
                    model.qia[c, r],
                    _ces_input(model.qinv[r], prices, alphas, 0.0, gamma, i),
                )
            )
        if pairs:
            out.append(_add(model, f"e_qia_{c}", pairs))

    # e_pinv: investment price index
    pairs = []
    for r in regs:
        cells = [c for c in comm if qia(c, r)]
        qsum = sum(model.qia[c, r] for c in cells)
        vsum = sum(model.pia[c, r] * model.qia[c, r] for c in cells)
        pairs.append(((r,), model.pinv[r] * qsum, vsum))
    out.append(_add(model, "e_pinv", pairs))

    # e_qidqim: Armington split of qia into {qid, qim}
    pairs_d, pairs_m = [], []
    for c in comm:
        for r in regs:
            if not qia(c, r):
                continue
            prices = [model.pid[c, r], model.pim[c, r]]
            alphas = [
                _get(sol, "α_qidqim", ("dom", c, r)),
                _get(sol, "α_qidqim", ("imp", c, r)),
            ]
            sigma = _get(sol, "esubd", (c, r))
            gamma = _get(sol, "γ_qidqim", (c, r))
            pairs_d.append(
                (
                    (c, r),
                    model.qid[c, r],
                    _ces_input(model.qia[c, r], prices, alphas, sigma, gamma, 0),
                )
            )
            pairs_m.append(
                (
                    (c, r),
                    model.qim[c, r],
                    _ces_input(model.qia[c, r], prices, alphas, sigma, gamma, 1),
                )
            )
    out.append(_add(model, "e_qid", pairs_d))
    out.append(_add(model, "e_qim", pairs_m))

    # e_pia: investment value balance  pia·qia == pid·qid + pim·qim
    pairs = []
    for c in comm:
        for r in regs:
            if qia(c, r):
                pairs.append(
                    (
                        (c, r),
                        model.pia[c, r] * model.qia[c, r],
                        model.pid[c, r] * model.qid[c, r]
                        + model.pim[c, r] * model.qim[c, r],
                    )
                )
    out.append(_add(model, "e_pia", pairs))

    return out


def _income(model, sol):
    regs = sol["sets"]["reg"]
    acts = sol["sets"]["acts"]
    comm = sol["sets"]["comm"]
    endw = sol["sets"]["endw"]
    out = []

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

    # e_ppd / e_ppm: private consumer prices  ppd==pds·tpd, ppm==pms·tpm
    pairs_d, pairs_m = [], []
    for c in comm:
        for r in regs:
            pairs_d.append(
                ((c, r), model.ppd[c, r], model.pds[c, r] * T("tpd", (c, r)))
            )
            pairs_m.append(
                ((c, r), model.ppm[c, r], model.pms[c, r] * T("tpm", (c, r)))
            )
    out.append(_add(model, "e_ppd", pairs_d))
    out.append(_add(model, "e_ppm", pairs_m))

    # e_ppa: private value balance  qpa·ppa == ppd·qpd + ppm·qpm
    pairs = []
    for c in comm:
        for r in regs:
            pairs.append(
                (
                    (c, r),
                    model.qpa[c, r] * model.ppa[c, r],
                    model.ppd[c, r] * model.qpd[c, r]
                    + model.ppm[c, r] * model.qpm[c, r],
                )
            )
    out.append(_add(model, "e_ppa", pairs))

    # e_fincome: factor income net of depreciation
    pairs = []
    for r in regs:
        cells = [(e, a) for e in endw for a in acts if evfp(e, a, r)]
        fac = sum(model.peb[e, a, r] * model.qes[e, a, r] for e, a in cells)
        pairs.append(
            ((r,), model.fincome[r], fac - T("δ", (r,)) * model.pinv[r] * model.kb[r])
        )
    out.append(_add(model, "e_fincome", pairs))

    # e_y: regional income = fincome + Σ tax revenue (each instrument, power-1)
    pairs = []
    for r in regs:
        rev = model.fincome[r]
        for c in comm:
            rev = rev + model.qpd[c, r] * model.pds[c, r] * (T("tpd", (c, r)) - 1)
            rev = rev + model.qpm[c, r] * model.pms[c, r] * (T("tpm", (c, r)) - 1)
            if qga(c, r):
                rev = rev + model.qgd[c, r] * model.pds[c, r] * (T("tgd", (c, r)) - 1)
                rev = rev + model.qgm[c, r] * model.pms[c, r] * (T("tgm", (c, r)) - 1)
            if qia(c, r):
                rev = rev + model.qid[c, r] * model.pds[c, r] * (T("tid", (c, r)) - 1)
                rev = rev + model.qim[c, r] * model.pms[c, r] * (T("tim", (c, r)) - 1)
        for c in comm:
            for a in acts:
                if qfa(c, a, r):
                    rev = rev + model.qfd[c, a, r] * model.pfd[c, a, r] / T(
                        "tfd", (c, a, r)
                    ) * (T("tfd", (c, a, r)) - 1)
                    rev = rev + model.qfm[c, a, r] * model.pfm[c, a, r] / T(
                        "tfm", (c, a, r)
                    ) * (T("tfm", (c, a, r)) - 1)
                if qca(c, a, r):
                    rev = rev + model.qca[c, a, r] * model.ps[c, a, r] * (
                        T("to", (c, a, r)) - 1
                    )
        for e in endw:
            for a in acts:
                if evfp(e, a, r):
                    rev = rev + model.qfe[e, a, r] * model.peb[e, a, r] * (
                        T("tfe", (e, a, r)) - 1
                    )
        for c in comm:
            for d in regs:
                if qxs(c, r, d):  # exports from r: pfob/txs revenue
                    rev = rev + model.qxs[c, r, d] * model.pfob[c, r, d] / T(
                        "txs", (c, r, d)
                    ) * (T("txs", (c, r, d)) - 1)
            for s in regs:
                if qxs(c, s, r):  # imports into r: pcif·(tms-1)
                    rev = rev + model.qxs[c, s, r] * model.pcif[c, s, r] * (
                        T("tms", (c, s, r)) - 1
                    )
        pairs.append(((r,), model.y[r], rev))
    out.append(_add(model, "e_y", pairs))

    # e_uepriv (non-log): uepriv = Σ qpa·ppa·incpar / yp
    pairs = []
    for r in regs:
        num = sum(model.qpa[c, r] * model.ppa[c, r] * T("incpar", (c, r)) for c in comm)
        pairs.append(((r,), model.uepriv[r], num / model.yp[r]))
    out.append(_add_raw(model, "e_uepriv", pairs))

    # e_uelas (non-log): uelas = 1 / (σyp/uepriv + σyg + (1-σyp-σyg))
    pairs = []
    for r in regs:
        syp, syg = T("σyp", (r,)), T("σyg", (r,))
        pairs.append(
            (
                (r,),
                model.uelas[r],
                1.0 / (syp / model.uepriv[r] + syg + (1.0 - syp - syg)),
            )
        )
    out.append(_add_raw(model, "e_uelas", pairs))

    # e_yp: household income  yp == y·σyp·uelas/uepriv
    pairs = [
        (
            (r,),
            model.yp[r],
            model.y[r] * T("σyp", (r,)) * model.uelas[r] / model.uepriv[r],
        )
        for r in regs
    ]
    out.append(_add(model, "e_yp", pairs))

    # e_yg: government income  yg == y·σyg·uelas
    pairs = [
        ((r,), model.yg[r], model.y[r] * T("σyg", (r,)) * model.uelas[r]) for r in regs
    ]
    out.append(_add(model, "e_yg", pairs))

    # e_ug: gov utility  ug == yg/pop/pgov
    pairs = [
        ((r,), model.ug[r], model.yg[r] / T("pop", (r,)) / model.pgov[r]) for r in regs
    ]
    out.append(_add(model, "e_ug", pairs))

    # e_p: consumer price index  p == ppriv·σyp + pgov·σyg + psave·(1-σyp-σyg)
    pairs = []
    for r in regs:
        syp, syg = T("σyp", (r,)), T("σyg", (r,))
        pairs.append(
            (
                (r,),
                model.p[r],
                model.ppriv[r] * syp
                + model.pgov[r] * syg
                + model.psave[r] * (1.0 - syp - syg),
            )
        )
    out.append(_add(model, "e_p", pairs))

    # e_u: regional utility  u == y/p/pop
    pairs = [((r,), model.u[r], model.y[r] / model.p[r] / T("pop", (r,))) for r in regs]
    out.append(_add(model, "e_u", pairs))

    return out


def _capital(model, sol, rordelta: int = 1):
    """Capital-account closure. rordelta=1 → capFlex (returns equalize, rore==rorg);
    rordelta=0 → the RORFLEX allocation closure."""
    regs = sol["sets"]["reg"]
    acts = sol["sets"]["acts"]
    endwc = sol["sets"].get("endwc", [])
    out = []

    def T(name, idx):
        return _get(sol, name, idx)

    def evfp(e, a, r):
        return _has(sol, "α_qfe", (e, a, r))

    def kap_supply(r):
        cells = [(e, a) for e in endwc for a in acts if evfp(e, a, r)]
        return cells

    # e_qsave: saving  y·uelas == σyp·y·uelas + σyg·y·uelas + psave·qsave
    pairs = []
    for r in regs:
        syp, syg = T("σyp", (r,)), T("σyg", (r,))
        pairs.append(
            (
                (r,),
                model.y[r] * model.uelas[r],
                syp * model.y[r] * model.uelas[r]
                + syg * model.y[r] * model.uelas[r]
                + model.psave[r] * model.qsave[r],
            )
        )
    out.append(_add(model, "e_qsave", pairs))

    # global net investment Σ_r (qinv - δ·kb)  and Σ_r pinv·net  (Julia sums globally)
    net_global = sum(model.qinv[r] - T("δ", (r,)) * model.kb[r] for r in regs)
    v_net_global = sum(
        model.pinv[r] * (model.qinv[r] - T("δ", (r,)) * model.kb[r]) for r in regs
    )

    # e_psave: saving price (semi-log form from Julia; the sum is GLOBAL)
    pairs = []
    for r in regs:
        rhs = (
            Log(model.pinv[r])
            + sum(
                ((model.qinv[rr] - T("δ", (rr,)) * model.kb[rr]) - model.qsave[rr])
                * Log(model.pinv[rr])
                for rr in regs
            )
            / net_global
        )
        pairs.append(((r,), Log(model.psave[r]), rhs))
    out.append(_add_raw(model, "e_psave", pairs))

    # e_pcgdswld: world price of capital goods (scalar, global sums)
    out.append(
        _add(model, "e_pcgdswld", [((), model.pcgdswld * net_global, v_net_global)])
    )

    # e_walras_sup / e_walras_dem
    out.append(
        _add(
            model,
            "e_walras_sup",
            [((), model.walras_sup, model.pcgdswld * model.globalcgds)],
        )
    )
    out.append(
        _add(
            model,
            "e_walras_dem",
            [
                (
                    (),
                    model.walras_dem,
                    sum(model.psave[r] * model.qsave[r] for r in regs),
                )
            ],
        )
    )

    # e_rorg: world factor price  pfactwld·Σqfe == Σ peb·qfe
    allc = [
        (e, a, r)
        for e in sol["sets"]["endw"]
        for a in acts
        for r in regs
        if evfp(e, a, r)
    ]
    out.append(
        _add(
            model,
            "e_rorg",
            [
                (
                    (),
                    model.pfactwld * sum(model.qfe[e, a, r] for e, a, r in allc),
                    sum(model.peb[e, a, r] * model.qfe[e, a, r] for e, a, r in allc),
                )
            ],
        )
    )

    # e_kb: capital stock  ρ·kb == Σ qe[capital]
    pairs = []
    for r in regs:
        rhs = sum(model.qe[e, r] for e in endwc)
        pairs.append(((r,), T("ρ", (r,)) * model.kb[r], rhs))
    out.append(_add(model, "e_kb", pairs))

    # e_ke: end-of-period capital  ke == qinv + (1-δ)·kb
    pairs = [
        ((r,), model.ke[r], model.qinv[r] + (1.0 - T("δ", (r,))) * model.kb[r])
        for r in regs
    ]
    out.append(_add(model, "e_ke", pairs))

    # e_rore: expected return  α_qinv·rore == rorc / (ke/kb)^rorflex
    pairs = []
    for r in regs:
        pairs.append(
            (
                (r,),
                T("α_qinv", (r,)) * model.rore[r],
                model.rorc[r] / (model.ke[r] / model.kb[r]) ** T("rorflex", (r,)),
            )
        )
    out.append(_add(model, "e_rore", pairs))

    # e_rorc: current return
    pairs = []
    for r in regs:
        cells = kap_supply(r)
        qsum = sum(model.qes[e, a, r] for e, a in cells)
        pairs.append(
            (
                (r,),
                model.rorc[r] * (qsum - T("δ", (r,)) * model.kb[r]),
                qsum * (model.rental[r] / model.pinv[r]),
            )
        )
    out.append(_add(model, "e_rorc", pairs))

    # e_rental: capital rental price
    pairs = []
    for r in regs:
        cells = kap_supply(r)
        qsum = sum(model.qes[e, a, r] for e, a in cells)
        vsum = sum(model.qes[e, a, r] * model.pes[e, a, r] for e, a in cells)
        pairs.append(((r,), qsum * model.rental[r], vsum))
    out.append(_add(model, "e_rental", pairs))

    if rordelta == 1:
        # capFlex: returns equalize  rore == rorg
        out.append(
            _add(model, "e_qinv", [((r,), model.rore[r], model.rorg) for r in regs])
        )
        # globalcgds == Σ net investment
        out.append(
            _add(
                model,
                "e_globalcgds",
                [
                    (
                        (),
                        model.globalcgds,
                        sum(model.qinv[r] - T("δ", (r,)) * model.kb[r] for r in regs),
                    )
                ],
            )
        )
    else:
        # RORFLEX: investment allocated by share; rorg clears
        out.append(
            _add(
                model,
                "e_qinv",
                [
                    (
                        (r,),
                        model.qinv[r],
                        T("α_qinv", (r,)) * model.globalcgds
                        + T("δ", (r,)) * model.kb[r],
                    )
                    for r in regs
                ],
            )
        )
        out.append(
            _add(
                model,
                "e_globalcgds",
                [
                    (
                        (),
                        model.rorg
                        * sum(model.qinv[r] - T("δ", (r,)) * model.kb[r] for r in regs),
                        sum(
                            (model.qinv[r] - T("δ", (r,)) * model.kb[r]) * model.rore[r]
                            for r in regs
                        ),
                    )
                ],
            )
        )

    return out


_GROUPS = {
    "production": _production,
    "factors": _factors,
    "trade": _trade,
    "final_demand": _final_demand,
    "income": _income,
    "capital": _capital,
}


def build_group(model, sol: dict[str, Any], group: str, **kwargs):
    """Build one thematic equation group, return list of (name, Constraint).

    Extra kwargs pass through to the group builder (e.g. rordelta for capital).
    """
    return _GROUPS[group](model, sol, **kwargs)
