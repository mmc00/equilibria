"""gtap_julia variables — the ~97 Julia GTAPv7 model variables with Julia's
positive lower bounds.

Quantities get a strictly-positive floor (`q_min`) so they can never reach the
exact zero that degenerates a levels Jacobian and breaks `log()`; prices get
`p_min`; taxes are multiplicative POWERS with floor `t_min` (>0) — the property
that keeps every logged term in-domain. `qsave` alone may go negative. Bounds are
transcribed from Julia `generate_starting_values.jl`.
"""

from __future__ import annotations

import pyomo.environ as pyo

# Julia bound magnitudes (generate_starting_values.jl)
Q_MIN = 1e-8
Q_MAX = 1e12
P_MIN = 1e-8
P_MAX = 1e12
T_MIN = 1e-8
T_MAX = 1e2
Y_MIN = 1e-8
Y_MAX = 1e12

# (name, dims-tuple-of-set-names, kind). dims=() → scalar.
# kind: q=quantity, p=price, t=tax-power, y=income, free=unbounded-ish, qsave=neg-ok
_VARS: list[tuple[str, tuple[str, ...], str]] = [
    ("pop", ("reg",), "q"),
    ("qint", ("acts", "reg"), "q"),
    ("pint", ("acts", "reg"), "p"),
    ("qva", ("acts", "reg"), "q"),
    ("pva", ("acts", "reg"), "p"),
    ("qo", ("acts", "reg"), "q"),
    ("po", ("acts", "reg"), "p"),
    ("qfa", ("comm", "acts", "reg"), "q"),
    ("pfa", ("comm", "acts", "reg"), "p"),
    ("qfe", ("endw", "acts", "reg"), "q"),
    ("pfe", ("endw", "acts", "reg"), "p"),
    ("tfe", ("endw", "acts", "reg"), "t"),
    ("qfd", ("comm", "acts", "reg"), "q"),
    ("pfd", ("comm", "acts", "reg"), "p"),
    ("qfm", ("comm", "acts", "reg"), "q"),
    ("pfm", ("comm", "acts", "reg"), "p"),
    ("qca", ("comm", "acts", "reg"), "q"),
    ("pca", ("comm", "acts", "reg"), "p"),
    ("ps", ("comm", "acts", "reg"), "p"),
    ("qc", ("comm", "reg"), "q"),
    ("pds", ("comm", "reg"), "p"),
    ("to", ("comm", "acts", "reg"), "t"),
    ("peb", ("endw", "acts", "reg"), "p"),
    ("qes", ("endw", "acts", "reg"), "q"),
    ("pfactor", ("reg",), "p"),
    ("fincome", ("reg",), "y"),
    ("y", ("reg",), "y"),
    ("p", ("reg",), "p"),
    ("u", ("reg",), "q"),
    ("ug", ("reg",), "q"),
    ("yp", ("reg",), "y"),
    ("up", ("reg",), "q"),
    ("uelas", ("reg",), "free"),
    ("uepriv", ("reg",), "free"),
    ("ppa", ("comm", "reg"), "p"),
    ("qpa", ("comm", "reg"), "q"),
    ("ppd", ("comm", "reg"), "p"),
    ("qpd", ("comm", "reg"), "q"),
    ("ppm", ("comm", "reg"), "p"),
    ("qpm", ("comm", "reg"), "q"),
    ("ppriv", ("reg",), "p"),
    ("yg", ("reg",), "y"),
    ("pgov", ("reg",), "p"),
    ("pga", ("comm", "reg"), "p"),
    ("qga", ("comm", "reg"), "q"),
    ("pgd", ("comm", "reg"), "p"),
    ("qgd", ("comm", "reg"), "q"),
    ("pgm", ("comm", "reg"), "p"),
    ("qgm", ("comm", "reg"), "q"),
    ("qsave", ("reg",), "qsave"),
    ("psave", ("reg",), "p"),
    ("pia", ("comm", "reg"), "p"),
    ("qia", ("comm", "reg"), "q"),
    ("pid", ("comm", "reg"), "p"),
    ("qid", ("comm", "reg"), "q"),
    ("pim", ("comm", "reg"), "p"),
    ("qim", ("comm", "reg"), "q"),
    ("qinv", ("reg",), "q"),
    ("pinv", ("reg",), "p"),
    ("qms", ("comm", "reg"), "q"),
    ("qxs", ("comm", "reg", "reg"), "q"),
    ("pmds", ("comm", "reg", "reg"), "p"),
    ("pms", ("comm", "reg"), "p"),
    ("ptrans", ("comm", "reg", "reg"), "p"),
    ("qtmfsd", ("marg", "comm", "reg", "reg"), "q"),
    ("pt", ("marg",), "p"),
    ("qtm", ("marg",), "q"),
    ("qst", ("marg", "reg"), "q"),
    ("txs", ("comm", "reg", "reg"), "t"),
    ("tx", ("comm", "reg"), "t"),
    ("pfob", ("comm", "reg", "reg"), "p"),
    ("pcif", ("comm", "reg", "reg"), "p"),
    ("tms", ("comm", "reg", "reg"), "t"),
    ("tm", ("comm", "reg"), "t"),
    ("qds", ("comm", "reg"), "q"),
    ("tfd", ("comm", "acts", "reg"), "t"),
    ("tfm", ("comm", "acts", "reg"), "t"),
    ("tpd", ("comm", "reg"), "t"),
    ("tpm", ("comm", "reg"), "t"),
    ("tgd", ("comm", "reg"), "t"),
    ("tgm", ("comm", "reg"), "t"),
    ("tid", ("comm", "reg"), "t"),
    ("tim", ("comm", "reg"), "t"),
    ("pes", ("endw", "acts", "reg"), "p"),
    ("pe", ("endwms", "reg"), "p"),
    ("qe", ("endwms", "reg"), "q"),
    ("qesf", ("endwf", "acts", "reg"), "q"),
    ("tinc", ("endw", "acts", "reg"), "t"),
    ("globalcgds", (), "q"),
    ("pcgdswld", (), "p"),
    ("walras_sup", (), "q"),
    ("walras_dem", (), "q"),
    ("pfactwld", (), "p"),
    ("kb", ("reg",), "q"),
    ("ke", ("reg",), "q"),
    ("rorg", (), "p"),
    ("rore", ("reg",), "p"),
    ("rorc", ("reg",), "p"),
    ("rental", ("reg",), "p"),
]

_BOUNDS = {
    "q": (Q_MIN, Q_MAX),
    "p": (P_MIN, P_MAX),
    "t": (T_MIN, T_MAX),
    "y": (Y_MIN, Y_MAX),
    "qsave": (-Q_MAX, Q_MAX),
    "free": (0.0, Q_MAX),
}


def build_variables(model, sets: dict[str, list[str]]) -> None:
    """Attach every gtap_julia Var to `model` with Julia's bounds + init 1.01."""
    # Pyomo Sets (only those actually used as dims)
    used = {d for _, dims, _ in _VARS for d in dims}
    for sname in used:
        if not hasattr(model, sname):
            members = sets.get(sname, [])
            model.add_component(sname, pyo.Set(initialize=members))

    for name, dims, kind in _VARS:
        lo, hi = _BOUNDS[kind]
        if dims:
            index_sets = [getattr(model, d) for d in dims]
            model.add_component(
                name,
                pyo.Var(*index_sets, bounds=(lo, hi), initialize=1.01),
            )
        else:
            model.add_component(name, pyo.Var(bounds=(lo, hi), initialize=1.01))
