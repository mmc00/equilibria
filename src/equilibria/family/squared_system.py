"""Incidence of one period read from the family backend's Jacobian pattern.

``prep_square.square`` pairs rows with columns from the variables each row mentions.
Pyomo supplies that incidence as strings; here it comes from the J pattern of the
backend's recorded rows, so the period prep needs no Pyomo model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from equilibria.family.layout import VarLayout


def _all_free_layout(backend: Any, state: Any) -> VarLayout:
    """Every variable cell a column, Parameters fixed slots (the pattern is the same
    whichever cells a closure fixes later)."""
    values, free = {}, {}
    for v in backend.var_domains:
        values[v] = np.asarray(state.values[v], dtype=float)
        free[v] = np.ones(values[v].shape, dtype=bool)
    for p in backend.param_domains:
        values[p] = state.find_component(p).data
        free[p] = np.zeros(values[p].shape, dtype=bool)
    domains = {**backend.var_domains, **backend.param_domains}
    return VarLayout.from_state(state.sets, domains, values, free)


def _with_period(name: str, period: str) -> str:
    """``eq[a,b]`` → ``eq[a,b,t]``; ``eq`` → ``eq[t]``."""
    if not name.endswith("]"):
        return f"{name}[{period}]"
    return f"{name[:-1]},{period}]"


def period_incidence(backend: Any, state: Any, period: str) -> dict[str, list[str]]:
    """For every recorded row ``eq[idx]``, the cells ``v[idx',period]`` in its J pattern.

    Keys and columns carry the period suffix (``eq_xp[USA,agr,check]``), the format
    ``square(incidence=...)`` expects. All variable cells are columns, so structural
    zeros stay in the pattern. Rows the backend does not record are absent.
    """
    layout = _all_free_layout(backend, state)
    system = backend.system(layout)
    J = system.jac(layout.free_values).tocsr()
    # Column names carry the period; layout.col_names is in column order.
    cols = [_with_period(n, period) for n in layout.col_names]
    rows = [_with_period(n, period) for n in system.row_names]
    out: dict[str, list[str]] = {}
    for i, name in enumerate(rows):
        lo, hi = J.indptr[i], J.indptr[i + 1]
        out[name] = [cols[j] for j in sorted(J.indices[lo:hi])]
    return out


__all__ = ["period_incidence"]
