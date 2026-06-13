"""Query implementations operating on a built+valued Pyomo GTAP model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pyomo.environ import Var, Constraint, value


def _idx_key(idx):
    if idx is None:
        return (None,)
    if isinstance(idx, tuple):
        return idx
    return (idx,)


def extract_solution(model: Any) -> dict:
    """Return {var_name: {idx: value}} for all active Var components."""
    sol: dict[str, dict] = {}
    for v in model.component_objects(Var, active=True):
        cells = {}
        for idx in v:
            try:
                val = v[idx].value
                if val is not None:
                    cells[_idx_key(idx)] = float(val)
            except Exception:
                pass
        if cells:
            sol[v.local_name] = cells
    return sol


def inject_solution(model: Any, solution: dict) -> int:
    """Set Var values from a solution dict. Returns cells set."""
    n = 0
    for name, cells in solution.items():
        comp = getattr(model, name, None)
        if comp is None:
            continue
        for idx, val in cells.items():
            pyidx = (idx[0] if (isinstance(idx, tuple) and len(idx) == 1
                                and idx[0] is None) else idx)
            try:
                item = comp[pyidx]
                if hasattr(item, "fixed") and item.fixed:
                    continue
                item.set_value(float(val))
                n += 1
            except Exception:
                pass
    return n


def query_show(model, var_names, region=None, index_filter=None):
    """Return [{var, idx, value}] for the named vars, optionally filtered."""
    rows = []
    for name in var_names:
        comp = getattr(model, name, None)
        if comp is None:
            continue
        for idx in comp:
            key = _idx_key(idx)
            if region is not None and region not in (str(k) for k in key):
                continue
            if index_filter is not None and index_filter not in (str(k) for k in key):
                continue
            try:
                val = comp[idx].value
            except Exception:
                val = None
            rows.append({"var": name, "idx": key, "value": val})
    return rows


def _family(name: str) -> str:
    for sep in ("[", "("):
        i = name.find(sep)
        if i != -1:
            return name[:i]
    return name


def query_residuals(model, top_n=15, family=None):
    """Return [{eq, idx, resid}] = |body - target|, sorted desc."""
    rows = []
    for c in model.component_objects(Constraint, active=True):
        if family is not None and c.local_name != family:
            continue
        for idx in c:
            con = c[idx]
            try:
                body = value(con.body)
                lo = con.lower
                up = con.upper
                tgt = (value(lo) if lo is not None
                       else (value(up) if up is not None else 0.0))
                rows.append({"eq": c.local_name, "idx": idx,
                             "resid": abs(body - tgt)})
            except Exception:
                pass
    rows.sort(key=lambda r: r["resid"], reverse=True)
    return rows[:top_n]
