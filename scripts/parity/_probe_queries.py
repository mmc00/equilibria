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
