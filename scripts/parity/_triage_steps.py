"""Three triage steps for closing parity gaps: locate → isolate → trace."""
from __future__ import annotations

from typing import Any


def _cell_diverges(py_val: float, gams_val: float, tol_rel: float, tol_abs: float) -> bool:
    abs_err = abs(py_val - gams_val)
    if abs_err <= tol_abs:
        return False
    denom = max(abs(gams_val), 1e-30)
    rel_err = abs_err / denom
    return rel_err > tol_rel


def _get_py_value(py_var: Any, key: tuple) -> float | None:
    try:
        item = py_var[key]
    except (KeyError, IndexError, ValueError):
        return None
    if hasattr(item, "value"):
        v = item.value
        return float(v) if v is not None else None
    try:
        return float(item)
    except (TypeError, ValueError):
        return None


def step_locate(
    adapter,
    model,
    gams_reference: dict[str, dict[tuple, float]],
    top_n: int = 10,
    tol_rel: float = 1e-3,
    tol_abs: float = 1e-6,
) -> list[dict]:
    """Step 1: list top-N variables with the most diverging cells."""
    rows: list[dict] = []
    for gams_name, gams_cells in gams_reference.items():
        py_var, py_name = adapter.find_py_var(model, gams_name)
        if py_var is None:
            continue
        cells = match = diverge = 0
        max_rel = 0.0
        worst_key = None
        for key, g_val in gams_cells.items():
            cells += 1
            p_val = _get_py_value(py_var, key)
            if p_val is None:
                continue
            if _cell_diverges(p_val, g_val, tol_rel, tol_abs):
                diverge += 1
                denom = max(abs(g_val), 1e-30)
                rel = abs(p_val - g_val) / denom
                if rel > max_rel:
                    max_rel = rel
                    worst_key = key
            else:
                match += 1
        if diverge == 0:
            continue
        rows.append({
            "gams_var": gams_name, "py_var": py_name,
            "cells": cells, "match": match, "diverge": diverge,
            "max_rel_err": max_rel, "worst_key": worst_key,
        })
    rows.sort(key=lambda r: r["diverge"], reverse=True)
    return rows[:top_n]


def step_isolate(
    adapter,
    model,
    gams_reference: dict[str, dict[tuple, float]],
    gams_var: str,
    tol_rel: float = 1e-3,
    tol_abs: float = 1e-6,
) -> dict | None:
    """Step 2: return the worst diverging cell of `gams_var` or None if all match."""
    if gams_var not in gams_reference:
        return None
    py_var, _ = adapter.find_py_var(model, gams_var)
    if py_var is None:
        return None
    worst = None
    worst_rel = 0.0
    for key, g_val in gams_reference[gams_var].items():
        p_val = _get_py_value(py_var, key)
        if p_val is None:
            continue
        if not _cell_diverges(p_val, g_val, tol_rel, tol_abs):
            continue
        abs_err = abs(p_val - g_val)
        denom = max(abs(g_val), 1e-30)
        rel = abs_err / denom
        if rel > worst_rel:
            worst_rel = rel
            worst = {
                "key": key, "py_val": p_val, "gams_val": g_val,
                "abs_err": abs_err, "rel_err": rel,
            }
    return worst


def step_trace(model, top_n: int = 20) -> list[dict]:
    """Step 3: return the top-N constraint residuals at the model's current values."""
    from pyomo.environ import Constraint, value as _pyomo_value

    out: list[dict] = []
    for component in model.component_objects(Constraint, active=True):
        cname = component.name
        try:
            iterator = list(component.items()) if component.is_indexed() else [(None, component)]
        except Exception:
            continue
        for index, con in iterator:
            if not con.active:
                continue
            try:
                body_val = _pyomo_value(con.body)
                target = _pyomo_value(con.upper) if con.upper is not None else (
                    _pyomo_value(con.lower) if con.lower is not None else 0.0
                )
                residual = abs(body_val - target)
                out.append({
                    "name": cname, "index": index,
                    "residual": residual, "body": body_val, "target": target,
                })
            except Exception:
                continue
    out.sort(key=lambda r: r["residual"], reverse=True)
    return out[:top_n]
