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
    # _DerivedVar (_diff_core) expects the raw tuple key — don't normalize
    _is_derived = type(py_var).__name__ == "_DerivedVar"
    if _is_derived:
        try:
            item = py_var[key]
            if hasattr(item, "value"):
                v = item.value
                return float(v) if v is not None else None
            return float(item)
        except Exception:
            return None

    # Scalars: GAMS strips to () after split_t; Pyomo scalars use key=None
    normalized_key: Any = key
    if isinstance(key, tuple):
        if len(key) == 0:
            normalized_key = None
        elif len(key) == 1:
            normalized_key = key[0]
    try:
        item = py_var[normalized_key]
    except (KeyError, IndexError, ValueError):
        return None
    if hasattr(item, "value"):
        v = item.value
        return float(v) if v is not None else None
    # Pyomo Expression objects (ExpressionData) have no .value; use pyo_value()
    try:
        from pyomo.environ import value as _pyo_val
        return float(_pyo_val(item))
    except Exception:
        pass
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


def step_check_warmstart(
    adapter,
    model,
    gams_reference: dict[str, dict[tuple, float]],
    top_n: int = 20,
    tol_rel: float = 1e-3,
    tol_abs: float = 1e-6,
) -> dict:
    """Check warm-start quality before the solver runs.

    Returns a dict with three sections:
      'equation_residuals': top-N active constraints with the largest residuals
          at the current (warm-started) model values.  A large residual here
          means the warm-start placed the model far from the GAMS equilibrium
          for that equation — PATH will have to cross a large gradient to get
          there, risking landing in a different basin.
      'var_gaps': variables where the warm-started value differs from the GAMS
          reference by more than tol.  This pinpoints *why* an equation has a
          large residual (e.g. pfy set to base value instead of GAMS check value
          because the warm-start key normalization failed).
      'seeding_coverage': for every Pyomo variable that has a GAMS reference,
          how many cells were seeded vs. total available.  Rows with seeded=0
          mean the warm-start loop silently skipped the variable entirely —
          most commonly because the GDX uses a different name (e.g. 'xa' vs
          'xaa') or the key normalization failed.  These are the bugs that
          cause PATH to start from a wrong basin without any explicit error.
    """
    from pyomo.environ import Constraint, value as _pyomo_value

    # --- equation residuals at warm-start point ---
    eq_residuals: list[dict] = []
    for component in model.component_objects(Constraint, active=True):
        cname = component.name
        try:
            items = list(component.items()) if component.is_indexed() else [(None, component)]
        except Exception:
            continue
        for index, con in items:
            if not con.active:
                continue
            try:
                body_val = _pyomo_value(con.body)
                if con.upper is not None:
                    target = float(_pyomo_value(con.upper))
                elif con.lower is not None:
                    target = float(_pyomo_value(con.lower))
                else:
                    continue
                residual = abs(body_val - target)
                if residual > tol_abs:
                    eq_residuals.append({
                        "name": cname, "index": index,
                        "residual": residual, "body": body_val, "target": target,
                    })
            except Exception:
                continue
    eq_residuals.sort(key=lambda r: r["residual"], reverse=True)

    # --- var gaps: warm-started value vs GAMS reference ---
    var_gaps: list[dict] = []
    for gams_name, gams_cells in gams_reference.items():
        py_var, py_name = adapter.find_py_var(model, gams_name)
        if py_var is None:
            continue
        not_set = 0
        found = 0
        for key, g_val in gams_cells.items():
            p_val = _get_py_value(py_var, key)
            if p_val is None:
                not_set += 1
                continue
            found += 1
            if _cell_diverges(p_val, g_val, tol_rel, tol_abs):
                denom = max(abs(g_val), 1e-30)
                var_gaps.append({
                    "gams_var": gams_name, "py_var": py_name,
                    "key": key, "py_val": p_val, "gams_val": g_val,
                    "rel_err": abs(p_val - g_val) / denom,
                    "not_set": False,
                })
        # Only report NOT FOUND when *no* cells were found (true key mismatch).
        # Partial not_set means GAMS has extra aggregated keys (e.g. pa[r,i] vs pa[r,i,aa]) — benign.
        if not_set and found == 0:
            var_gaps.append({
                "gams_var": gams_name, "py_var": py_name,
                "key": None, "py_val": None, "gams_val": None,
                "rel_err": float("inf"), "not_set": True,
                "_not_set_count": not_set,
            })
    var_gaps.sort(key=lambda r: r["rel_err"], reverse=True)

    # --- seeding coverage: per-variable match rate of warm-start values vs GAMS ref ---
    # "Matched" = warm-start value agrees with GAMS reference within 5% (loose tol).
    # "Diverged" = warm-start value differs substantially (likely not seeded at all).
    # A variable with matched=0/N means the warm-start loop silently skipped it —
    # GDX name mismatch ('xa' vs 'xaa') or key normalization failure.

    # Step 1: collect (py_var_obj, py_name, gams_cells) per unique Python variable
    _seen_py: dict[str, tuple] = {}  # py_name → (py_var, merged_gams_cells)
    for gn, gcells in gams_reference.items():
        pv, pname = adapter.find_py_var(model, gn)
        if pv is None or pname is None:
            continue
        if pname not in _seen_py:
            _seen_py[pname] = (pv, {})
        _seen_py[pname][1].update(gcells)

    seeding_coverage: list[dict] = []
    for py_name, (py_var, gcells) in _seen_py.items():
        matched = diverged = skipped = 0
        for key, g_val in gcells.items():
            p_val = _get_py_value(py_var, key)
            if p_val is None:
                skipped += 1
                continue
            if _cell_diverges(p_val, g_val, tol_rel=0.05, tol_abs=tol_abs):
                diverged += 1
            else:
                matched += 1
        total = matched + diverged + skipped
        if total == 0:
            continue
        seeding_coverage.append({
            "py_var": py_name,
            "total": total,
            "matched": matched,
            "diverged": diverged,
            "skipped": skipped,
            "frac": matched / (matched + diverged) if (matched + diverged) > 0 else 0.0,
        })
    seeding_coverage.sort(key=lambda r: r["frac"])

    return {
        "equation_residuals": eq_residuals[:top_n],
        "var_gaps": var_gaps[:top_n],
        "seeding_coverage": seeding_coverage,
    }


def step_check_solution(model, top_n: int = 20, tol_abs: float = 1e-6) -> list[dict]:
    """Check post-solve equation residuals.

    Evaluates every active constraint at the current (post-PATH) model values.
    Any residual > tol_abs after PATH convergence means the variable matched to
    that equation has a zero (or near-zero) Jacobian column — PATH never moved it
    and it's stuck at its lower bound.  This is a class of bug invisible to
    .nl / closure / value-diff tools.

    Classic example: eq_uh[r] = sum(zcons/bh) - 1, which has ∂/∂uh = 0
    because uh does not appear in the expression.  PATH satisfies eq_uh by
    adjusting zcons, leaves uh at lb=0.001, and eq_zcons has residual 0.115.
    """
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
                if con.upper is not None:
                    target = float(_pyomo_value(con.upper))
                elif con.lower is not None:
                    target = float(_pyomo_value(con.lower))
                else:
                    continue
                residual = abs(body_val - target)
                if residual > tol_abs:
                    out.append({
                        "name": cname, "index": index,
                        "residual": residual, "body": body_val, "target": target,
                    })
            except Exception:
                continue
    out.sort(key=lambda r: r["residual"], reverse=True)
    return out[:top_n]


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
