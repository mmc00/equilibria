"""Param/calibration diff: compare a built model's Pyomo Params vs the GAMS GDX.

Covers the cascade blind spot — build-time calibrated constants (pf0, base_rgdpmp,
p_gf, betap) that the .nl / residual / warm-start / closure tools take as literals.
"""
from __future__ import annotations

from typing import Any, Optional

from pyomo.environ import Param, value

from _probe_queries import _idx_key  # reuse index normalization

# Python Param name -> GAMS symbol, optionally "@<period>" for a different period.
# pf0/xf0 are the base-period pf/xf; base_rgdpmp is rgdpmp@base; betap is betaP.
ALIAS_MAP = {
    "pf0": "pf@base",
    "xf0": "xf@base",
    "base_rgdpmp": "rgdpmp@base",
    "base_pabs": "pabs@base",
    "betap": "betaP",
    "betag": "betaG",
    "betas": "betaS",
    "kappaf_activity": "kappaf",
}


def resolve_gams_symbol(param_name: str) -> tuple[str, Optional[str]]:
    """Return (gams_symbol, period_override_or_None) for a Python Param name."""
    alias = ALIAS_MAP.get(param_name, param_name)
    if "@" in alias:
        sym, period = alias.split("@", 1)
        return sym, period
    return alias, None


def extract_params(model: Any) -> dict:
    """Return {param_name: {idx_tuple: float}} for all active Params."""
    out: dict[str, dict] = {}
    for p in model.component_objects(Param, active=True):
        cells = {}
        for idx in p:
            try:
                v = value(p[idx])
                if v is not None:
                    cells[_idx_key(idx)] = float(v)
            except Exception:
                pass
        if cells:
            out[p.local_name] = cells
    return out


import re
from pathlib import Path

_PREFIXES = ("a_", "c_", "f_", "r_")


def _strip(s) -> str:
    s = str(s)
    for p in _PREFIXES:
        if s.startswith(p):
            return s[len(p):]
    return s


def _gams_period_slice(gdx_path: Path, symbol: str, period: str) -> dict:
    """Read a GAMS symbol, keep entries for `period`, strip prefixes + period dim."""
    from _diff_core import gams_levels  # type: ignore[import-not-found]
    raw = gams_levels(Path(gdx_path), symbol)
    out = {}
    for k, v in raw.items():
        if not isinstance(k, tuple) or k[-1] != period:
            continue
        nk = tuple(_strip(x) for x in k[:-1])
        out[nk] = float(v)
    return out


def diff_params_vs_gams(model, gdx_path, period: str, tol_rel: float = 1e-3) -> dict:
    """Compare all model Params vs the GAMS GDX for `period`.

    Returns {'diverge': [...], 'ok': [...], 'no_match': [...]}. Each diverge entry:
    {param, gams_symbol, cells, match, diverge, max_rel, worst}.
    """
    params = extract_params(model)
    diverge, ok, no_match = [], [], []
    for name, cells in params.items():
        gsym, period_override = resolve_gams_symbol(name)
        gperiod = period_override or period
        gams = _gams_period_slice(Path(gdx_path), gsym, gperiod)
        if not gams:
            no_match.append({"param": name, "gams_symbol": gsym, "cells": len(cells)})
            continue
        n_match = n_div = 0
        max_rel = 0.0
        worst = None
        for idx, pv in cells.items():
            key = idx[0] if (isinstance(idx, tuple) and len(idx) == 1) else idx
            gv = gams.get(key)
            if gv is None:
                continue
            rel = abs(pv - gv) / max(abs(gv), 1e-12)
            if rel > tol_rel:
                n_div += 1
                if rel > max_rel:
                    max_rel = rel
                    worst = (key, pv, gv)
            else:
                n_match += 1
        rec = {"param": name, "gams_symbol": gsym, "cells": n_match + n_div,
               "match": n_match, "diverge": n_div, "max_rel": max_rel, "worst": worst}
        (diverge if n_div > 0 else ok).append(rec)
    diverge.sort(key=lambda r: r["max_rel"], reverse=True)
    return {"diverge": diverge, "ok": ok, "no_match": no_match}
