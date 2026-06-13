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
