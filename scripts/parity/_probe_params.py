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


def _build_gtap7_har_models(dataset: str):
    """Build the altertax model twice: WITHOUT and WITH t0_snapshot. Returns (m_no_t0, m_t0)."""
    _ROOT = Path(__file__).resolve().parents[2]
    import sys as _sys
    if str(_ROOT / "src") not in _sys.path:
        _sys.path.insert(0, str(_ROOT / "src"))
    from equilibria.templates.gtap import GTAPParameters, GTAPModelEquations
    from equilibria.templates.gtap.altertax import apply_altertax_elasticities
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    har = _ROOT / "datasets" / dataset
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=har / "basedata.har", sets_path=har / "sets.har",
        default_path=har / "default.prm", baserate_path=har / "baserate.har",
    )
    p_alt = apply_altertax_elasticities(p, in_place=False)
    res = list(p_alt.sets.r)[-1]
    base_cl = GTAPClosureConfig(name="base", closure_type="MCP",
                                capital_mobility="sluggish", fix_endowments=False,
                                fix_taxes=False, fix_technology=False, if_sub=False,
                                numeraire="pnum")
    alt_cl = GTAPClosureConfig(name="altertax", closure_type="MCP",
                               capital_mobility="mobile", fix_endowments=False,
                               fix_taxes=True, fix_technology=True, if_sub=False,
                               numeraire="pnum")
    m_no_t0 = GTAPModelEquations(p_alt.sets, p_alt, alt_cl, residual_region=res).build_model()
    # t0_snapshot must be a SOLVED base — the construction-dependence (pf0/Fisher
    # anchors) only shows up against the solved base levels, not the freshly-built
    # init values (which equal the no-t0 calibrated benchmark).
    m_b = GTAPModelEquations(p_alt.sets, p_alt, base_cl, residual_region=res).build_model()
    import importlib.util as _u
    _spec = _u.spec_from_file_location("run_gtap", str(_ROOT / "scripts" / "gtap" / "run_gtap.py"))
    _rg = _u.module_from_spec(_spec)
    _sys.modules["run_gtap"] = _rg
    _spec.loader.exec_module(_rg)
    _rg._run_path_capi_nonlinear_full(
        m_b, p_alt, enforce_post_checks=False, strict_path_capi=False,
        closure_config=base_cl, equation_scaling=True,
    )
    m_t0 = GTAPModelEquations(p_alt.sets, p_alt, alt_cl, residual_region=res,
                              t0_snapshot=m_b).build_model()
    return m_no_t0, m_t0


def diff_param_builds(dataset: str, tol_rel: float = 1e-3) -> list:
    """Return Params whose value changes between build-without-t0 and build-with-t0.

    Auto-detects the construction-dependent universe (the shock-bug signature)
    without needing GAMS. Each entry: {param, cells, changed, max_rel, worst}.
    """
    m_a, m_b = _build_gtap7_har_models(dataset)
    pa, pb = extract_params(m_a), extract_params(m_b)
    changed = []
    for name in sorted(set(pa) & set(pb)):
        n_chg = 0
        max_rel = 0.0
        worst = None
        for idx, va in pa[name].items():
            vb = pb[name].get(idx)
            if vb is None:
                continue
            rel = abs(va - vb) / max(abs(vb), 1e-12)
            if rel > tol_rel:
                n_chg += 1
                if rel > max_rel:
                    max_rel = rel
                    worst = (idx, va, vb)
        if n_chg > 0:
            changed.append({"param": name, "cells": len(pa[name]), "changed": n_chg,
                            "max_rel": max_rel, "worst": worst})
    changed.sort(key=lambda r: r["max_rel"], reverse=True)
    return changed
