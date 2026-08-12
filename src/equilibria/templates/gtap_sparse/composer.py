"""GTAP sparse-trade composer: fix padding trade routes to ~0 and deactivate their eqs.

Reuses build_block_single_period (the standard levels block model); the sparse layer
is a post-build step that fixes the padding-route VarData (base flow ~0) and
deactivates their route constraints, so the active KKT system shrinks and the
Jacobian loses the extreme-scale entries those degenerate cells contribute.
"""

from __future__ import annotations

from typing import Any

from pyomo.environ import Constraint, Var, value

# Trade var families indexed by route. xw/xwmg/pe/pwmg/pmcif/pefob/pm are (r,i,rp)
# = (exporter, commodity, importer). xmgm is (m,r,i,rp).
_ROUTE_VARS_3D = ("xw", "xwmg", "pe", "pwmg", "pmcif", "pefob", "pm")
_ROUTE_VARS_4D = ("xmgm",)
# Route constraints keyed (r,i,rp) or (m,r,i,rp).
_ROUTE_CONS_3D = (
    "eq_xweq",
    "eq_peeq",
    "eq_pmeq",
    "eq_pwmg",
    "eq_xwmg",
    "eq_pmcif",
    "eq_pefob",
    "eq_pmcifeq",
    "eq_pefobeq",
)
_ROUTE_CONS_4D = ("eq_xmgm",)

_PAD = 1e-6  # base flow below this = padding route


def _padding_routes(params: Any, sets: Any) -> set:
    """Return the set of (exporter, commodity, importer) route tuples whose base
    bilateral trade flow (vxsb) is below _PAD — the non-existent routes."""
    b = params.benchmark
    vxsb = getattr(b, "vxsb", {}) or {}
    pad = set()
    for k, v in vxsb.items():
        if abs(v) < _PAD:
            pad.add(k)  # (exporter r, commodity i, importer rp)
    return pad


def _margin_padding(params: Any, sets: Any) -> set:
    """Return (margin, r, i, rp) tuples whose base margin usage (vtwr) is ~0."""
    b = params.benchmark
    vtwr = getattr(b, "vtwr", {}) or {}
    margins = [str(m) for m in sets.m]
    regs = [str(r) for r in sets.r]
    comms = [str(i) for i in sets.i]
    # vtwr keys are (r, i, rp, margin). Present+nonzero = active; everything else pads.
    active = set()
    for k, v in vtwr.items():
        if abs(v) > _PAD and len(k) == 4:
            r, i, rp, m = k
            active.add((m, r, i, rp))
    pad = set()
    for m in margins:
        for r in regs:
            for i in comms:
                for rp in regs:
                    if (m, r, i, rp) not in active:
                        pad.add((m, r, i, rp))
    return pad


def fix_padding_routes(pm: Any, params: Any, sets: Any) -> dict:
    """Fix padding-route VarData to their (~0) benchmark value and deactivate their
    route constraints, IN PLACE on a built (single- or multi-period) Pyomo model.
    Handles both bare (r,i,rp) keys and multi-period (r,i,rp,period) keys.
    Returns counts."""
    route_pad = _padding_routes(params, sets)  # (r,i,rp)
    margin_pad = _margin_padding(params, sets)  # (m,r,i,rp)

    n_var_fixed = 0
    n_con_off = 0

    def _route_key(idx):
        # strip trailing period label if present
        if (
            isinstance(idx, tuple)
            and idx
            and str(idx[-1]) in ("base", "check", "shock")
        ):
            return idx[:-1]
        return idx if isinstance(idx, tuple) else (idx,)

    # 3-D route vars
    for vn in _ROUTE_VARS_3D:
        v = getattr(pm, vn, None)
        if v is None:
            continue
        for idx in v:
            rk = _route_key(idx)
            if len(rk) == 3 and rk in route_pad:
                vd = v[idx]
                if not vd.fixed:
                    try:
                        vd.fix(float(value(vd, exception=False) or 0.0))
                        n_var_fixed += 1
                    except Exception:
                        pass
    # 4-D margin vars (xmgm)
    for vn in _ROUTE_VARS_4D:
        v = getattr(pm, vn, None)
        if v is None:
            continue
        for idx in v:
            rk = _route_key(idx)
            if len(rk) == 4 and rk in margin_pad:
                vd = v[idx]
                if not vd.fixed:
                    try:
                        vd.fix(float(value(vd, exception=False) or 0.0))
                        n_var_fixed += 1
                    except Exception:
                        pass
    # route constraints (3-D and 4-D)
    for cn in _ROUTE_CONS_3D:
        c = getattr(pm, cn, None)
        if c is None:
            continue
        for idx in c:
            rk = _route_key(idx)
            if len(rk) == 3 and rk in route_pad and c[idx].active:
                c[idx].deactivate()
                n_con_off += 1
    for cn in _ROUTE_CONS_4D:
        c = getattr(pm, cn, None)
        if c is None:
            continue
        for idx in c:
            rk = _route_key(idx)
            if len(rk) == 4 and rk in margin_pad and c[idx].active:
                c[idx].deactivate()
                n_con_off += 1

    return {
        "route_padding": len(route_pad),
        "margin_padding": len(margin_pad),
        "vars_fixed": n_var_fixed,
        "cons_deactivated": n_con_off,
    }
