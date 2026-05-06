"""Post-closure patches for GTAP MCP squareness.

`GTAPSolver.apply_aggressive_fixing_for_mcp` only fixes when the system is
under-determined (gap > 0). When the closure leaves the system *over*-determined
(gap <= 0) — as happens with the 9x10 dataset and `closure=gtap_standard` — the
helper does nothing and the wrapper rejects with a non-square error.

The three patches below mirror what `compare_nus333_vs_neos._solve` applies
inline; extracting them lets both the NUS333 script and `_run_path_capi_nonlinear_full`
share the same logic.

Order matters:
  1. Fix sluggish `pft(r,f)` (dangling — no eq references them).
  2. Deactivate `eq_xfteq` for mobile factors with xft fixed (over-determining).
  3. Hopcroft-Karp matching → deactivate unmatched `eq_xseq` under omegax=inf.
"""
from __future__ import annotations

from collections import deque
from typing import Any

from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import Constraint, Var, value


def fix_sluggish_pft(model, params, *, label: str = "") -> int:
    """Fix `pft(r,f)` for sluggish factors — they're dangling (no eq references them)."""
    sf_set = set(getattr(params.sets, "sf", []) or [])
    fixed = 0
    if hasattr(model, "pft") and sf_set:
        for r in model.r:
            for f in model.f:
                if str(f) in sf_set and not model.pft[r, f].fixed:
                    model.pft[r, f].fix()
                    fixed += 1
    if fixed and label:
        print(f"[{label}] fixed {fixed} sluggish pft(r,f) (no eq references them)")
    return fixed


def deactivate_xfteq_for_fixed_mobile(model, params, *, label: str = "") -> int:
    """Deactivate eq_xfteq for mobile factors whose xft is fixed (over-determining)."""
    mf_set = set(getattr(params.sets, "mf", []) or [])
    deact = 0
    if hasattr(model, "eq_xfteq") and mf_set:
        for r in model.r:
            for f in model.f:
                if str(f) not in mf_set:
                    continue
                if value(model.xftflag[r, f]) <= 0.0:
                    continue
                if not model.xft[r, f].fixed:
                    continue
                try:
                    con = model.eq_xfteq[r, f]
                except KeyError:
                    continue
                if con.active:
                    con.deactivate()
                    deact += 1
    if deact and label:
        print(f"[{label}] deactivated {deact} eq_xfteq (xft fixed → over-determining)")
    return deact


def deactivate_unmatched_xseq(model, params, *, label: str = "") -> int:
    """Hopcroft-Karp matching → deactivate unmatched `eq_xseq` under omegax=inf.

    Under omegax=inf the supply identity xs = xds + xet collapses (degenerate CET).
    Some (r,i) eq_xseq end up unmatched; those are the redundant ones to remove.
    """
    if not hasattr(model, "eq_xseq"):
        return 0

    cons_snap = sorted(
        model.component_data_objects(Constraint, active=True), key=lambda c: c.name
    )
    vars_snap = sorted(
        (v for v in model.component_data_objects(Var, active=True) if not v.fixed),
        key=lambda v: v.name,
    )
    id2col = {id(v): j for j, v in enumerate(vars_snap)}
    adj: list[list[int]] = []
    for c in cons_snap:
        cols: list[int] = []
        seen: set[int] = set()
        for v in identify_variables(c.body, include_fixed=False):
            if v.fixed:
                continue
            col = id2col.get(id(v))
            if col is None or col in seen:
                continue
            seen.add(col)
            cols.append(col)
        adj.append(cols)

    n = len(cons_snap)
    nv = len(vars_snap)
    pl = [-1] * n
    pr = [-1] * nv
    dist = [0] * n
    INF = 10**9

    def bfs() -> bool:
        q: deque[int] = deque()
        found = False
        for u in range(n):
            if pl[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = INF
        while q:
            u = q.popleft()
            for v in adj[u]:
                m = pr[v]
                if m == -1:
                    found = True
                elif dist[m] == INF:
                    dist[m] = dist[u] + 1
                    q.append(m)
        return found

    def dfs(u: int) -> bool:
        for v in adj[u]:
            m = pr[v]
            if m == -1 or (dist[m] == dist[u] + 1 and dfs(m)):
                pl[u] = v
                pr[v] = u
                return True
        dist[u] = INF
        return False

    while bfs():
        for u in range(n):
            if pl[u] == -1:
                dfs(u)

    deact = 0
    omegax = getattr(params.elasticities, "omegax", {})
    for u in range(n):
        if pl[u] != -1:
            continue
        c = cons_snap[u]
        if c.parent_component().name != "eq_xseq":
            continue
        idx = c.index()
        omega = omegax.get(idx, float("inf"))
        if omega == float("inf"):
            c.deactivate()
            deact += 1

    if label:
        unmatched_eqs = [
            cons_snap[u].name for u in range(n)
            if pl[u] == -1 and cons_snap[u].active
        ]
        unmatched_vars = [
            vars_snap[v].name for v in range(nv)
            if pr[v] == -1 and not vars_snap[v].fixed
        ]
        if deact:
            print(f"[{label}] deactivated {deact} unmatched eq_xseq under omegax=inf")
        if unmatched_eqs:
            print(f"[{label}] unmatched active eqs ({len(unmatched_eqs)}): {unmatched_eqs[:8]}")
        if unmatched_vars:
            print(f"[{label}] unmatched free vars ({len(unmatched_vars)}): {unmatched_vars[:8]}")
    return deact


def apply_squareness_patches(model, params, *, label: str = "") -> dict[str, int]:
    """Run all three post-closure patches in order. Returns counts per patch."""
    return {
        "pft_fixed": fix_sluggish_pft(model, params, label=label),
        "xfteq_deact": deactivate_xfteq_for_fixed_mobile(model, params, label=label),
        "xseq_deact": deactivate_unmatched_xseq(model, params, label=label),
    }
