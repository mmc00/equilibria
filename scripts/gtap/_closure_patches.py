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
    """Fix `pft(r,f)` for factors that are dangling (no eq references them).

    Sluggish (sf) factors with xftflag=0 have no eq_xfteq constraint, so pft floats.
    Fixed/fnm factors (neither mf nor sf) also never appear in pft-dependent equations.
    Skips factors where xftflag=1 — those have eq_xfteq active and pft is anchored.
    """
    sf_set = set(getattr(params.sets, "sf", []) or [])
    mf_set = set(getattr(params.sets, "mf", []) or [])
    fixed = 0
    if hasattr(model, "pft"):
        for r in model.r:
            for f in model.f:
                f_str = str(f)
                # Only need to fix if not already fixed
                if model.pft[r, f].fixed:
                    continue
                # Sluggish (sf) with active xftflag: pft anchored by eq_xfteq → skip
                # Mobile (mf): pft determined by eq_pfeq law-of-one-price → skip
                # Sluggish with xftflag=0 OR fnm (neither mf nor sf): pft is dangling → fix
                if f_str in mf_set:
                    continue
                if hasattr(model, "xftflag"):
                    try:
                        if float(value(model.xftflag[r, f]) or 0.0) > 0.0:
                            continue
                    except Exception:
                        pass
                model.pft[r, f].fix()
                fixed += 1
    if fixed and label:
        print(f"[{label}] fixed {fixed} dangling pft(r,f) (sluggish xftflag=0 or fnm)")
    return fixed


def deactivate_xfteq_for_fixed_mobile(model, params, *, label: str = "") -> int:
    """Deactivate eq_xfteq for all factors whose xft is fixed (over-determining).

    When xft is fixed, eq_xfteq (supply curve) is over-determining for both
    mobile and sluggish factors:
    - Mobile (omegaf=inf): pft determined by eq_pfeq law-of-one-price.
    - Sluggish (finite omegaf): pft determined by eq_xft market clearing.
    In both cases, the supply curve eq_xfteq must be dropped.
    """
    deact = 0
    if hasattr(model, "eq_xfteq") and hasattr(model, "xftflag"):
        for r in model.r:
            for f in model.f:
                try:
                    if value(model.xftflag[r, f]) <= 0.0:
                        continue
                except Exception:
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


def deactivate_fixed_quantity_eqs(model, *, label: str = "") -> int:
    """Deactivate eq_va when both va[r,a] and xp[r,a] are fixed.

    When a sector has zero activity in the benchmark data and conditional fixing
    pins both va and xp to zero, eq_va becomes over-determining — the price ratio
    it encodes is already determined by other price equations.
    """
    deact = 0

    # eq_va: deactivate when both va[r,a] and xp[r,a] are fixed
    n_va = 0
    if hasattr(model, "eq_va") and hasattr(model, "va") and hasattr(model, "xp"):
        for idx in list(model.eq_va):
            con = model.eq_va[idx]
            if not con.active:
                continue
            r, a = idx
            try:
                va_fixed = model.va[r, a].fixed
                xp_fixed = model.xp[r, a].fixed
            except (KeyError, AttributeError):
                continue
            if va_fixed and xp_fixed:
                con.deactivate()
                n_va += 1
                deact += 1
    if n_va and label:
        print(f"[{label}] deactivated {n_va} eq_va (va+xp fixed → over-determining)")

    return deact


def deactivate_zero_unique_var_eqs(model, *, label: str = "") -> int:
    """Deactivate over-determining equations that have no unique free variable.

    An equation is structurally redundant when every free variable it references
    also appears in at least one other active equation. In that case the equation
    adds a constraint without contributing a new DOF — it cannot be matched by
    Hopcroft-Karp and will block solver convergence.

    Only fires when the system is over-determined (n_eqs > n_vars). Cross-checks
    each HK-unmatched equation to ensure it genuinely has zero unique variables
    before deactivating.

    NEVER_DEACT: equation blocks that define key variables even when those variables
    appear in many other equations (the variable-appearance count gives false zero-unique
    signals for widely-used variables like pmt, pm, pa, etc.).
    """
    # Equation families that DEFINE core variables — never deactivate even if
    # HK marks them unmatched due to wide variable sharing.
    NEVER_DEACT = {
        "eq_pmteq",   # defines pmt (import price aggregate)
        "eq_pmeq",    # defines pm (bilateral import price)
        "eq_pfaeq",   # defines pfa (agent factor price)
        "eq_pfyeq",   # defines pfy (household factor price)
        "eq_pefobeq", # defines pefob (export fob price)
        "eq_pmcifeq", # defines pmcif (import cif price)
        "eq_xwmg",    # defines xwmg (margins)
        "eq_walras",  # Walras law — must never be dropped
        "eq_ytax",    # income tax streams
        "eq_xfteq",   # defines xft[r,f] total factor supply (CET supply curve, altertax)
        "eq_pxeq",    # defines px[r,i] export price; dropped in omegax=inf leads to unconstrained px
    }
    cons_snap = sorted(
        model.component_data_objects(Constraint, active=True), key=lambda c: c.name
    )
    vars_snap = sorted(
        (v for v in model.component_data_objects(Var, active=True) if not v.fixed),
        key=lambda v: v.name,
    )
    n = len(cons_snap)
    nv = len(vars_snap)
    if n <= nv:
        return 0

    id2col = {id(v): j for j, v in enumerate(vars_snap)}
    var_name_to_col = {v.name: j for j, v in enumerate(vars_snap)}
    con_name_to_row = {c.name: u for u, c in enumerate(cons_snap)}
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

    pl = [-1] * n
    pr = [-1] * nv
    dist = [0] * n
    INF = 10**9

    # Pre-seed HK with forced pairs for key defining equations.
    # Without this, HK may assign a widely-used variable (pmt, pm, pft, etc.)
    # to a downstream equation (eq_xma, eq_paa, etc.) leaving the DEFINING
    # equation unmatched — even though the variable is only DEFINED by one eq.
    def _force_pair(eq_comp_name: str, var_comp_name: str) -> None:
        comp_eq = getattr(model, eq_comp_name, None)
        comp_var = getattr(model, var_comp_name, None)
        if comp_eq is None or comp_var is None:
            return
        for idx in comp_eq:
            con = comp_eq[idx]
            if not con.active:
                continue
            eq_name = con.name
            idx_str = ",".join(str(x) for x in idx) if isinstance(idx, tuple) else str(idx)
            var_name = f"{var_comp_name}[{idx_str}]"
            row = con_name_to_row.get(eq_name)
            col = var_name_to_col.get(var_name)
            if row is not None and col is not None and pl[row] == -1 and pr[col] == -1:
                if col in adj[row]:
                    pl[row] = col
                    pr[col] = row

    # Force-match defining equations to their primary output variable
    _force_pair("eq_pmteq", "pmt")   # pmt[r,i] = Armington import price aggregate
    _force_pair("eq_pmeq", "pm")     # pm[rp,i,r] = bilateral import price
    _force_pair("eq_pft", "pft")     # pft[r,f] = sluggish factor aggregate price (legacy)
    _force_pair("eq_pfteq", "pft")   # pft[r,f] = CET factor price equilibrium (altertax)
    _force_pair("eq_xfteq", "xft")   # xft[r,f] = CET factor supply aggregate (altertax)
    _force_pair("eq_pxeq", "px")     # px[r,i] = export price

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

    # Count how many equations reference each variable (for unique-var test)
    var_eq_count: dict[int, int] = {}
    for u in range(n):
        for col in adj[u]:
            var_eq_count[col] = var_eq_count.get(col, 0) + 1

    deact = 0
    for u in range(n):
        if pl[u] != -1:
            continue
        c = cons_snap[u]
        # Never deactivate core-defining equation families
        if c.parent_component().name in NEVER_DEACT:
            if label:
                print(f"[{label}] SKIP protected eq {c.name}")
            continue
        # Only deactivate if ALL free variables in this eq also appear elsewhere
        has_unique = any(var_eq_count.get(col, 0) == 1 for col in adj[u])
        if has_unique:
            if label:
                print(f"[{label}] WARN: skipping deactivation of {c.name} — has unique var")
            continue
        c.deactivate()
        deact += 1

    if deact and label:
        print(f"[{label}] deactivated {deact} zero-unique-var over-determining eqs")
    return deact


def apply_squareness_patches(model, params, *, label: str = "") -> dict[str, int]:
    """Run all post-closure patches in order. Returns counts per patch."""
    return {
        "pft_fixed": fix_sluggish_pft(model, params, label=label),
        "xfteq_deact": deactivate_xfteq_for_fixed_mobile(model, params, label=label),
        "xseq_deact": deactivate_unmatched_xseq(model, params, label=label),
        "fixed_qty_deact": deactivate_fixed_quantity_eqs(model, label=label),
        "zero_unique_deact": deactivate_zero_unique_var_eqs(model, label=label),
    }


def structural_matching(constraints, free_vars, *, forced_pairs=None, label: str = ""):
    """Hopcroft-Karp maximum bipartite matching: eq row -> var col.

    The PATH adapter pairs F[i] with var[i] positionally. Without a structural
    matching, alphabetical sort can pair an equation with an unrelated spectator
    var that sits at its lower bound, allowing PATH to terminate "feasible" with
    a large F[i] residual (see user memory gtap_mcp_pairing_pitfall).

    Returns var permutation so var[i] is structurally tied to constraints[i].

    forced_pairs: optional list of (eq_name, var_name) tuples to pin upfront
    (mirrors GAMS `model gtap / eq.var, ... /` declared matching).
    """
    n = len(constraints)
    nv = len(free_vars)
    var_id_to_col = {id(v): j for j, v in enumerate(free_vars)}
    adjacency: list[list[int]] = []
    for con in constraints:
        cols: list[int] = []
        seen: set[int] = set()
        for var_data in identify_variables(con.body, include_fixed=False):
            if var_data.fixed:
                continue
            col = var_id_to_col.get(id(var_data))
            if col is None or col in seen:
                continue
            seen.add(col)
            cols.append(col)
        adjacency.append(cols)

    pair_left = [-1] * n
    pair_right = [-1] * n
    distance = [0] * n
    INF = 10**9

    eq_name_to_row = {c.name: i for i, c in enumerate(constraints)}
    var_name_to_col = {v.name: j for j, v in enumerate(free_vars)}
    if forced_pairs:
        for eq_name, var_name in forced_pairs:
            r = eq_name_to_row.get(eq_name)
            c = var_name_to_col.get(var_name)
            if r is None or c is None:
                if label:
                    print(f"[{label}] matching WARN: forced pair {eq_name}<->{var_name} not found")
                continue
            if c not in adjacency[r]:
                if label:
                    print(f"[{label}] matching WARN: forced pair {eq_name}<->{var_name} not in adjacency")
                continue
            pair_left[r] = c
            pair_right[c] = r

    def bfs() -> bool:
        q: deque[int] = deque()
        found = False
        for u in range(n):
            if pair_left[u] == -1:
                distance[u] = 0
                q.append(u)
            else:
                distance[u] = INF
        while q:
            u = q.popleft()
            for v in adjacency[u]:
                m = pair_right[v]
                if m == -1:
                    found = True
                elif distance[m] == INF:
                    distance[m] = distance[u] + 1
                    q.append(m)
        return found

    def dfs(u: int) -> bool:
        for v in adjacency[u]:
            m = pair_right[v]
            if m == -1 or (distance[m] == distance[u] + 1 and dfs(m)):
                pair_left[u] = v
                pair_right[v] = u
                return True
        distance[u] = INF
        return False

    while bfs():
        for u in range(n):
            if pair_left[u] == -1:
                dfs(u)

    leftover = [j for j, m in enumerate(pair_right) if m == -1]
    li = 0
    for u in range(n):
        if pair_left[u] == -1:
            if li < len(leftover):
                pair_left[u] = leftover[li]
                li += 1

    if label:
        n_natural = sum(1 for u in range(n) if pair_left[u] != -1)
        print(f"[{label}] structural matching: {n_natural}/{n} pairs assigned")

    # Over-determined (n > nv): pair_left has n entries but free_vars has nv.
    # Return only the first nv matched entries (caller will trim constraints to match).
    if n > nv:
        matched_cols = [c for c in pair_left if 0 <= c < nv]
        return [free_vars[c] for c in matched_cols[:nv]]
    return [free_vars[c] for c in pair_left]
