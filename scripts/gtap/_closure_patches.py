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

from _t_utils import t0


def fix_sluggish_pft(model, params, *, label: str = "") -> int:
    """Fix ``pft(r,f)`` for sluggish factors across ALL periods.

    Sluggish factors (``sf``) have ``xftflag[r,f]=0``, so neither ``eq_xft``
    nor ``eq_xfteq`` is generated for them. Their ``pft[r,f,t]`` would be a
    dangling free var with no equation at any period. Fix them at their
    initialized (benchmark) value for all periods.

    Mobile-factor ``pft`` is handled by ``eq_xfteq`` (supply equation) which
    remains active at non-base periods and is the only dedicated ``pft`` equation.

    Applies to ALL periods in ``model.t``.
    """
    sf_set = set(getattr(params.sets, "sf", []) or [])
    fixed = 0
    if hasattr(model, "pft") and sf_set:
        t_iter = list(model.t) if hasattr(model, "t") else [t0(model)]
        for r in model.r:
            for f in model.f:
                if str(f) not in sf_set:
                    continue
                for t in t_iter:
                    try:
                        vd = model.pft[r, f, t]
                    except KeyError:
                        continue
                    if not vd.fixed:
                        vd.fix()
                        fixed += 1
    if fixed and label:
        print(f"[{label}] fixed {fixed} sluggish pft(r,f) (no eq references them)")
    return fixed


def deactivate_xft_for_fixed_endowments(model, params, *, label: str = "") -> int:
    """Deactivate ``eq_xft[r,f,t]`` when both ``xft`` and ``pft`` are fixed.

    When xft[r,f,t] is fixed (endowment exogenous) AND pft[r,f,t] is fixed
    (e.g. by fix_sluggish_pft for sluggish factors, or by _fix_base_only_vars),
    the demand-sum equation ``eq_xft`` is redundant:

        sum_a xf[r,f,a,t] = xft[r,f,t]   (already pinned by fixed xft)

    With eq_xft active, H-K can match it to any ``xf[r,f,a,t]`` variable,
    stealing that xf from its natural ``eq_xfeq[r,f,a,t]`` equation and leaving
    eq_xfeq unmatched (hence unable to be deactivated by the whitelist). By
    proactively deactivating eq_xft BEFORE the H-K pass, all xf variables
    remain available for their eq_xfeq matches.

    Only deactivates when BOTH xft AND pft are fixed (not just one). This is
    the safe condition: with pft fixed there is no free endowment price either,
    so the demand-sum equation carries no information for the solver.

    Applies to ALL periods in ``model.t``.
    """
    deact = 0
    if not hasattr(model, "eq_xft"):
        return deact
    t_iter = list(model.t) if hasattr(model, "t") else [t0(model)]
    for r in model.r:
        for f in model.f:
            for t in t_iter:
                try:
                    xft_vd = model.xft[r, f, t]
                    pft_vd = model.pft[r, f, t]
                except KeyError:
                    continue
                if not (xft_vd.fixed and pft_vd.fixed):
                    continue
                try:
                    con = model.eq_xft[r, f, t]
                except KeyError:
                    continue
                if con.active:
                    con.deactivate()
                    deact += 1
    if deact and label:
        print(f"[{label}] deactivated {deact} eq_xft (xft+pft both fixed → redundant demand-sum)")
    return deact


def deactivate_xfteq_for_fixed_mobile(model, params, *, label: str = "") -> int:
    """Deactivate ``eq_xfteq`` (supply) for mobile factors when ``xft`` is fixed.

    When the closure fixes ``xft[r,f,t]`` (endowments exogenous), the supply
    equation ``eq_xfteq[r,f,t]`` would over-determine the system together with
    the demand-sum ``eq_xft[r,f,t]``. Deactivate ``eq_xfteq`` for ALL periods
    where ``xft`` is fixed.

    Without this, for each fixed (r,f,t) we have N+2 equations for N+1 free vars
    (N xf[r,f,a,t] + pft[r,f,t]) — over-determined by 1 per mobile factor pair.

    With ``eq_xfteq`` deactivated, ``pft[r,f,t]`` is still constrained: the
    law-of-one-price ``eq_pfeq[r,f,a,t]`` (omegaf=inf: pfy == pft) ties each
    ``pf[r,f,a,t]`` to ``pft[r,f,t]``, so the factor price system remains
    consistent without needing ``eq_xfteq``.

    Applies to ALL periods in ``model.t``.
    """
    mf_set = set(getattr(params.sets, "mf", []) or [])
    deact = 0
    if hasattr(model, "eq_xfteq") and mf_set:
        t_iter = list(model.t) if hasattr(model, "t") else [t0(model)]
        for r in model.r:
            for f in model.f:
                if str(f) not in mf_set:
                    continue
                if value(model.xftflag[r, f]) <= 0.0:
                    continue
                for t in t_iter:
                    try:
                        xft_vd = model.xft[r, f, t]
                    except KeyError:
                        continue
                    if not xft_vd.fixed:
                        continue
                    try:
                        con = model.eq_xfteq[r, f, t]
                    except KeyError:
                        continue
                    if con.active:
                        con.deactivate()
                        deact += 1
    if deact and label:
        print(f"[{label}] deactivated {deact} eq_xfteq (xft fixed → over-determining)")
    return deact


def deactivate_unmatched_redundant(
    model, params, *, label: str = "", forced_pairs: "list[tuple[str,str]] | None" = None
) -> int:
    """Hopcroft-Karp matching → deactivate structurally unmatched redundant equations.

    Runs bipartite matching on active constraints vs free variables. Equations that
    remain unmatched are structurally redundant (all their free-variable neighbours
    are already claimed by other equations). We deactivate the redundant ones from a
    whitelist of economically-dualizable equation families:

    * ``eq_xseq``: supply identity (xs = xds + xet) collapses under omegax=inf.
    * ``eq_xft``:  factor-market clearing redundant when xft is fixed and pft is
                   fixed (individual xf determined solely by eq_xfeq demands).
                   NOTE: ``apply_squareness_patches`` calls
                   ``deactivate_xft_for_fixed_endowments`` BEFORE this H-K pass,
                   so eq_xft is usually already deactivated and never reaches here.
                   The whitelist entry is a safety fallback.
    * ``eq_pxeq``: unit-cost duality equation — the CES price index identity is
                   implied by the CES demand system via Shephard's lemma; redundant
                   when all free vars (px, pnd, pva) are already matched.
    * ``eq_xweq``: Armington bilateral trade identity, redundant for zero-benchmark
                   self-trade routes.

    ``forced_pairs``: optional list of (eq_name_prefix, var_name_prefix) pairs to pin
    upfront before H-K runs, using the same family-matching logic as
    :func:`structural_matching`. This prevents global aggregate equations (e.g.
    ``eq_facty``, ``eq_pfact``, ``eq_ytax``) from stealing factor-demand variables
    away from their local equations (``eq_xfeq``).
    """
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

    # Apply forced pairs (family matching: "eq_facty" matches eq_facty[r,t] ↔ facty[r,t])
    # Forced pairs are LOCKED — the H-K augmenting-path search must not re-assign them.
    # We track locked rows and cols to prevent the BFS/DFS from traversing through them.
    locked_rows: set[int] = set()
    locked_cols: set[int] = set()
    if forced_pairs:
        eq_name_to_row = {c.name: i for i, c in enumerate(cons_snap)}
        var_name_to_col = {v.name: j for j, v in enumerate(vars_snap)}
        for eq_pfx, var_pfx in forced_pairs:
            eq_prefix = eq_pfx + "["
            var_prefix = var_pfx + "["
            suffix_to_row: dict[str, int] = {}
            for name, row_i in eq_name_to_row.items():
                if name == eq_pfx or name.startswith(eq_prefix):
                    suffix_to_row[name[len(eq_pfx):]] = row_i
            suffix_to_col: dict[str, int] = {}
            for name, col_j in var_name_to_col.items():
                if name == var_pfx or name.startswith(var_prefix):
                    suffix_to_col[name[len(var_pfx):]] = col_j
            for suffix, row_i in suffix_to_row.items():
                col_j = suffix_to_col.get(suffix)
                if col_j is None:
                    continue
                if col_j not in adj[row_i]:
                    continue
                # Undo any prior conflicting assignments
                old_col = pl[row_i]
                if old_col != -1:
                    pr[old_col] = -1
                    locked_rows.discard(row_i)
                    locked_cols.discard(old_col)
                old_row = pr[col_j]
                if old_row != -1:
                    pl[old_row] = -1
                    locked_rows.discard(old_row)
                    locked_cols.discard(col_j)
                pl[row_i] = col_j
                pr[col_j] = row_i
                locked_rows.add(row_i)
                locked_cols.add(col_j)

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
                if v in locked_cols:
                    # This column is locked to a forced-pair row — do not traverse.
                    continue
                m = pr[v]
                if m == -1:
                    found = True
                elif dist[m] == INF:
                    dist[m] = dist[u] + 1
                    q.append(m)
        return found

    def dfs(u: int) -> bool:
        for v in adj[u]:
            if v in locked_cols:
                continue
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
    deact_by_family: dict[str, int] = {}
    omegax = getattr(params.elasticities, "omegax", {})

    # Whitelist of equation families that may be deactivated when unmatched.
    # Each entry: (eq_name, condition_fn or None)
    # condition_fn(c, idx, params) → True if safe to deactivate.
    def _xseq_cond(c, idx, params) -> bool:
        omega = omegax.get(idx, float("inf"))
        return omega == float("inf")

    def _xft_cond(c, idx, params) -> bool:
        # Safe to remove when xft and pft are both fixed (demand eqs determine xf).
        try:
            r, f, t = idx
            xft_vd = model.xft[r, f, t]
            return xft_vd.fixed
        except (KeyError, ValueError, TypeError):
            return False

    def _pxeq_cond(c, idx, params) -> bool:
        # Unit-cost duality: always structurally dualizable.
        return True

    def _xweq_cond(c, idx, params) -> bool:
        # Bilateral trade identity: safe to remove when unmatched.
        return True

    def _peeq_cond(c, idx, params) -> bool:
        # Export price identity (pe == pet at omegaw=inf): dually redundant.
        return True

    def _pvaeq_cond(c, idx, params) -> bool:
        # Price of value-added bundle: unit-cost duality (Shephard's lemma).
        # eq_pvaeq is the dual of the factor-demand CES/CD system — it is
        # structurally redundant (implied by eq_xfeq + eq_pfeq) when all its
        # free variables (pva, va, pfa, xf) are already matched by other eqs.
        return True

    def _pnum_cond(c, idx, params) -> bool:
        # Numeraire equation: pnum[t] == pwfact[t].
        # pnum[t] is the global price numeraire; pwfact[t] is the Törnqvist
        # world factor price index (determined by eq_pwfact via forced pair).
        # At non-base periods: eq_pwfact forces pwfact[t] to be matched,
        # and eq_ptmg (ptmg[m,t] == pnum[t] for margins without supply) matches
        # pnum[t] — leaving eq_pnum[t] with zero free-var neighbors → unmatched.
        # At that point, eq_pnum is structurally redundant: pnum and pwfact are
        # independently determined by their own equations; the identity pnum==pwfact
        # is a Walras' law consequence.
        # At base period: eq_pnum[base] is also unmatched (same reason), but we
        # must NOT deactivate it there because yi[USA,base] (residual income) has
        # no dedicated equation — the yi/pnum pair provides the Walras' law closure.
        # Only deactivate at non-first periods where the Walras closure is handled
        # differently (lagged-state anchoring replaces income circuit closure).
        try:
            t = idx if not isinstance(idx, tuple) else idx[-1]
            first = t0(model)
            return str(t) != str(first)
        except Exception:
            return False

    def _xfeq_cond(c, idx, params) -> bool:
        # Factor demand equation: safe to deactivate when xf is already fixed
        # (by _fix_inactive_flows for zero-VFM factor-activity pairs). With xf
        # fixed, eq_xfeq is over-determining — the demand relationship is already
        # pinned by the zero-flow assumption.
        try:
            r, f, a, t = idx
            xf_vd = model.xf[r, f, a, t]
            return xf_vd.fixed
        except (KeyError, ValueError, TypeError):
            return False

    whitelist = {
        "eq_xseq": _xseq_cond,
        "eq_xft": _xft_cond,
        # eq_pxeq is NOT in the whitelist: the unit cost duality equation changes
        # the solution if deactivated (px determined by eq_po gives different xp
        # than px from CES price index). Must remain active.
        # "eq_pxeq": _pxeq_cond,
        "eq_xweq": _xweq_cond,
        "eq_peeq": _peeq_cond,
        "eq_pvaeq": _pvaeq_cond,
        "eq_pnum": _pnum_cond,
        "eq_xfeq": _xfeq_cond,
    }

    for u in range(n):
        if pl[u] != -1:
            continue
        c = cons_snap[u]
        eq_family = c.parent_component().name
        cond_fn = whitelist.get(eq_family)
        if cond_fn is None:
            continue
        idx = c.index()
        if not cond_fn(c, idx, params):
            continue
        c.deactivate()
        deact += 1
        deact_by_family[eq_family] = deact_by_family.get(eq_family, 0) + 1

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
            by_fam = ", ".join(f"{k}:{v}" for k, v in sorted(deact_by_family.items()))
            print(f"[{label}] deactivated {deact} unmatched redundant eqs ({by_fam})")
        if unmatched_eqs:
            print(f"[{label}] unmatched active eqs ({len(unmatched_eqs)}): {unmatched_eqs[:8]}")
        if unmatched_vars:
            print(f"[{label}] unmatched free vars ({len(unmatched_vars)}): {unmatched_vars[:8]}")
    return deact


# Keep old name as alias for backward-compat callers that import by name.
deactivate_unmatched_xseq = deactivate_unmatched_redundant


def apply_squareness_patches(
    model,
    params,
    *,
    label: str = "",
    forced_pairs: "list[tuple[str,str]] | None" = None,
) -> dict[str, int]:
    """Run all post-closure patches in order. Returns counts per patch.

    ``deactivate_unmatched_redundant`` is iterated until convergence: deactivating
    one equation may free up a variable that causes a previously-matched equation
    to become unmatched in the next round. We iterate up to 8 rounds.

    ``forced_pairs``: optional list of (eq_name_prefix, var_name_prefix) pairs
    passed to :func:`deactivate_unmatched_redundant` to pin global aggregate
    equations (e.g. ``eq_facty`` → ``facty``) before H-K runs so they don't
    steal individual-level factor-demand variables away from ``eq_xfeq``.
    """
    pft_fixed = fix_sluggish_pft(model, params, label=label)
    xft_deact = deactivate_xft_for_fixed_endowments(model, params, label=label)
    xfteq_deact = deactivate_xfteq_for_fixed_mobile(model, params, label=label)
    xseq_total = 0
    for _round in range(8):
        n = deactivate_unmatched_xseq(model, params, label=label, forced_pairs=forced_pairs)
        xseq_total += n
        if n == 0:
            break
    return {
        "pft_fixed": pft_fixed,
        "xft_deact": xft_deact,
        "xfteq_deact": xfteq_deact,
        "xseq_deact": xseq_total,
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
            # Three matching strategies (tried in order):
            # 1. Exact match: "eq_pwfact[check]" + "pwfact[check]"
            # 2. Single-prefix match: "eq_pwfact" matches exactly ONE constraint
            # 3. Family match: "eq_facty" + "facty" → pair eq_facty[r,t] with facty[r,t]
            #    by stripping the family prefix and matching the index suffix.
            r = eq_name_to_row.get(eq_name)
            c = var_name_to_col.get(var_name)
            if r is not None and c is not None:
                # Exact match → single pair
                if c not in adjacency[r]:
                    if label:
                        print(f"[{label}] matching WARN: forced pair {eq_name}<->{var_name} not in adjacency")
                    continue
                pair_left[r] = c
                pair_right[c] = r
                continue
            if r is None and c is None:
                # Family match: pair eq_name[idx] <-> var_name[idx] for each shared idx suffix.
                eq_prefix = eq_name + "["
                var_prefix = var_name + "["
                # Build suffix → row dict for equations
                suffix_to_row: dict[str, int] = {}
                for name, row_i in eq_name_to_row.items():
                    if name == eq_name or name.startswith(eq_prefix):
                        suffix = name[len(eq_name):]  # "[r,t]" or ""
                        suffix_to_row[suffix] = row_i
                # Build suffix → col dict for variables
                suffix_to_col: dict[str, int] = {}
                for name, col_j in var_name_to_col.items():
                    if name == var_name or name.startswith(var_prefix):
                        suffix = name[len(var_name):]  # "[r,t]" or ""
                        suffix_to_col[suffix] = col_j
                n_pinned = 0
                for suffix, row_i in suffix_to_row.items():
                    col_j = suffix_to_col.get(suffix)
                    if col_j is None:
                        continue
                    if col_j not in adjacency[row_i]:
                        continue
                    pair_left[row_i] = col_j
                    pair_right[col_j] = row_i
                    n_pinned += 1
                if n_pinned == 0 and label:
                    print(f"[{label}] matching WARN: forced pair {eq_name}<->{var_name} not found")
                continue
            # One side exact, other needs prefix match (single unique)
            if r is None:
                matches_r = [
                    i for name, i in eq_name_to_row.items()
                    if name == eq_name or name.startswith(eq_name + "[")
                ]
                r = matches_r[0] if len(matches_r) == 1 else None
            if c is None:
                matches_c = [
                    j for name, j in var_name_to_col.items()
                    if name == var_name or name.startswith(var_name + "[")
                ]
                c = matches_c[0] if len(matches_c) == 1 else None
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

    return [free_vars[c] for c in pair_left]
