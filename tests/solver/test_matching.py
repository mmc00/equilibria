"""``match_rows`` is the old recursive Hopcroft-Karp of ``structural_matching``, exactly."""

from __future__ import annotations

import random
from collections import deque
from typing import Any, cast

import pytest

from equilibria.solver._matching import match_rows


def reference(n, adjacency, forced):
    """The loop body of the pre-extraction ``structural_matching``, recursive DFS."""
    pair_left = [-1] * n
    pair_right = [-1] * n
    distance = [0] * n
    INF = 10**9
    locked: set[int] = set()
    for r, c, hard in forced:
        if c not in adjacency[r] and not hard:
            continue
        pair_left[r] = c
        pair_right[c] = r
        if hard:
            locked.add(r)

    def bfs():
        q = deque()
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
                if m in locked:
                    continue
                if m == -1:
                    found = True
                elif distance[m] == INF:
                    distance[m] = distance[u] + 1
                    q.append(m)
        return found

    def dfs(u):
        for v in adjacency[u]:
            m = pair_right[v]
            if m in locked:
                continue
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
        if pair_left[u] == -1 and li < len(leftover):
            pair_left[u] = leftover[li]
            li += 1
    return pair_left


def _random_case(rng, n, density, n_forced):
    adj = []
    for _ in range(n):
        k = max(1, int(density * n * rng.random() * 2))
        cols = []
        for c in rng.sample(range(n), min(k, n)):
            cols.append(c)
        adj.append(cols)  # random order, no dups
    forced = [
        (rng.randrange(n), rng.randrange(n), rng.random() < 0.4)
        for _ in range(n_forced)
    ]
    return adj, forced


@pytest.mark.parametrize("seed", range(300))
def test_equals_recursive_reference(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 40)
    adj, forced = _random_case(rng, n, rng.choice([0.05, 0.1, 0.3]), rng.randint(0, 8))
    assert match_rows(n, adj, forced) == reference(n, adj, forced)


def test_ties_complete_bipartite():
    n = 12
    adj = [list(range(n)) for _ in range(n)]
    assert match_rows(n, adj) == reference(n, adj, [])
    adj = [list(range(n))[::-1] for _ in range(n)]
    assert match_rows(n, adj) == reference(n, adj, [])


def test_later_forced_pair_overwrites():
    adj = [[0, 1], [0, 1]]
    forced = [(0, 0, False), (0, 1, False), (1, 1, True)]
    assert match_rows(2, adj, forced) == reference(2, adj, forced)


def _chain(n):
    """Row 0 reaches the free column 0 only through an augmenting path of depth n."""
    adj = [[1]] + [[i, i + 1] for i in range(1, n - 1)] + [[n - 1, 0]]
    forced = [(i, i, False) for i in range(1, n)]
    return adj, forced


def test_deep_augmenting_path_equals_reference():
    import sys

    n = 1500
    adj, forced = _chain(n)
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(20000)
    try:
        want = reference(n, adj, forced)
    finally:
        sys.setrecursionlimit(old)
    got = match_rows(n, adj, forced)
    assert got == want
    assert got[0] == 1  # the path was really augmented


def test_deep_augmenting_path_does_not_recurse():
    n = 100000  # a recursive DFS would exceed the interpreter's limit
    adj, forced = _chain(n)
    got = match_rows(n, adj, forced)
    assert sorted(got) == list(range(n))


# structural_matching, before and after the extraction ───────────────────────────

BEFORE = "ed3c9496"  # last commit with the inline algorithm


def _old_structural_matching():
    import subprocess
    import types
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    src = subprocess.run(
        [
            "git",
            "show",
            f"{BEFORE}:src/equilibria/solver/_closure_patches.py",
        ],
        capture_output=True,
        text=True,
        cwd=root,
    )
    if src.returncode:
        pytest.skip(f"{BEFORE} not in this clone")
    mod = types.ModuleType("_closure_patches_before")
    exec(compile(src.stdout, "_closure_patches_before", "exec"), mod.__dict__)
    return mod.structural_matching


def _pyomo_model(seed, n, nv_extra=0):
    import pyomo.environ as pyo

    rng = random.Random(seed)
    m = cast(Any, pyo.ConcreteModel())
    m.x = pyo.Var(range(n + nv_extra), initialize=1.0)
    for j in rng.sample(range(n + nv_extra), 2):
        m.x[j].fix(1.0)
    m.c = pyo.Constraint(range(n))
    for i in range(n):
        cols = [rng.randrange(n + nv_extra) for _ in range(rng.randint(1, 4))]
        m.c[i] = sum(m.x[j] for j in cols) + m.x[cols[0]] * m.x[cols[-1]] == 1
    return m


@pytest.mark.parametrize("seed", range(40))
def test_structural_matching_unchanged(seed):
    import pyomo.environ as pyo

    from equilibria.solver._closure_patches import structural_matching

    old = _old_structural_matching()
    n = 3 + seed % 9
    m = _pyomo_model(seed, n, nv_extra=seed % 3 * 2)  # nv = n-2, n, n+2
    cons = sorted(
        m.component_data_objects(pyo.Constraint, active=True), key=lambda c: c.name
    )
    free = sorted(
        (v for v in m.component_data_objects(pyo.Var) if not v.fixed),
        key=lambda v: v.name,
    )
    rng = random.Random(seed)
    forced = []
    for _ in range(rng.randint(0, 5)):
        kind = rng.random()
        eq = cons[rng.randrange(len(cons))].name if kind > 0.1 else "nope"
        var = free[rng.randrange(len(free))].name
        forced.append((eq, var, rng.random() < 0.5) if kind > 0.5 else (eq, var))
    try:
        want = [v.name for v in old(cons, free, forced_pairs=forced)]
    except IndexError:  # more columns than rows: the old code fails, so must the new
        with pytest.raises(IndexError):
            structural_matching(cons, free, forced_pairs=forced)
        return
    assert [
        v.name for v in structural_matching(cons, free, forced_pairs=forced)
    ] == want
