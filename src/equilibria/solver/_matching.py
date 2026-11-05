"""Hopcroft-Karp row/column matching on plain integer adjacency (no Pyomo).

The one algorithm behind ``structural_matching`` (Pyomo path) and ``square()`` (family
path): both build the same ordered adjacency, so both get the same pairs. The visiting
order is part of the contract (ties are broken by it), so the DFS is iterative but
visits exactly what the recursive formulation visits.
"""

from __future__ import annotations

from collections import deque

_INF = 10**9


def match_rows(
    n: int,
    adjacency: list[list[int]],
    forced: list[tuple[int, int, bool]] | None = None,
    label: str = "",
) -> list[int]:
    """Column of each of ``n`` rows (``-1`` if none could be given).

    ``adjacency[r]`` lists the columns of row ``r`` in the order they are tried.
    ``forced`` are ``(row, col, hard)`` applied in order, a later pair overwriting an
    earlier one on the same row or column exactly as a plain assignment does: a soft
    pair that is not adjacent is skipped, a hard pair is kept even when not adjacent
    and its row is locked (no augmenting path reassigns it). After the maximum
    matching, unmatched rows take the unmatched columns ``0..n-1`` in order. Column
    indices of ``adjacency`` must be below ``n``; with more rows than columns the
    returned entries ``>= nv`` (the columns that do not exist) are the caller's to
    drop.
    """
    pair_left = [-1] * n
    pair_right = [-1] * n
    distance = [0] * n
    locked: set[int] = set()

    for r, c, hard in forced or ():
        if c not in adjacency[r] and not hard:
            continue
        pair_left[r] = c
        pair_right[c] = r
        if hard:
            locked.add(r)

    def bfs() -> bool:
        q: deque[int] = deque()
        found = False
        for u in range(n):
            if pair_left[u] == -1:
                distance[u] = 0
                q.append(u)
            else:
                distance[u] = _INF
        while q:
            u = q.popleft()
            for v in adjacency[u]:
                m = pair_right[v]
                if m in locked:
                    continue
                if m == -1:
                    found = True
                elif distance[m] == _INF:
                    distance[m] = distance[u] + 1
                    q.append(m)
        return found

    def dfs(root: int) -> bool:
        """The recursive ``dfs(u)`` with an explicit stack: frame = (row, next index)."""
        rows = [root]
        nxt = [0]
        while True:
            u = rows[-1]
            adj = adjacency[u]
            i = nxt[-1]
            descended = False
            while i < len(adj):
                m = pair_right[adj[i]]
                if m in locked:
                    i += 1
                    continue
                if m == -1:  # augmenting path found: assign from the deepest frame up
                    nxt[-1] = i
                    for k in range(len(rows) - 1, -1, -1):
                        v = adjacency[rows[k]][nxt[k]]
                        pair_left[rows[k]] = v
                        pair_right[v] = rows[k]
                    return True
                if distance[m] == distance[u] + 1:
                    nxt[-1] = i
                    rows.append(m)
                    nxt.append(0)
                    descended = True
                    break
                i += 1
            if descended:
                continue
            distance[u] = _INF  # row exhausted: back to the parent, next column
            rows.pop()
            nxt.pop()
            if not rows:
                return False
            nxt[-1] += 1

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
    if label:
        n_natural = sum(1 for u in range(n) if pair_left[u] != -1)
        print(f"[{label}] structural matching: {n_natural}/{n} pairs assigned")
    return pair_left
