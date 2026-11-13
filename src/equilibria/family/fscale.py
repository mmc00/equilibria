"""Per-row convergence scale of the Newton-TR relative residual, from the family records.

Pyomo's path scales row ``i`` by ``max(1, |x_j|)`` over the *unfixed* variables
``identify_variables(c.body, include_fixed=False)`` finds in it, structural zeros
included (``path_capi``, ``TR_RELTOL``). The family route reads the same set off the
recorded rows: the variable leaves of ``lhs`` and ``rhs`` as the rule wrote them, **before**
the zero folding ``compile_family`` does (a leaf multiplied by a fixed 0 is gone from the
compiled terms and the J pattern, but Pyomo still lists it). A leaf counts when its cell
is free; a NaN level is skipped, as Pyomo skips a variable without value.

``pyomo_fscale_from_photo`` recomputes Pyomo's number from a reference capture (photo B's
incidence, fixed flags and levels) so the two can be compared bitwise.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from equilibria.family.recorder import Node


def row_leaves(res: Any) -> set[tuple[str, int]]:
    """Variable leaves ``(var, flat index)`` of a recorded residual, both sides."""
    out: set[tuple[str, int]] = set()
    for side in (res.lhs, res.rhs):
        if isinstance(side, Node):
            out.update(side.var_leaves())
    return out


def row_name(family: str, idx: tuple) -> str:
    lab = ",".join(str(i) for i in idx)
    return f"{family}[{lab}]" if lab != "" else family


def record_fscale(
    records: Mapping[str, list],
    values: Mapping[str, np.ndarray],
    free: Mapping[str, np.ndarray],
    keep: Callable[[str], bool] | None = None,
) -> dict[str, float]:
    """``{row name: fscale}`` for the recorded rows.

    ``values[v]`` / ``free[v]`` are the dense level and free-cell arrays of variable
    ``v`` (``var_domains`` order, flat C-order index = the leaf's second element).
    ``keep(name)`` restricts the rows computed (skips the work for rows not wanted).
    """
    flat_v = {v: np.asarray(a).ravel() for v, a in values.items()}
    flat_f = {v: np.asarray(a).ravel() for v, a in free.items()}
    out: dict[str, float] = {}
    for fam, recs in records.items():
        for idx, res in recs:
            name = row_name(fam, idx)
            if keep is not None and not keep(name):
                continue
            mag = 1.0
            for v, flat in row_leaves(res):
                if flat_f[v][flat]:
                    x = abs(float(flat_v[v][flat]))
                    if x == x:  # NaN: Pyomo has no value for it
                        mag = max(mag, x)
            out[name] = mag
    return out


class LeafTable:
    """The variable leaves of every recorded row of a recording (family order, record
    order: the rows of its system), as flat positions into the concatenation of the
    variables' dense arrays in ``var_order``. Built once per recording; :meth:`fscale`
    is then ``record_fscale`` of every row as array operations."""

    def __init__(self, records: Mapping[str, list], sizes: Mapping[str, int]) -> None:
        self.var_order = list(sizes)
        self.sizes = dict(sizes)
        off, o = {}, 0
        for v in self.var_order:
            off[v] = o
            o += int(sizes[v])
        pos: list[int] = []
        counts: list[int] = []
        for recs in records.values():
            for _, res in recs:
                leaves = row_leaves(res)
                counts.append(len(leaves))
                pos.extend(off[v] + flat for v, flat in leaves)
        self.n_rows = len(counts)
        self.pos = np.asarray(pos, dtype=np.int64)
        self.owner = np.repeat(
            np.arange(self.n_rows, dtype=np.int64), np.asarray(counts, dtype=np.int64)
        )

    def fscale(
        self, values: Mapping[str, np.ndarray], free: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        """Per row ``max(1, |x_j|)`` over its free, non-NaN leaves (``record_fscale``)."""
        x = np.concatenate(
            [np.asarray(values[v], dtype=float).ravel() for v in self.var_order]
        )
        f = np.concatenate(
            [np.asarray(free[v], dtype=bool).ravel() for v in self.var_order]
        )
        a = np.abs(x[self.pos])
        use = f[self.pos] & (a == a)  # NaN: Pyomo has no value for it
        out = np.ones(self.n_rows)
        np.maximum.at(out, self.owner[use], a[use])
        return out


_LEAF_TABLES: list[tuple[Any, tuple, LeafTable]] = []  # (records, sizes, table)
_LEAF_TABLES_KEPT = 4


def leaf_table(
    records: Mapping[str, list], values: Mapping[str, np.ndarray]
) -> LeafTable:
    """The :class:`LeafTable` of ``records`` over the variables (and their sizes) of
    ``values``, kept for the recordings used last (the recordings live for the run)."""
    sizes = tuple((v, int(np.asarray(a).size)) for v, a in values.items())
    for rec, sz, table in _LEAF_TABLES:
        if rec is records and sz == sizes:
            return table
    table = LeafTable(records, dict(sizes))
    _LEAF_TABLES.append((records, sizes, table))
    del _LEAF_TABLES[:-_LEAF_TABLES_KEPT]
    return table


def pyomo_fscale_from_photo(b: Any, a2: Any) -> dict[str, float]:
    """Pyomo's fscale of every squared row of call photo ``b``, recomputed from it.

    ``b.incidence`` lists, per row of ``b.con_names``, the variables of its body
    (``include_fixed=True``); the free ones are read at photo B's fixed flags and
    levels. Photo B holds the active period only: a variable of another period is read
    from ``a2`` (the end of the same ``prepare``). Loop and ``max`` as in ``path_capi``.
    """
    cell: dict[str, tuple[float, bool]] = {}
    for photo in (a2, b):  # B last: it wins for the active period
        for name, vc in photo.vars.items():
            for k, v, fx in zip(vc.keys, vc.value, vc.fixed, strict=True):
                cell[f"{name}[{','.join(k)}]" if k else name] = (float(v), bool(fx))
    out: dict[str, float] = {}
    for con, names in zip(b.con_names, b.incidence, strict=True):
        mag = 1.0
        for n in names:
            v, fixed = cell[n]
            if not fixed and v == v:
                mag = max(mag, abs(v))
        out[con] = mag
    return out
