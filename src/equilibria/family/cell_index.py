"""Integer ids for the cells and rows of one run's multiperiod arrays.

``prep_square.square`` used to build the name of every active row and free column, sort
the name lists and look names up in dicts, once per solve call. The names do not change
within a run, so ``CellIndex`` builds them once and ranks them: ``row_rank`` /
``col_rank`` give the position of every row / variable cell in the global sort of the
names (Python's string order, code point by code point), so the sorted active subset of
a call is ``argsort`` of the ranks of a mask, with exactly the order ``list.sort`` of
the subset's names gives.

Ids are global: a row is ``row_off[family] + i`` and a variable cell
``var_off[name] + i`` for position ``i`` in the family's / variable's arrays.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from equilibria.family.mp_arrays import MPArrays, _name


def _ranks(names: list[str]) -> np.ndarray:
    order = sorted(range(len(names)), key=lambda i: names[i])
    rank = np.empty(len(names), dtype=np.int64)
    rank[np.asarray(order, dtype=np.int64)] = np.arange(len(names), dtype=np.int64)
    return rank


class CellIndex:
    def __init__(self, mp: MPArrays) -> None:
        self._keys = {n: v.keys for n, v in mp.vars.items()}
        self._row_keys = {n: r.keys for n, r in mp.rows.items()}
        self.var_off: dict[str, int] = {}
        self.row_off: dict[str, int] = {}
        self.col_names: list[str] = []
        self.row_names: list[str] = []
        for n, v in mp.vars.items():
            self.var_off[n] = len(self.col_names)
            self.col_names += [_name(n, k) for k in v.keys]
        for n, r in mp.rows.items():
            self.row_off[n] = len(self.row_names)
            self.row_names += [_name(n, k) for k in r.keys]
        self.col_rank = _ranks(self.col_names)
        self.row_rank = _ranks(self.row_names)
        self._period_ids: dict[tuple[str, str, str], np.ndarray] = {}
        self._var_pos: dict[str, dict[tuple, int]] = {}
        self._name_gid: dict[str, dict[str, int]] = {}
        self.cache: dict[Any, Any] = {}  # derived per-run tables (see ArrayIncidence)

    def fits(self, mp: MPArrays) -> bool:
        """Whether ``mp`` has the very key lists this index was built from (a copy of
        the arrays shares them)."""
        return (
            mp.vars.keys() == self._keys.keys()
            and mp.rows.keys() == self._row_keys.keys()
            and all(mp.vars[n].keys is k for n, k in self._keys.items())
            and all(mp.rows[n].keys is k for n, k in self._row_keys.items())
        )

    def period_ids(self, kind: str, name: str, period: str, arr: Any) -> np.ndarray:
        """Positions in the family / variable ``name`` of the cells of ``period``."""
        key = (kind, name, period)
        hit = self._period_ids.get(key)
        if hit is None:
            hit = self._period_ids[key] = np.flatnonzero(arr.period == period)
        return hit

    def var_pos(self, name: str, arr: Any) -> dict[tuple, int]:
        hit = self._var_pos.get(name)
        if hit is None:
            hit = self._var_pos[name] = {k: i for i, k in enumerate(arr.keys)}
        return hit

    def gid_of_name(self, name: str, mp: MPArrays) -> int | None:
        """Global id of the variable cell called ``name`` (``None``: no such cell)."""
        comp = name.partition("[")[0]
        table = self._name_gid.get(comp)
        if table is None:
            v = mp.vars.get(comp)
            table = {}
            if v is not None:
                off = self.var_off[comp]
                table = {self.col_names[off + i]: off + i for i in range(len(v.keys))}
            self._name_gid[comp] = table
        return table.get(name)


class SquaredIds:
    """Global ids of the rows and columns of a ``Squared`` (paired order) and the values
    captured when it was squared, so the system is built without name strings.

    ``row_gid[k]`` / ``col_gid[k]`` are ``CellIndex`` ids, -1 for a name the arrays do
    not have. ``snap`` is ``(var name, positions, values)`` per variable: the cells of
    the squared period as they were when squaring."""

    def __init__(self, index: CellIndex, row_gid, col_gid, snap, x0=None) -> None:
        self.index = index
        self.row_gid = np.asarray(row_gid, dtype=np.int64)
        self.col_gid = np.asarray(col_gid, dtype=np.int64)
        self.snap = snap
        self._x0 = x0
        self._row_pos: np.ndarray | None = None
        self._col_pos: np.ndarray | None = None
        self._vals: tuple[np.ndarray, np.ndarray] | None = None

    @classmethod
    def from_names(cls, sq: Any, mp: MPArrays, index: CellIndex | None = None):
        """Ids of a ``Squared`` built some other way (names only)."""
        index = index or CellIndex(mp)
        rows = {n: i for i, n in enumerate(index.row_names)}
        cols = {n: i for i, n in enumerate(index.col_names)}
        out = cls(
            index,
            [rows.get(n, -1) for n in sq.row_names],
            [cols.get(n, -1) for n in sq.col_names],
            [],
        )
        vals = np.zeros(len(index.col_names))
        has = np.zeros(len(index.col_names), dtype=bool)
        for name, cell in sq.cells.items():
            g = cols.get(name)
            if g is not None:
                vals[g], has[g] = cell[0], True
        out._vals = (vals, has)
        out._x0 = np.array([sq.cells[n][0] for n in sq.col_names], dtype=float)
        return out

    @property
    def row_pos(self) -> np.ndarray:
        """Squared row of every global row id (-1: not a row of the system)."""
        if self._row_pos is None:
            pos = np.full(len(self.index.row_names), -1, dtype=np.int64)
            ok = self.row_gid >= 0
            pos[self.row_gid[ok]] = np.flatnonzero(ok)
            self._row_pos = pos
        return self._row_pos

    @property
    def col_pos(self) -> np.ndarray:
        """Squared column of every global variable id (-1: not a column)."""
        if self._col_pos is None:
            pos = np.full(len(self.index.col_names), -1, dtype=np.int64)
            ok = self.col_gid >= 0
            pos[self.col_gid[ok]] = np.flatnonzero(ok)
            self._col_pos = pos
        return self._col_pos

    def values(self) -> tuple[np.ndarray, np.ndarray]:
        """(value, present) of every global variable id at squaring time."""
        if self._vals is None:
            vals = np.zeros(len(self.index.col_names))
            has = np.zeros(len(self.index.col_names), dtype=bool)
            for name, idx, value in self.snap:
                g = self.index.var_off[name] + idx
                vals[g], has[g] = value, True
            self._vals = (vals, has)
        return self._vals

    def x0(self) -> np.ndarray:
        """Values of the squared columns; ``KeyError`` for one with no captured value."""
        if self._x0 is None:
            vals, has = self.values()
            g = self.col_gid
            ok = g >= 0
            ok[ok] = has[g[ok]]
            if not ok.all():
                k = int(np.flatnonzero(~ok)[0])
                name = self.index.col_names[g[k]] if g[k] >= 0 else f"column {k}"
                raise KeyError(f"{name}: free column missing from sq.cells")
            self._x0 = vals[g]
        return self._x0
