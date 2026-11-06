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
