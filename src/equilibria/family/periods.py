"""``PeriodModel`` over ``MPArrays`` (ADR-0011): the family backend's model.

Each operation is a masked write on the arrays of one variable (``VarArr``) or row
family (``RowArr``) and one period. Positions come from a per-period table cached
on the arrays (keys never change), the same ``_table`` the array recipe uses, so a
step pays one dict lookup per cell it names and no Python loop over the rest.

Bounds: an infinite side is reported as ``None`` (Pyomo's "no bound"), and
``free_cells`` sets both sides infinite (the array twin of a real domain).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from itertools import compress

import numpy as np

from equilibria.family.mp_arrays import MPArrays, RowArr, VarArr

PERIODS = ("base", "check", "shock")


def positions(obj: VarArr | RowArr, period: str) -> dict[tuple, int]:
    """Position of each cell of ``period``, by its key without the period. Cached
    on ``obj`` under the name ``prep_recipe._table`` uses, so both share it."""
    cache = obj.__dict__.setdefault("_table", {})
    if period not in cache:
        sel = np.flatnonzero(obj.period == period)
        cache[period] = {obj.keys[i][:-1]: int(i) for i in sel}
    return cache[period]


def _columns(obj: VarArr | RowArr, period: str) -> tuple[list[tuple], np.ndarray]:
    """``positions`` as two columns, its keys and their positions (cached beside
    it), so whole-period reads are numpy masks instead of a loop per cell."""
    cache = obj.__dict__.setdefault("_columns", {})
    if period not in cache:
        table = positions(obj, period)
        cache[period] = (list(table), np.fromiter(table.values(), dtype=np.intp))
    return cache[period]


def _at(table: Mapping[tuple, int], cells: Iterable[tuple]) -> np.ndarray:
    return np.fromiter(map(table.__getitem__, cells), dtype=np.intp)


def _side(x: float, inf: float) -> float | None:
    return None if x == inf else float(x)


class FamilyPeriodModel:
    """``PeriodModel`` over ``self.mp``; a solver may replace ``mp`` with new
    arrays (the squared copy it solved), the cells keep their keys."""

    backend = "family"

    def __init__(self, mp: MPArrays, periods: tuple[str, ...] = PERIODS) -> None:
        self.mp = mp
        self._periods = tuple(periods)

    @property
    def periods(self) -> tuple[str, ...]:
        return self._periods

    def exogenous(self) -> frozenset[tuple[str, tuple]]:
        first = self._periods[0]
        return frozenset(
            (name, k)
            for name in self.mp.exogenous
            if name in self.mp.vars
            for k in positions(self.mp.vars[name], first)
        )

    # ── one cell ───────────────────────────────────────────────────────────

    def _pos(self, name: str, index: tuple, period: str) -> tuple[VarArr, int]:
        v = self.mp.var(name)
        return v, positions(v, period)[tuple(index)]

    def cells(self, period: str) -> Iterator[tuple[str, tuple]]:
        for name, v in self.mp.vars.items():
            for k in positions(v, period):
                yield name, k

    def value(self, name: str, index: tuple, period: str) -> float:
        v, i = self._pos(name, index, period)
        return float(v.value[i])

    def set_value(self, name: str, index: tuple, period: str, value: float) -> None:
        v, i = self._pos(name, index, period)
        v.value[i] = value

    def fix(
        self, name: str, index: tuple, period: str, value: float | None = None
    ) -> None:
        v, i = self._pos(name, index, period)
        if value is not None:
            v.value[i] = value
        v.fixed[i] = True

    def unfix(self, name: str, index: tuple, period: str) -> None:
        v, i = self._pos(name, index, period)
        v.fixed[i] = False

    def is_fixed(self, name: str, index: tuple, period: str) -> bool:
        v, i = self._pos(name, index, period)
        return bool(v.fixed[i])

    def activate_rows(self, period: str) -> None:
        for r in self.mp.rows.values():
            r.active[:] = r.period == period

    def residual(self, period: str) -> float:
        raise NotImplementedError(
            "the family model does not evaluate its rows; GTAP's reports the solver's"
        )

    # ── a variable or a row family at a time ───────────────────────────────

    def var_names(self) -> tuple[str, ...]:
        return tuple(self.mp.vars)

    def row_names(self) -> tuple[str, ...]:
        return tuple(self.mp.rows)

    def has_var(self, name: str) -> bool:
        return name in self.mp.vars

    def has_row(self, name: str) -> bool:
        return name in self.mp.rows

    def var_cells(self, name: str, period: str) -> tuple[tuple, ...]:
        return tuple(positions(self.mp.var(name), period))

    def row_cells(self, name: str, period: str) -> tuple[tuple, ...]:
        return tuple(positions(self.mp.row(name), period))

    def values(self, name: str, period: str) -> dict[tuple, float]:
        keys, at = _columns(self.mp.var(name), period)
        return dict(zip(keys, self.mp.var(name).value[at].tolist(), strict=True))

    def set_values(self, name: str, period: str, levels: Mapping[tuple, float]) -> None:
        v = self.mp.var(name)
        v.value[_at(positions(v, period), levels)] = np.fromiter(
            levels.values(), dtype=float, count=len(levels)
        )

    def fix_cells(
        self, name: str, period: str, levels: Mapping[tuple, float | None]
    ) -> None:
        v = self.mp.var(name)
        table = positions(v, period)
        given = {k: x for k, x in levels.items() if x is not None}
        if given:
            v.value[_at(table, given)] = np.fromiter(
                given.values(), dtype=float, count=len(given)
            )
        v.fixed[_at(table, levels)] = True

    def unfix_cells(self, name: str, period: str, cells: Iterable[tuple]) -> None:
        v = self.mp.var(name)
        v.fixed[_at(positions(v, period), cells)] = False

    def _all_but(
        self, obj: VarArr | RowArr, period: str, keep: frozenset
    ) -> np.ndarray:
        """Positions of every cell of ``period`` but those in ``keep``."""
        _, at = _columns(obj, period)
        if not keep:
            return at
        if len(keep) >= len(at):
            return at[:0]
        out = np.ones(len(obj.keys), dtype=bool)
        out[_at(positions(obj, period), keep)] = False
        return at[out[at]]

    def fix_all(self, name: str, period: str, *, keep: frozenset = frozenset()) -> None:
        v = self.mp.var(name)
        v.fixed[self._all_but(v, period, keep)] = True

    def unfix_all(
        self, name: str, period: str, *, keep: frozenset = frozenset()
    ) -> None:
        v = self.mp.var(name)
        v.fixed[self._all_but(v, period, keep)] = False

    def fixed_cells(self, name: str, period: str) -> frozenset[tuple]:
        keys, at = _columns(self.mp.var(name), period)
        return frozenset(compress(keys, self.mp.var(name).fixed[at].tolist()))

    def bounds(
        self, name: str, period: str
    ) -> dict[tuple, tuple[float | None, float | None]]:
        v = self.mp.var(name)
        keys, at = _columns(v, period)
        return {
            k: (_side(lo, -np.inf), _side(hi, np.inf))
            for k, lo, hi in zip(
                keys, v.lb[at].tolist(), v.ub[at].tolist(), strict=True
            )
        }

    def set_bounds(
        self,
        name: str,
        period: str,
        bounds: Mapping[tuple, tuple[float | None, float | None]],
    ) -> None:
        v = self.mp.var(name)
        table = positions(v, period)
        for k, (lb, ub) in bounds.items():
            i = table[k]
            if lb is not None:
                v.lb[i] = lb
            if ub is not None:
                v.ub[i] = ub

    def free_cells(self, name: str, period: str, cells: Iterable[tuple]) -> None:
        v = self.mp.var(name)
        at = _at(positions(v, period), cells)
        v.lb[at] = -np.inf
        v.ub[at] = np.inf

    def active_rows(self, name: str, period: str) -> frozenset[tuple]:
        keys, at = _columns(self.mp.row(name), period)
        return frozenset(compress(keys, self.mp.row(name).active[at].tolist()))

    def set_rows_active(
        self, name: str, period: str, cells: Iterable[tuple], active: bool
    ) -> None:
        r = self.mp.row(name)
        r.active[_at(positions(r, period), cells)] = active

    def set_all_rows_active(self, name: str, period: str, active: bool) -> None:
        r = self.mp.row(name)
        r.active[_columns(r, period)[1]] = active
