"""The multiperiod GTAP state on arrays, and the squared system the solver receives.

``MPArrays`` holds, per Var component, every cell of every period (value, fixed,
lb, ub) and, per Constraint component, every cell's active flag: what the Pyomo
``ConcreteModel`` carries for the period preparation, without expressions. Cells
keep their Pyomo index tuples (period last), so names match Pyomo's.

``Squared`` is the per-period system right before Newton: active rows and free
columns in paired order, plus every cell of the active period. ``compare`` is the
parity rule of the array port: same rows, same columns, same cells (1e-9
relative, identical fixed flags) and a VALID pairing, not an identical one (see
``compare``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

RTOL = 1e-9


@dataclass
class VarArr:
    keys: list[tuple]
    value: np.ndarray
    fixed: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    period: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.period = np.array([k[-1] if k else "" for k in self.keys], dtype=object)
        self._pos: dict[tuple, int] | None = None

    def pos(self, key: tuple) -> int:
        """Position of a cell; ``KeyError`` names it if it does not exist."""
        if self._pos is None:
            self._pos = {k: i for i, k in enumerate(self.keys)}
        return self._pos[key]


@dataclass
class RowArr:
    keys: list[tuple]
    active: np.ndarray
    period: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.period = np.array([k[-1] if k else "" for k in self.keys], dtype=object)


def _name(comp: str, key: tuple) -> str:
    return f"{comp}[{','.join(key)}]" if key else comp


@dataclass
class MPArrays:
    vars: dict[str, VarArr]
    rows: dict[str, RowArr]

    @classmethod
    def from_photo(cls, photo: Any) -> MPArrays:
        return cls(
            vars={
                n: VarArr(
                    list(c.keys),
                    c.value.copy(),
                    c.fixed.copy(),
                    c.lb.copy(),
                    c.ub.copy(),
                )
                for n, c in photo.vars.items()
            },
            rows={
                n: RowArr(list(r.keys), r.active.copy()) for n, r in photo.rows.items()
            },
        )

    def copy(self) -> MPArrays:
        """Independent arrays; the (immutable) keys and their lookup caches are shared."""
        vars_ = {}
        for n, v in self.vars.items():
            c = copy.copy(v)
            c.value, c.fixed, c.lb, c.ub = (
                v.value.copy(),
                v.fixed.copy(),
                v.lb.copy(),
                v.ub.copy(),
            )
            vars_[n] = c
        rows = {}
        for n, r in self.rows.items():
            c = copy.copy(r)
            c.active = r.active.copy()
            rows[n] = c
        return MPArrays(vars_, rows)

    def var(self, name: str) -> VarArr:
        """The component, or ``KeyError``: a step must never skip a missing one."""
        try:
            return self.vars[name]
        except KeyError:
            raise KeyError(f"variable component {name!r} missing") from None

    def row(self, name: str) -> RowArr:
        try:
            return self.rows[name]
        except KeyError:
            raise KeyError(f"constraint component {name!r} missing") from None

    def name(self, var: str, i: int) -> str:
        return _name(var, self.vars[var].keys[i])

    def cells(self, var: str, period: str | None = None) -> np.ndarray:
        v = self.var(var)
        if period is None:
            return np.arange(len(v.keys))
        return np.flatnonzero(v.period == period)


@dataclass
class Squared:
    row_names: list[str]
    col_names: list[str]  # paired: row_names[k] <-> col_names[k]
    cells: dict[str, tuple[float, bool, float, float]]  # active period, all cells


def squared_from_photo(b: Any) -> Squared:
    cells = {}
    for n, c in b.vars.items():
        for k, v, f, lo, up in zip(c.keys, c.value, c.fixed, c.lb, c.ub, strict=True):
            cells[_name(n, k)] = (float(v), bool(f), float(lo), float(up))
    return Squared(list(b.con_names), list(b.var_names), cells)


def _close(a: float, b: float) -> bool:
    if np.isnan(a) or np.isnan(b):
        return bool(np.isnan(a) and np.isnan(b))
    if np.isinf(a) or np.isinf(b):
        return a == b
    return abs(a - b) <= RTOL * max(1.0, abs(b))


def _few(items) -> str:
    items = sorted(items)
    return ", ".join(items[:5]) + (f" (+{len(items) - 5})" if len(items) > 5 else "")


def compare(
    got: Squared,
    want: Squared,
    incidence: dict[str, set[str]],
    hard: set[tuple[str, str]],
) -> list[str]:
    """Differences between ``got`` and the reference ``want`` (empty = parity).

    A pairing is valid when it is a bijection, keeps every ``hard`` GAMS pair
    (locked; may lie outside the row's incidence) and pairs no fewer rows inside
    their incidence than the reference: the rest are leftover columns, as in
    Pyomo's ``structural_matching``."""
    msgs = []
    gr, wr = set(got.row_names), set(want.row_names)
    if gr != wr:
        msgs.append(f"rows: missing {_few(wr - gr)}; extra {_few(gr - wr)}")
    gc, wc = set(got.col_names), set(want.col_names)
    if gc != wc:
        msgs.append(f"cols: missing {_few(wc - gc)}; extra {_few(gc - wc)}")

    if got.cells.keys() != want.cells.keys():
        msgs.append(
            f"cells: missing {_few(want.cells.keys() - got.cells.keys())}; "
            f"extra {_few(got.cells.keys() - want.cells.keys())}"
        )
    bad: dict[str, list[str]] = {"value": [], "fixed": [], "lb": [], "ub": []}
    for name in got.cells.keys() & want.cells.keys():
        (gv, gf, gl, gu), (wv, wf, wl, wu) = got.cells[name], want.cells[name]
        if gf != wf:
            bad["fixed"].append(name)
        if not _close(gv, wv):
            bad["value"].append(name)
        if not _close(gl, wl):
            bad["lb"].append(name)
        if not _close(gu, wu):
            bad["ub"].append(name)
    msgs += [f"{what}: {_few(names)}" for what, names in bad.items() if names]

    if len(got.row_names) != len(got.col_names):
        msgs.append(f"pairing: {len(got.row_names)} rows vs {len(got.col_names)} cols")
    else:
        if len(set(got.col_names)) != len(got.col_names):
            msgs.append("pairing: a column is paired twice")
        pairs = set(zip(got.row_names, got.col_names, strict=True))
        lost = [f"{r}<->{c}" for r, c in hard - pairs if r in gr]
        if lost:
            msgs.append(f"hard pairs not kept: {_few(lost)}")

        def inside(names: Squared) -> int:
            return sum(
                1
                for r, c in zip(names.row_names, names.col_names, strict=True)
                if c in incidence.get(r, ()) or (r, c) in hard
            )

        n_got, n_want = inside(got), inside(want)
        if n_got < n_want:
            msgs.append(
                f"pairing: {n_got} rows paired inside their incidence, "
                f"the reference pairs {n_want}"
            )
    return msgs
