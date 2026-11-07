"""The multiperiod GTAP state on arrays, and the squared system the solver receives.

``MPArrays`` holds, per Var component, every cell of every period (value, fixed,
lb, ub) and, per Constraint component, every cell's active flag: what the Pyomo
``ConcreteModel`` carries for the period preparation, without expressions. Cells
keep their Pyomo index tuples (period last), so names match Pyomo's.

``MPArrays.build`` makes that state straight from the composed single-period state
(``compose_block_state``), replacing the reflection ``build_block_model`` runs over a
Pyomo model; ``tests/family/test_mp_build.py`` ties it to the photo of the Pyomo model.

``Squared`` is the per-period system right before Newton: active rows and free
columns in paired order, plus every cell of the active period. ``compare`` is the
parity rule of the array port: same rows, same columns, same cells (1e-9
relative, identical fixed flags) and a VALID pairing, not an identical one (see
``compare``).
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
from collections.abc import Callable
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
    # What ``build_block_model`` leaves on the Pyomo model besides Vars and Constraints.
    exogenous: frozenset = frozenset()  # registered ShockBlock instruments
    endogenous_instrument_cells: dict = field(default_factory=dict)
    targets: dict = field(default_factory=dict)
    shock_rows: frozenset = frozenset()
    fctts: Any = None
    residual_region: str | None = None
    gtap_mode: bool = False  # the driver sets it when it runs
    base_calibrated: bool = False
    settled_seed: Any = None

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

    @classmethod
    def build(
        cls,
        state: Any,
        backend: Any,
        cross: Any,
        *,
        params: Any,
        base_calibrated: bool = False,
    ) -> MPArrays:
        """The multiperiod state of ``build_block_model``, from the composed state.

        ``state`` is the ``StateModel`` of ``compose_block_state``, ``backend`` the
        single-period recording and ``cross`` the cross-period one. Reproduces the
        reflection (``GTAPBlockMultiPeriodModel.build_vars`` and
        ``build_equations_all_periods``) cell by cell:

        * every single-period Var in each period, at its level (1.0 when it has none),
          lb 0 only if Pyomo declares it ``NonNegativeReals``, no other bound; free
          except the ShockBlock / ``@overwrite`` instruments, fixed in every period;
        * every single-period row in each period, active; the Fisher auxiliaries and
          the ifSUB report rows are absent under ifSUB=1 (Pyomo drops them from the
          reflected SP; under ifSUB=0 they are present and active);
        * the rows of ``cross`` replace the single-period ones of the same name, and its
          ``mq_*`` Vars (free, at the value of the sum they define) are added;
          ``pwfact[base]`` is fixed at 1.

        Row cells come in recording order, not Pyomo's (sorted within a period)."""
        from equilibria.blocks.gtap import _derived_params as dp
        from equilibria.blocks.gtap.closure import FISHER_AUX_EQS, FISHER_AUX_VARS
        from equilibria.blocks.gtap.overwrite import overwrite
        from equilibria.blocks.gtap.periods import PERIODS
        from equilibria.blocks.gtap.shock import SHOCK_INSTRUMENTS
        from equilibria.family.cross_period import mq_keys, refresh_aggregates
        from equilibria.templates.gtap.instruments import is_free_cell

        if overwrite.active():
            # ``shock_only`` would keep the hooks' rows in the shock period only and
            # anchor their targets to the check; not ported.
            raise NotImplementedError("family backend: @overwrite hooks")
        periods = list(PERIODS)
        n_t = len(periods)
        endo = dict(getattr(state, "_endogenous_instrument_cells", {}) or {})
        targets = dict(getattr(state, "_targets", {}) or {})
        instruments = {*SHOCK_INSTRUMENTS, *targets}

        vars_: dict[str, VarArr] = {}
        for name, doms in state.var_domains.items():
            if name in FISHER_AUX_VARS:
                continue
            keys = [
                (*k, t)
                for k in itertools.product(*(state.sets[d] for d in doms))
                for t in periods
            ]
            value = np.repeat(np.asarray(state.values[name], dtype=float).ravel(), n_t)
            value = np.where(np.isnan(value), 1.0, value)
            fixed = np.zeros(len(keys), dtype=bool)
            if name in instruments:
                libres = endo.get(name, frozenset())
                fixed[:] = True
                if libres:
                    for i, k in enumerate(keys):
                        if is_free_cell(k, libres):
                            fixed[i] = False
            lb = np.full(len(keys), 0.0 if name in state.nonneg_vars else -np.inf)
            vars_[name] = VarArr(keys, value, fixed, lb, np.full(len(keys), np.inf))

        rows: dict[str, RowArr] = {}
        for fam, recs in backend.records.items():
            if fam in FISHER_AUX_EQS or fam in cross.records:
                continue
            keys = [(*idx, t) for t in periods for idx, _ in recs]
            rows[fam] = RowArr(keys, np.ones(len(keys), dtype=bool))
        for fam, recs in cross.records.items():
            keys = [tuple(idx) for idx, _ in recs]
            rows[fam] = RowArr(keys, np.ones(len(keys), dtype=bool))

        for name, cells in mq_keys(list(state.sets["r"]), periods).items():
            n = len(cells)
            vars_[name] = VarArr(
                list(cells),
                np.zeros(n),
                np.zeros(n, dtype=bool),
                np.full(n, -np.inf),
                np.full(n, np.inf),
            )
        # ``build_equations_fisher`` anchors the world factor price index by fixing the
        # Var (a row would duplicate eq_pnum): ``pwfact[base] = 1``.
        pwfact = vars_.get("pwfact")
        if pwfact is not None and ("base",) in pwfact.keys:
            i = pwfact.pos(("base",))
            pwfact.value[i] = 1.0
            pwfact.fixed[i] = True

        mp = cls(
            vars_,
            rows,
            exogenous=frozenset(getattr(state, "_exogenous_instruments", ())),
            endogenous_instrument_cells=endo,
            targets=targets,
            shock_rows=frozenset(getattr(state, "_shock_rows", ()) or ()),
            fctts=dp.fctts_data(params, params.sets),
            residual_region=getattr(state, "_residual_region", None),
            base_calibrated=base_calibrated,
        )
        refresh_aggregates(cross, mp)
        return mp

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
        return dataclasses.replace(self, vars=vars_, rows=rows)

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


class Squared:
    """Active rows and free columns in paired order (``row_names[k]`` <->
    ``col_names[k]``) and every cell of the active period.

    ``cells`` may be given as a callable that builds the dict on first access (the
    values it reads were captured when the system was squared).

    ``ids`` is a snapshot taken by ``square``: it is dropped when ``cells`` is assigned,
    but not when ``row_names`` / ``col_names`` are mutated in place. Do not mutate them
    after ``square``; ``SquaredSystem.build`` only guards against a length change (and
    rebuilds the ids from the names then)."""

    def __init__(
        self,
        row_names: list[str],
        col_names: list[str],
        cells: dict[str, tuple[float, bool, float, float]] | Callable[[], dict],
    ) -> None:
        self.row_names = row_names
        self.col_names = col_names
        self._cells = cells
        # integer ids of the rows / columns (``cell_index.SquaredIds``), set by
        # ``square``; ``SquaredSystem.build`` reads them instead of the name strings
        self.ids: Any = None

    @property
    def cells(self) -> dict[str, tuple[float, bool, float, float]]:
        if callable(self._cells):
            self._cells = self._cells()
        return self._cells

    @cells.setter
    def cells(self, value: dict) -> None:
        self._cells = value
        self.ids = None  # the ids describe the cells square captured

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Squared):
            return NotImplemented
        return (self.row_names, self.col_names, self.cells) == (
            other.row_names,
            other.col_names,
            other.cells,
        )

    __hash__ = None

    def __repr__(self) -> str:
        return f"Squared({len(self.row_names)} rows, {len(self.col_names)} cols)"


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
