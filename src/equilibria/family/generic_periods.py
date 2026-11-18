"""``PeriodModel`` of any blocks template on the family arrays (ADR-0011).

``FamilyPeriodModel`` edits cells; GTAP's subclass reports the solver's residual. Here
the rows are evaluated: one period at a time, the recorded single-period rows are
compiled for that period's layout (its free cells are columns, the rest fixed slots),
so a residual and a Newton system come from the same ``FamilySystem``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from equilibria.backends.family_backend import FamilyBackend
from equilibria.family.core import FamilySystem
from equilibria.family.layout import VarLayout
from equilibria.family.mp_arrays import MPArrays
from equilibria.family.periods import PERIODS, FamilyPeriodModel
from equilibria.family.state import StateModel


@dataclass
class PeriodSystem:
    """One period's rows over its free cells.

    ``cols[name]`` are the positions in ``mp.vars[name]`` of the variable's free
    cells, in column order of ``layout.slots[name]``; ``active`` masks the rows of
    ``system`` (family order, cells in recording order) active in the period."""

    period: str
    system: FamilySystem
    layout: VarLayout
    cols: dict[str, tuple[np.ndarray, np.ndarray]]  # name -> (slots, mp positions)
    active: np.ndarray

    def x0(self) -> np.ndarray:
        return np.asarray(self.layout.free_values, dtype=float)


class GenericFamilyPeriodModel(FamilyPeriodModel):
    """``PeriodModel`` over ``mp`` whose rows are the recording in ``backend``."""

    def __init__(
        self,
        mp: MPArrays,
        backend: FamilyBackend,
        state: StateModel,
        periods: tuple[str, ...] = PERIODS,
    ) -> None:
        super().__init__(mp, periods)
        self.backend = backend
        self.state = state
        self.last_solve: dict[str, Any] | None = None

    def period_system(self, period: str) -> PeriodSystem:
        """The rows of ``period`` compiled for its current values and fixed cells."""
        values: dict[str, np.ndarray] = {}
        free: dict[str, np.ndarray] = {}
        where: dict[str, np.ndarray] = {}
        for name in self.backend.var_domains:
            v = self.mp.vars[name]
            at = np.flatnonzero(v.period == period)
            shape = self.state.values[name].shape
            values[name] = v.value[at].reshape(shape)
            free[name] = ~v.fixed[at].reshape(shape)
            where[name] = at
        for p in self.backend.param_domains:
            values[p] = self.state.find_component(p).data
            free[p] = np.zeros(values[p].shape, dtype=bool)
        domains = {**self.backend.var_domains, **self.backend.param_domains}
        layout = VarLayout.from_state(self.backend.sets, domains, values, free)
        system = self.backend.system(layout)
        cols = {}
        for name in self.backend.var_domains:
            slots = layout.slots[name].ravel()
            mask = slots < layout.n_cols
            cols[name] = (slots[mask], where[name][mask])
        active = np.concatenate(
            [
                self.mp.rows[fam].active[self.mp.rows[fam].period == period]
                for fam in self.backend.records
            ]
        )
        return PeriodSystem(period, system, layout, cols, active)

    def residual(self, period: str) -> float:
        """Largest |row| of ``period`` at the current cell values (its rows, active
        or not: a period's rows are its equations)."""
        ps = self.period_system(period)
        f = ps.system.F(ps.x0())
        return float(np.max(np.abs(f))) if f.size else 0.0

    def write_back(self, ps: PeriodSystem, x: np.ndarray) -> None:
        """Put the free-column values ``x`` back into the period's cells."""
        for name, (slots, at) in ps.cols.items():
            self.mp.vars[name].value[at] = x[slots]
