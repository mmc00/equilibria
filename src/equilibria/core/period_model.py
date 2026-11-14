"""A model with periods, as the run machinery sees it (ADR-0011).

Steps and the driver talk to the model only through these operations, never to
Pyomo (or JAX) directly, so the same recipe can fly a model on any backend. A
cell is a variable name plus its index without the period (``()`` for a
scalar). Exogenous cells (fixed in the single-period model: instruments) are
never freed by the generic steps; a shock changes their value in its period.

The Pyomo implementation is ``equilibria.backends.pyomo_periods.PyomoPeriodModel``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

Index = tuple
Cell = tuple[str, Index]


class PeriodModel(Protocol):
    @property
    def periods(self) -> tuple[str, ...]:
        """The period names, in order."""
        ...

    def exogenous(self) -> frozenset[Cell]:
        """The cells fixed in the single-period model (instruments)."""
        ...

    def cells(self, period: str) -> Iterator[Cell]:
        """Every variable cell of ``period``."""
        ...

    def value(self, name: str, index: Index, period: str) -> float: ...

    def set_value(self, name: str, index: Index, period: str, value: float) -> None:
        """The cell's level (its start point, or its fixed value if fixed)."""
        ...

    def fix(
        self, name: str, index: Index, period: str, value: float | None = None
    ) -> None:
        """Fix the cell, at ``value`` or at its current level."""
        ...

    def unfix(self, name: str, index: Index, period: str) -> None: ...

    def is_fixed(self, name: str, index: Index, period: str) -> bool: ...

    def activate_rows(self, period: str) -> None:
        """Leave active the equations of ``period`` only."""
        ...

    def residual(self, period: str) -> float:
        """Largest ``|body - bound|`` over the active equations of ``period``."""
        ...
