"""A model with periods, as the run machinery sees it (ADR-0011).

Steps and the driver talk to the model only through these operations, never to
Pyomo (or JAX) directly, so the same recipe can fly a model on any backend. A
cell is a variable name plus its index without the period (``()`` for a
scalar). Exogenous cells (fixed in the single-period model: instruments) are
never freed by the generic steps; a shock changes their value in its period.

The Pyomo implementation is ``equilibria.backends.pyomo_periods.PyomoPeriodModel``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
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

    # Per-variable and per-row operations. A model with many cells (GTAP 20x41:
    # ~395k per period) is edited a variable or a row family at a time, so an
    # array backend can do each call as one masked write. An index never carries
    # the period; a row not indexed by period never appears.

    def var_names(self) -> tuple[str, ...]: ...

    def row_names(self) -> tuple[str, ...]: ...

    def has_var(self, name: str) -> bool: ...

    def has_row(self, name: str) -> bool: ...

    def var_cells(self, name: str, period: str) -> tuple[Index, ...]:
        """The cells of variable ``name`` in ``period``, in index order."""
        ...

    def row_cells(self, name: str, period: str) -> tuple[Index, ...]:
        """The cells of row family ``name`` in ``period``, in index order."""
        ...

    def values(self, name: str, period: str) -> dict[Index, float]:
        """Every cell's level (NaN where the cell has none)."""
        ...

    def set_values(self, name: str, period: str, levels: Mapping[Index, float]) -> None:
        """Set levels as given (NaN allowed; no domain validation)."""
        ...

    def fix_cells(
        self, name: str, period: str, levels: Mapping[Index, float | None]
    ) -> None:
        """Fix each cell at its level, or at its current level for ``None``."""
        ...

    def unfix_cells(self, name: str, period: str, cells: Iterable[Index]) -> None: ...

    def fix_all(
        self, name: str, period: str, *, keep: frozenset[Index] = frozenset()
    ) -> None:
        """Fix every cell of the period at its current level, but those in ``keep``
        (cells of the period; one call for what ``fix_cells`` would need a mapping
        of every cell for)."""
        ...

    def unfix_all(
        self, name: str, period: str, *, keep: frozenset[Index] = frozenset()
    ) -> None:
        """Unfix every cell of the period but those in ``keep``."""
        ...

    def fixed_cells(self, name: str, period: str) -> frozenset[Index]: ...

    def bounds(
        self, name: str, period: str
    ) -> dict[Index, tuple[float | None, float | None]]:
        """Every cell's ``(lower, upper)``; ``None`` for no bound."""
        ...

    def set_bounds(
        self,
        name: str,
        period: str,
        bounds: Mapping[Index, tuple[float | None, float | None]],
    ) -> None:
        """Set the given sides; a ``None`` side is left as it is."""
        ...

    def free_cells(self, name: str, period: str, cells: Iterable[Index]) -> None:
        """No bounds and a real domain (a non-negative cell loses its floor)."""
        ...

    def active_rows(self, name: str, period: str) -> frozenset[Index]: ...

    def set_rows_active(
        self, name: str, period: str, cells: Iterable[Index], active: bool
    ) -> None: ...

    def set_all_rows_active(self, name: str, period: str, active: bool) -> None:
        """``set_rows_active`` on every row of the family in the period."""
        ...
