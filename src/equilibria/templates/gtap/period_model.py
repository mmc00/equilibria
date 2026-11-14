"""GTAP's model with periods, as its steps see it (ADR-0011).

``GTAPPeriodModel`` is the core ``PeriodModel`` plus what GTAP's period recipe
needs that is not a cell edit: the reference fixing of a single-period model,
rebuilding a row (altertax ``and``/``ava``), solving a row for its own variable
(the welfare tail), writing a shock. Each backend implements it; the steps in
``templates/gtap/steps.py`` use only these operations.

The Pyomo implementation is ``templates/gtap/pyomo_periods.GTAPPyomoPeriodModel``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from equilibria.core.period_model import Index, PeriodModel


@dataclass(frozen=True)
class Reference:
    """What a single-period model with the period's closure fixes and bounds,
    by variable, with indices without the period."""

    fixed: dict[str, dict[Index, float]] = field(default_factory=dict)
    bounds: dict[str, dict[Index, tuple[float | None, float | None]]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class Staged:
    """The closure and parameters the next solve of ``period`` uses
    (``GTAPPeriodModel.stage``)."""

    closure: Any
    params: Any
    gtap_mode: bool
    period: str


class GTAPPeriodModel(PeriodModel, Protocol):
    def exogenous(self) -> frozenset[tuple[str, Index]]:
        """The instrument cells exogenous in the shock period (an ``@overwrite``
        hook may free some there; in base and check every instrument is
        exogenous: see ``exogenous_cells``)."""
        ...

    def exogenous_cells(self, name: str, period: str) -> frozenset[Index]:
        """The cells of ``name`` that are exogenous instruments in ``period``:
        never freed, re-seeded or re-bounded by a step."""
        ...

    @property
    def residual_region(self) -> str: ...

    @property
    def base_calibrated(self) -> bool:
        """The model was built with the settled seed (F3.5)."""
        ...

    @property
    def settled_seed(self) -> Mapping[str, Mapping[Any, Any]] | None:
        """``{variable: {index without period: level}}`` from the settle, or None."""
        ...

    @property
    def staged(self) -> Staged | None: ...

    def stage(self, closure: Any, params: Any, *, gtap_mode: bool, period: str) -> None:
        """Record the closure and parameters the next solve (of ``period``) uses."""
        ...

    def record_solve(self, result: Mapping[str, Any]) -> None:
        """The solver's result for the staged period; ``residual`` reports its
        ``"residual"``."""
        ...

    def residual(self, period: str) -> float:
        """The residual the last solve measured, when it solved ``period``;
        NaN when the period was not solved (GTAP does not re-evaluate its rows:
        on 20x41 that costs ~10 s a period, and the old driver reported the
        solver's residual)."""
        ...

    def sp_reference(
        self, closure: Any, params: Any, period: str, *, fresh: bool
    ) -> Reference:
        """The single-period reference with ``closure``. ``fresh=False`` reuses
        the last one while the closure does not change (and shares it across
        runs on the same ``params``); ``fresh=True`` builds it anew."""
        ...

    def recalibrate_and_ava(self, period: str, prior: str, params: Any) -> int:
        """altertax: rebuild ``eq_nd``/``eq_va`` of ``period`` with and/ava from
        ``prior``'s levels. Returns the rows rebuilt."""
        ...

    def refresh_fisher_aggregates(self) -> None:
        """Recompute the named Fisher aggregates from the sums they define."""
        ...

    def solve_welfare_tail(self, period: str) -> int:
        """Solve each muted welfare row of ``period`` for its own variable."""
        ...

    def recompute_ifsub_report_vars(self, period: str, params: Any) -> int:
        """ifSUB: recompute the report variables from the solved levels."""
        ...

    def write_shock(self, shock: Mapping[str, Mapping[Index, float]]) -> None:
        """Write ``{instrument: {cell: %}}`` into the shock period."""
        ...

    def write_default_tariff(self, *, gtap_mode: bool) -> None: ...

    def shock_labels(self) -> list[str]:
        """``name(cell)`` of each shock cell that differs from the check."""
        ...

    def check_endowment_shock_entered(self) -> None:
        """Raise if an endowment shock did not reach the solution."""
        ...

    def debug_before_solve(self, period: str) -> None: ...
