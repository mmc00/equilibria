"""IEEM simulation adapter (state-based placeholder until native solver integration)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from equilibria.simulations.adapters.mapping import (
    MappingAdapter,
    StateCompareFn,
    StateIndicatorsFn,
    StateSolveFn,
)
from equilibria.simulations.types import ShockDefinition


class IEEMAdapter(MappingAdapter):
    """IEEM adapter using generic state-scenario mechanics."""

    def __init__(
        self,
        *,
        base_state: dict[str, Any] | None = None,
        state_loader: Callable[[], dict[str, Any]] | None = None,
        shock_definitions: list[ShockDefinition] | None = None,
        solve_fn: StateSolveFn | None = None,
        compare_fn: StateCompareFn | None = None,
        key_indicators_fn: StateIndicatorsFn | None = None,
    ) -> None:
        super().__init__(
            model_label="ieem",
            base_state=base_state,
            state_loader=state_loader,
            shock_definitions=shock_definitions,
            solve_fn=solve_fn,
            compare_fn=compare_fn,
            key_indicators_fn=key_indicators_fn,
        )
