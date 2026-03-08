"""ICIO simulation adapter (state-based placeholder until native solver integration)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from equilibria.simulations.adapters.parameter_state import ParameterStateAdapter
from equilibria.simulations.types import ShockDefinition


class ICIOAdapter(ParameterStateAdapter):
    """ICIO adapter using generic state-scenario mechanics."""

    def __init__(
        self,
        *,
        base_state: dict[str, Any] | None = None,
        state_loader: Callable[[], dict[str, Any]] | None = None,
        shock_definitions: list[ShockDefinition] | None = None,
    ) -> None:
        super().__init__(
            model_label="icio",
            base_state=base_state,
            state_loader=state_loader,
            shock_definitions=shock_definitions,
        )
