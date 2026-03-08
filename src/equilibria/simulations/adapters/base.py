"""Base adapter interface for model-agnostic scenario simulation."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from equilibria.simulations.types import Shock, ShockDefinition


class BaseModelAdapter(ABC):
    """Adapter contract used by :class:`~equilibria.simulations.Simulator`."""

    @abstractmethod
    def fit_base_state(self) -> Any:
        """Calibrate/build the base state for the selected model."""

    def clone_state(self, state: Any) -> Any:
        """Deep-copy a calibrated state before applying scenario shocks."""
        return copy.deepcopy(state)

    @abstractmethod
    def available_shocks(self) -> list[ShockDefinition]:
        """Return model-specific shock catalog."""

    @abstractmethod
    def apply_shock(self, state: Any, shock: Shock) -> None:
        """Apply one shock to a mutable model state."""

    @abstractmethod
    def solve_state(
        self,
        state: Any,
        *,
        initial_vars: Any | None,
        reference_results_gdx: Path | None,
        reference_slice: str,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Solve one state and return ``(solver, solution, validation)``."""

    @abstractmethod
    def compare_with_reference(
        self,
        *,
        solution_vars: Any,
        solution_params: dict[str, Any],
        reference_results_gdx: Path,
        reference_slice: str,
        abs_tol: float,
        rel_tol: float,
    ) -> dict[str, Any]:
        """Compare one solution against external reference results."""

    @abstractmethod
    def key_indicators(self, vars_obj: Any) -> dict[str, float]:
        """Return standardized high-level indicators for report summaries."""
