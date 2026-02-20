"""Initialization strategy registry for solver benchmark starts."""

from __future__ import annotations

from equilibria.templates.init_strategies.base import InitializationStrategy
from equilibria.templates.init_strategies.gams_flow import (
    GAMSFlowInitializationStrategy,
)
from equilibria.templates.init_strategies.strict_gams import (
    StrictGAMSInitializationStrategy,
)

CANONICAL_INIT_MODES: tuple[str, ...] = ("gams", "excel")

LEGACY_INIT_MODE_ALIASES: dict[str, str] = {
    "gams": "gams",
    "strict_gams": "gams",
    "excel": "excel",
    "gams_flow": "excel",
    "gams_levels": "excel",
    "equation_consistent": "excel",
    "gams_blockwise": "excel",
}

_CANONICAL_STRATEGIES: dict[str, type[InitializationStrategy]] = {
    StrictGAMSInitializationStrategy.mode: StrictGAMSInitializationStrategy,
    GAMSFlowInitializationStrategy.mode: GAMSFlowInitializationStrategy,
}


def normalize_init_mode(mode: str) -> str:
    """Normalize user input into one of the canonical init modes."""
    key = str(mode).strip().lower()
    normalized = LEGACY_INIT_MODE_ALIASES.get(key)
    if normalized is None:
        supported = ", ".join(CANONICAL_INIT_MODES)
        raise ValueError(f"Unknown init mode '{mode}'. Supported modes: {supported}")
    return normalized


def build_init_strategy(mode: str) -> InitializationStrategy:
    """Create an initialization strategy for the selected mode."""
    normalized = normalize_init_mode(mode)
    strategy_cls = _CANONICAL_STRATEGIES.get(normalized)
    if strategy_cls is None:
        supported = ", ".join(CANONICAL_INIT_MODES)
        raise ValueError(f"Unknown init mode '{mode}'. Supported modes: {supported}")
    return strategy_cls()


__all__ = [
    "InitializationStrategy",
    "StrictGAMSInitializationStrategy",
    "GAMSFlowInitializationStrategy",
    "CANONICAL_INIT_MODES",
    "LEGACY_INIT_MODE_ALIASES",
    "normalize_init_mode",
    "build_init_strategy",
]
