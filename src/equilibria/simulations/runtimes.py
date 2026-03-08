"""Runtime hook registry for mapping-based model adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

SolveHook = Callable[..., tuple[Any, Any, dict[str, Any]]]
CompareHook = Callable[..., dict[str, Any]]
IndicatorsHook = Callable[[Any], dict[str, float]]


class MappingRuntime(BaseModel):
    """Hook bundle used by mapping-based adapters."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    solve_fn: SolveHook | None = None
    compare_fn: CompareHook | None = None
    key_indicators_fn: IndicatorsHook | None = None

    def has_any_hook(self) -> bool:
        return (
            self.solve_fn is not None
            or self.compare_fn is not None
            or self.key_indicators_fn is not None
        )


_MAPPING_RUNTIME_REGISTRY: dict[str, MappingRuntime] = {}


def _normalize_model_key(model: str) -> str:
    key = str(model).strip().lower()
    if not key:
        raise ValueError("Model key must be non-empty.")
    return key


def register_mapping_runtime(
    model: str,
    *,
    solve_fn: SolveHook | None = None,
    compare_fn: CompareHook | None = None,
    key_indicators_fn: IndicatorsHook | None = None,
) -> None:
    """Register runtime hooks for one model key."""
    runtime = MappingRuntime(
        solve_fn=solve_fn,
        compare_fn=compare_fn,
        key_indicators_fn=key_indicators_fn,
    )
    if not runtime.has_any_hook():
        raise ValueError(
            "register_mapping_runtime requires at least one hook "
            "(solve_fn, compare_fn, or key_indicators_fn)."
        )
    _MAPPING_RUNTIME_REGISTRY[_normalize_model_key(model)] = runtime


def get_mapping_runtime(model: str) -> MappingRuntime | None:
    """Return registered runtime hooks for model, if any."""
    return _MAPPING_RUNTIME_REGISTRY.get(_normalize_model_key(model))


def clear_mapping_runtime(model: str | None = None) -> None:
    """Clear one model runtime or the full runtime registry."""
    if model is None:
        _MAPPING_RUNTIME_REGISTRY.clear()
        return
    _MAPPING_RUNTIME_REGISTRY.pop(_normalize_model_key(model), None)


def available_mapping_runtimes() -> tuple[str, ...]:
    """List model keys with registered runtime hooks."""
    return tuple(sorted(_MAPPING_RUNTIME_REGISTRY))

