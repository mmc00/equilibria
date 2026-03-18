"""Runtime configuration models for the simple open economy template."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field

from equilibria.contracts import (
    ModelReferenceConfig,
    ModelRuntimeConfig,
    deep_merge_model_dicts,
)


class SimpleOpenReferenceConfig(ModelReferenceConfig):
    """Reference settings for the simple open economy template."""

    source: Literal["none"] = "none"
    model_type: Literal["template"] | None = None


class SimpleOpenRuntimeConfig(ModelRuntimeConfig):
    """Execution configuration for the simple open economy template."""

    name: str = "default_template"
    problem_type: Literal["template"] = "template"
    solver: Literal["none"] = "none"
    tolerance: float = 1e-8
    max_iterations: int = 1
    require_solver_success: bool = False
    accept_square_feasible: bool = False
    reference: SimpleOpenReferenceConfig = Field(default_factory=SimpleOpenReferenceConfig)


def default_simple_open_runtime_config() -> SimpleOpenRuntimeConfig:
    """Default runtime config for the simple open economy template."""

    return SimpleOpenRuntimeConfig()


def build_simple_open_runtime_config(
    value: str | Mapping[str, object] | SimpleOpenRuntimeConfig | None = None,
) -> SimpleOpenRuntimeConfig:
    """Resolve a runtime config preset, mapping override, or concrete config."""

    if value is None:
        return default_simple_open_runtime_config()
    if isinstance(value, SimpleOpenRuntimeConfig):
        return value
    if isinstance(value, str):
        preset = value.strip()
        if preset == "default_template":
            return default_simple_open_runtime_config()
        raise ValueError(f"Unsupported simple open runtime config preset: {value!r}")
    if isinstance(value, Mapping):
        base = default_simple_open_runtime_config().model_dump(mode="python")
        merged = deep_merge_model_dicts(base, value)
        return SimpleOpenRuntimeConfig.model_validate(merged)
    raise TypeError(
        "Simple open runtime config value must be None, a preset string, a mapping, or SimpleOpenRuntimeConfig."
    )
