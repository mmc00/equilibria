"""Contract models for the simple open economy template."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, field_validator

from equilibria.contracts import (
    ModelBoundsConfig,
    ModelClosureConfig,
    ModelContract,
    ModelEquationConfig,
    deep_merge_model_dicts,
)


def _simple_open_equation_ids() -> tuple[str, ...]:
    return ("EQ_VA", "EQ_INT", "EQ_CET")


def _simple_open_closure_template_data(name: str) -> dict[str, Any]:
    closure_name = str(name).strip().lower()
    if not closure_name:
        raise ValueError("Closure name must be non-empty.")

    base = {
        "name": closure_name,
        "numeraire": "PFX",
        "numeraire_mode": "fixed_benchmark",
        "capital_mobility": "mobile",
        "fixed": ("PFX", "FSAV"),
        "endogenous": ("ER", "CAB"),
        "label": None,
    }

    if closure_name == "simple_open_default":
        base["label"] = "Default simple open economy closure"
        return base
    if closure_name == "flexible_external_balance":
        base["label"] = "Flexible external balance closure"
        base["fixed"] = ("PFX",)
        base["endogenous"] = ("ER", "CAB", "FSAV")
        return base

    raise ValueError(f"Unsupported simple open closure name: {name!r}")


class SimpleOpenClosureConfig(ModelClosureConfig):
    """Closure configuration for the simple open economy template."""

    name: str = "simple_open_default"
    numeraire: str = "PFX"
    numeraire_mode: Literal["fixed_benchmark"] = "fixed_benchmark"
    capital_mobility: Literal["mobile", "sector_specific"] = "mobile"
    fixed: tuple[str, ...] = Field(default_factory=lambda: ("PFX", "FSAV"))
    endogenous: tuple[str, ...] = Field(default_factory=lambda: ("ER", "CAB"))

    @field_validator("name", mode="before")
    @classmethod
    def _normalize_name(cls, value: Any) -> str:
        text = str(value).strip().lower()
        if not text:
            raise ValueError("Closure name must be non-empty.")
        return text


class SimpleOpenEquationConfig(ModelEquationConfig):
    """Equation activation policy for the simple open economy template."""

    name: str = "simple_open_core"
    include: tuple[str, ...] = Field(default_factory=_simple_open_equation_ids)
    activation_masks: Literal["template_defaults", "all_active"] = "template_defaults"

    @field_validator("activation_masks", mode="before")
    @classmethod
    def _normalize_activation_masks(cls, value: Any) -> str:
        text = str(value).strip().lower()
        if not text:
            raise ValueError("Equation activation_masks must be non-empty.")
        return text


class SimpleOpenBoundsConfig(ModelBoundsConfig):
    """Bounds policy for the simple open economy template."""

    name: str = "template_defaults"
    positive: Literal["lower_only"] = "lower_only"
    fixed_from_closure: bool = True
    free: tuple[str, ...] = Field(default_factory=lambda: ("ER", "CAB"))


class SimpleOpenContract(ModelContract):
    """Resolved contract for the simple open economy template."""

    name: str = "simple_open_v1"
    closure: SimpleOpenClosureConfig = Field(default_factory=SimpleOpenClosureConfig)
    equations: SimpleOpenEquationConfig = Field(default_factory=SimpleOpenEquationConfig)
    bounds: SimpleOpenBoundsConfig = Field(default_factory=SimpleOpenBoundsConfig)


def default_simple_open_contract() -> SimpleOpenContract:
    """Return the canonical simple open economy contract."""

    return SimpleOpenContract(
        closure=SimpleOpenClosureConfig.model_validate(
            _simple_open_closure_template_data("simple_open_default")
        )
    )


def build_simple_open_closure_config(
    value: str | Mapping[str, Any] | SimpleOpenClosureConfig | None = None,
) -> SimpleOpenClosureConfig:
    """Resolve one simple-open closure name, mapping override, or concrete config."""

    if value is None:
        return SimpleOpenClosureConfig.model_validate(
            _simple_open_closure_template_data("simple_open_default")
        )
    if isinstance(value, SimpleOpenClosureConfig):
        return value
    if isinstance(value, str):
        return SimpleOpenClosureConfig.model_validate(_simple_open_closure_template_data(value))
    if isinstance(value, Mapping):
        closure_name = value.get("name", value.get("preset", "simple_open_default"))
        base = _simple_open_closure_template_data(str(closure_name))
        merged = deep_merge_model_dicts(base, value)
        if "preset" in merged and "name" in merged:
            merged.pop("preset", None)
        return SimpleOpenClosureConfig.model_validate(merged)
    raise TypeError(
        "Simple open closure value must be None, a closure name string, a mapping, or SimpleOpenClosureConfig."
    )


def build_simple_open_contract(
    value: str | Mapping[str, Any] | SimpleOpenContract | None = None,
) -> SimpleOpenContract:
    """Resolve a contract name, mapping override, or concrete contract."""

    if value is None:
        return default_simple_open_contract()
    if isinstance(value, SimpleOpenContract):
        return value
    if isinstance(value, str):
        contract_name = value.strip()
        if contract_name == "simple_open_v1":
            return default_simple_open_contract()
        raise ValueError(f"Unsupported simple open contract name: {value!r}")
    if isinstance(value, Mapping):
        base = default_simple_open_contract().model_dump(mode="python")
        updates = dict(value)
        closure_value = updates.get("closure")
        if isinstance(closure_value, Mapping):
            updates["closure"] = build_simple_open_closure_config(closure_value).model_dump(
                mode="python"
            )
        elif isinstance(closure_value, (str, SimpleOpenClosureConfig)):
            updates["closure"] = build_simple_open_closure_config(closure_value).model_dump(
                mode="python"
            )
        merged = deep_merge_model_dicts(base, updates)
        return SimpleOpenContract.model_validate(merged)
    raise TypeError(
        "Simple open contract value must be None, a contract name string, a mapping, or SimpleOpenContract."
    )
