"""Canonical contract models for the public PEP NLP problem."""

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


def _full_pep_equation_ids() -> tuple[str, ...]:
    return tuple(f"EQ{i}" for i in range(1, 99)) + ("WALRAS",)


def _closure_template_data(name: str) -> dict[str, Any]:
    closure_name = str(name).strip().lower()
    if not closure_name:
        raise ValueError("Closure name must be non-empty.")

    base = {
        "name": closure_name,
        "numeraire": "e",
        "numeraire_mode": "fixed_benchmark",
        "capital_mobility": "mobile",
        "fixed": (
            "G",
            "CAB",
            "KS",
            "LS",
            "PWM",
            "PWX",
            "CMIN",
            "VSTK",
            "TR_SELF",
        ),
        "endogenous": (
            "IT",
            "SH",
            "SF",
            "SG",
            "SROW",
        ),
        "label": None,
    }

    if closure_name == "pep_default":
        base["label"] = "Default public PEP closure"
        return base
    if closure_name == "trade_policy":
        base["label"] = "Trade policy closure"
        return base
    if closure_name == "world_price_shock":
        base["label"] = "World price shock closure"
        return base
    if closure_name == "government_spending":
        base["label"] = "Government spending closure"
        return base

    raise ValueError(f"Unsupported PEP closure name: {name!r}")


class PEPClosureConfig(ModelClosureConfig):
    """Economic closure choices that define what remains fixed/endogenous."""

    name: str = "pep_default"
    numeraire: str = "e"
    numeraire_mode: Literal["fixed_benchmark"] = "fixed_benchmark"
    capital_mobility: Literal["mobile", "sector_specific"] = "mobile"
    fixed: tuple[str, ...] = Field(
        default_factory=lambda: (
            "G",
            "CAB",
            "KS",
            "LS",
            "PWM",
            "PWX",
            "CMIN",
            "VSTK",
            "TR_SELF",
        )
    )
    endogenous: tuple[str, ...] = Field(
        default_factory=lambda: (
            "IT",
            "SH",
            "SF",
            "SG",
            "SROW",
        )
    )

    @field_validator("name", mode="before")
    @classmethod
    def _normalize_name(cls, value: Any) -> str:
        text = str(value).strip().lower()
        if not text:
            raise ValueError("Closure name must be non-empty.")
        return text


class PEPEquationConfig(ModelEquationConfig):
    """Activated equation system for the canonical PEP NLP."""

    name: str = "full_pep"
    include: tuple[str, ...] = Field(default_factory=_full_pep_equation_ids)
    activation_masks: Literal["gams_parity", "all_active"] = "gams_parity"

    @field_validator("activation_masks", mode="before")
    @classmethod
    def _normalize_activation_masks(cls, value: Any) -> str:
        text = str(value).strip().lower()
        if not text:
            raise ValueError("Equation activation_masks must be non-empty.")
        return text


class PEPBoundsConfig(ModelBoundsConfig):
    """Domain/bounds policy, kept separate from closure policy."""

    name: str = "economic"
    positive: Literal["lower_only"] = "lower_only"
    fixed_from_closure: bool = True
    free: tuple[str, ...] = Field(default_factory=lambda: ("SG", "SF", "SH", "SROW", "LEON"))


class PEPContract(ModelContract):
    """Resolved public contract for the canonical PEP NLP problem."""

    name: str = "pep_nlp_v1"
    closure: PEPClosureConfig = Field(default_factory=PEPClosureConfig)
    equations: PEPEquationConfig = Field(default_factory=PEPEquationConfig)
    bounds: PEPBoundsConfig = Field(default_factory=PEPBoundsConfig)


def default_pep_contract() -> PEPContract:
    """Return the canonical public PEP NLP contract."""

    return PEPContract(closure=PEPClosureConfig.model_validate(_closure_template_data("pep_default")))


def build_pep_closure_config(
    value: str | Mapping[str, Any] | PEPClosureConfig | None = None,
) -> PEPClosureConfig:
    """Resolve one PEP closure name, mapping override, or concrete config."""

    if value is None:
        return PEPClosureConfig.model_validate(_closure_template_data("pep_default"))
    if isinstance(value, PEPClosureConfig):
        return value
    if isinstance(value, str):
        return PEPClosureConfig.model_validate(_closure_template_data(value))
    if isinstance(value, Mapping):
        closure_name = value.get("name", value.get("preset", "pep_default"))
        base = _closure_template_data(str(closure_name))
        merged = deep_merge_model_dicts(base, value)
        if "preset" in merged and "name" in merged:
            merged.pop("preset", None)
        return PEPClosureConfig.model_validate(merged)
    raise TypeError(
        "PEP closure value must be None, a closure name string, a mapping, or PEPClosureConfig."
    )


def build_pep_contract(value: str | Mapping[str, Any] | PEPContract | None = None) -> PEPContract:
    """Resolve a contract name, mapping override, or concrete contract."""

    if value is None:
        return default_pep_contract()
    if isinstance(value, PEPContract):
        return value
    if isinstance(value, str):
        contract_name = value.strip()
        if contract_name == "pep_nlp_v1":
            return default_pep_contract()
        raise ValueError(f"Unsupported PEP contract name: {value!r}")
    if isinstance(value, Mapping):
        base = default_pep_contract().model_dump(mode="python")
        updates = dict(value)
        closure_value = updates.get("closure")
        if isinstance(closure_value, Mapping):
            updates["closure"] = build_pep_closure_config(closure_value).model_dump(mode="python")
        elif isinstance(closure_value, (str, PEPClosureConfig)):
            updates["closure"] = build_pep_closure_config(closure_value).model_dump(mode="python")
        merged = deep_merge_model_dicts(base, updates)
        return PEPContract.model_validate(merged)
    raise TypeError("PEP contract value must be None, a contract name string, a mapping, or PEPContract.")
