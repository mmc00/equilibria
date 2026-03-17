"""Canonical contract models for the public PEP NLP problem."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _normalize_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [value]
    else:
        items = list(value)
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return tuple(normalized)


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


class PEPClosureConfig(BaseModel):
    """Economic closure choices that define what remains fixed/endogenous."""

    model_config = ConfigDict(frozen=True, extra="forbid")

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
    label: str | None = None

    @field_validator("numeraire", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Closure text fields must be non-empty.")
        return text

    @field_validator("name", mode="before")
    @classmethod
    def _normalize_name(cls, value: Any) -> str:
        text = str(value).strip().lower()
        if not text:
            raise ValueError("Closure name must be non-empty.")
        return text

    @field_validator("label", mode="before")
    @classmethod
    def _normalize_optional_label(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("fixed", "endogenous", mode="before")
    @classmethod
    def _normalize_symbol_groups(cls, value: Any) -> tuple[str, ...]:
        return _normalize_string_tuple(value)

    @model_validator(mode="after")
    def _check_overlap(self) -> "PEPClosureConfig":
        overlap = set(self.fixed) & set(self.endogenous)
        if overlap:
            overlap_text = ", ".join(sorted(overlap))
            raise ValueError(f"Closure fixed/endogenous overlap is not allowed: {overlap_text}")
        return self


class PEPEquationConfig(BaseModel):
    """Activated equation system for the canonical PEP NLP."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "full_pep"
    include: tuple[str, ...] = Field(default_factory=_full_pep_equation_ids)
    activation_masks: Literal["gams_parity", "all_active"] = "gams_parity"

    @field_validator("name", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Equation text fields must be non-empty.")
        return text

    @field_validator("activation_masks", mode="before")
    @classmethod
    def _normalize_activation_masks(cls, value: Any) -> str:
        text = str(value).strip().lower()
        if not text:
            raise ValueError("Equation activation_masks must be non-empty.")
        return text

    @field_validator("include", mode="before")
    @classmethod
    def _normalize_include(cls, value: Any) -> tuple[str, ...]:
        normalized = _normalize_string_tuple(value)
        if not normalized:
            raise ValueError("Equation include list must be non-empty.")
        return normalized


class PEPBoundsConfig(BaseModel):
    """Domain/bounds policy, kept separate from closure policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "economic"
    positive: Literal["lower_only"] = "lower_only"
    fixed_from_closure: bool = True
    free: tuple[str, ...] = Field(default_factory=lambda: ("SG", "SF", "SH", "SROW", "LEON"))

    @field_validator("name", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Bounds name must be non-empty.")
        return text

    @field_validator("free", mode="before")
    @classmethod
    def _normalize_free(cls, value: Any) -> tuple[str, ...]:
        return _normalize_string_tuple(value)


class PEPContract(BaseModel):
    """Resolved public contract for the canonical PEP NLP problem."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "pep_nlp_v1"
    closure: PEPClosureConfig = Field(default_factory=PEPClosureConfig)
    equations: PEPEquationConfig = Field(default_factory=PEPEquationConfig)
    bounds: PEPBoundsConfig = Field(default_factory=PEPBoundsConfig)

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Contract name must be non-empty.")
        return text


def _deep_merge(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


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
        merged = _deep_merge(base, value)
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
        merged = _deep_merge(base, updates)
        return PEPContract.model_validate(merged)
    raise TypeError("PEP contract value must be None, a contract name string, a mapping, or PEPContract.")
