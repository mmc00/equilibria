"""Runtime configuration models for executing and validating the PEP NLP."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PEPReferenceConfig(BaseModel):
    """Optional parity/reference settings kept outside the economic contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    source: Literal["gams", "none"] = "none"
    model_type: Literal["nlp", "cns"] | None = None
    solver: str | None = None
    slice: str | None = None
    levels_tol: float = 1e-8
    params_tol: float = 1e-8

    @field_validator("solver", "slice", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("levels_tol", "params_tol")
    @classmethod
    def _positive_tol(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Reference tolerances must be positive.")
        return value

    @model_validator(mode="after")
    def _check_enabled_fields(self) -> "PEPReferenceConfig":
        if not self.enabled:
            return self
        missing: list[str] = []
        if self.source == "none":
            missing.append("source")
        if self.model_type is None:
            missing.append("model_type")
        if self.solver is None:
            missing.append("solver")
        if self.slice is None:
            missing.append("slice")
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"Enabled reference requires: {missing_text}")
        return self


class PEPRuntimeConfig(BaseModel):
    """Execution configuration for the canonical PEP NLP."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "default_ipopt"
    problem_type: Literal["nlp", "cns"] = "nlp"
    solver: str = "ipopt"
    tolerance: float = 1e-8
    max_iterations: int = 300
    require_solver_success: bool = True
    accept_square_feasible: bool = True
    reference: PEPReferenceConfig = Field(default_factory=PEPReferenceConfig)

    @field_validator("name", "solver", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Runtime config text fields must be non-empty.")
        return text

    @field_validator("tolerance")
    @classmethod
    def _positive_tolerance(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Solver tolerance must be positive.")
        return value

    @field_validator("max_iterations")
    @classmethod
    def _positive_iterations(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Max iterations must be positive.")
        return value


def _deep_merge(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def default_pep_runtime_config() -> PEPRuntimeConfig:
    """Default runtime config for standard IPOPT execution."""

    return PEPRuntimeConfig()


def default_pep_parity_runtime_config() -> PEPRuntimeConfig:
    """Runtime config for IPOPT execution with GAMS NLP parity enabled."""

    return PEPRuntimeConfig(
        name="parity_ipopt_gams_nlp",
        reference=PEPReferenceConfig(
            enabled=True,
            source="gams",
            model_type="nlp",
            solver="ipopt",
            slice="sim1",
            levels_tol=1e-8,
            params_tol=1e-8,
        ),
    )


def build_pep_runtime_config(
    value: str | Mapping[str, Any] | PEPRuntimeConfig | None = None,
) -> PEPRuntimeConfig:
    """Resolve a runtime config preset, mapping override, or concrete config."""

    if value is None:
        return default_pep_runtime_config()
    if isinstance(value, PEPRuntimeConfig):
        return value
    if isinstance(value, str):
        preset = value.strip()
        if preset == "default_ipopt":
            return default_pep_runtime_config()
        if preset == "parity_ipopt_gams_nlp":
            return default_pep_parity_runtime_config()
        raise ValueError(f"Unsupported PEP runtime config preset: {value!r}")
    if isinstance(value, Mapping):
        base = default_pep_runtime_config().model_dump(mode="python")
        merged = _deep_merge(base, value)
        return PEPRuntimeConfig.model_validate(merged)
    raise TypeError(
        "PEP runtime config value must be None, a preset string, a mapping, or PEPRuntimeConfig."
    )
