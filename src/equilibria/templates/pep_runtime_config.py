"""Runtime configuration models for executing and validating the PEP NLP."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field

from equilibria.contracts import (
    ModelReferenceConfig,
    ModelRuntimeConfig,
    deep_merge_model_dicts,
)


class PEPReferenceConfig(ModelReferenceConfig):
    """Optional parity/reference settings kept outside the economic contract."""

    source: Literal["gams", "none"] = "none"
    model_type: Literal["nlp", "cns"] | None = None


class PEPRuntimeConfig(ModelRuntimeConfig):
    """Execution configuration for the canonical PEP NLP."""

    name: str = "default_ipopt"
    problem_type: Literal["nlp", "cns"] = "nlp"
    solver: str = "ipopt"
    jacobian_mode: Literal["analytic", "numeric"] = "analytic"
    tolerance: float = 1e-8
    max_iterations: int = 300
    require_solver_success: bool = True
    accept_square_feasible: bool = True
    reference: PEPReferenceConfig = Field(default_factory=PEPReferenceConfig)


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
    value: str | Mapping[str, object] | PEPRuntimeConfig | None = None,
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
        merged = deep_merge_model_dicts(base, value)
        return PEPRuntimeConfig.model_validate(merged)
    raise TypeError(
        "PEP runtime config value must be None, a preset string, a mapping, or PEPRuntimeConfig."
    )
