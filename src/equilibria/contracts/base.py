"""Generic contract/runtime primitives shared across model templates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def normalize_string_tuple(value: Any) -> tuple[str, ...]:
    """Normalize string-like inputs into an ordered unique tuple."""

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


def deep_merge_model_dicts(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge nested dictionaries used to build contract/config models."""

    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_model_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


class ModelClosureConfig(BaseModel):
    """Generic closure definition shared across model contracts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "default"
    numeraire: str = "e"
    numeraire_mode: str = "fixed_benchmark"
    capital_mobility: str = "mobile"
    fixed: tuple[str, ...] = Field(default_factory=tuple)
    endogenous: tuple[str, ...] = Field(default_factory=tuple)
    label: str | None = None

    @field_validator("name", "numeraire", "numeraire_mode", "capital_mobility", mode="before")
    @classmethod
    def _normalize_required_text(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Model closure text fields must be non-empty.")
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
        return normalize_string_tuple(value)

    @model_validator(mode="after")
    def _check_overlap(self) -> "ModelClosureConfig":
        overlap = set(self.fixed) & set(self.endogenous)
        if overlap:
            overlap_text = ", ".join(sorted(overlap))
            raise ValueError(f"Closure fixed/endogenous overlap is not allowed: {overlap_text}")
        return self


class ModelEquationConfig(BaseModel):
    """Generic equation activation policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "full_model"
    include: tuple[str, ...] = Field(default_factory=tuple)
    activation_masks: str = "default"

    @field_validator("name", "activation_masks", mode="before")
    @classmethod
    def _normalize_text(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Model equation text fields must be non-empty.")
        return text

    @field_validator("include", mode="before")
    @classmethod
    def _normalize_include(cls, value: Any) -> tuple[str, ...]:
        normalized = normalize_string_tuple(value)
        if not normalized:
            raise ValueError("Equation include list must be non-empty.")
        return normalized


class ModelBoundsConfig(BaseModel):
    """Generic bounds/domain policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "economic"
    positive: str = "lower_only"
    fixed_from_closure: bool = True
    free: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("name", "positive", mode="before")
    @classmethod
    def _normalize_text(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Model bounds text fields must be non-empty.")
        return text

    @field_validator("free", mode="before")
    @classmethod
    def _normalize_free(cls, value: Any) -> tuple[str, ...]:
        return normalize_string_tuple(value)


class ModelContract(BaseModel):
    """Generic model contract: closure + equations + bounds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    closure: ModelClosureConfig
    equations: ModelEquationConfig
    bounds: ModelBoundsConfig

    @field_validator("name", mode="before")
    @classmethod
    def _normalize_name(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Contract name must be non-empty.")
        return text


class ModelReferenceConfig(BaseModel):
    """Optional parity/reference settings kept outside the economic contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    source: str = "none"
    model_type: str | None = None
    solver: str | None = None
    slice: str | None = None
    levels_tol: float = 1e-8
    params_tol: float = 1e-8

    @field_validator("source", mode="before")
    @classmethod
    def _normalize_source(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Reference source must be non-empty.")
        return text

    @field_validator("model_type", "solver", "slice", mode="before")
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
    def _check_enabled_fields(self) -> "ModelReferenceConfig":
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


class ModelRuntimeConfig(BaseModel):
    """Generic runtime configuration for executing a model contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "default"
    problem_type: str = "nlp"
    solver: str = "ipopt"
    tolerance: float = 1e-8
    max_iterations: int = 300
    require_solver_success: bool = True
    accept_square_feasible: bool = True
    reference: ModelReferenceConfig = Field(default_factory=ModelReferenceConfig)

    @field_validator("name", "problem_type", "solver", mode="before")
    @classmethod
    def _normalize_text(cls, value: Any) -> str:
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


def _normalize_names(values: Iterable[str] | None) -> tuple[str, ...]:
    return normalize_string_tuple(values)


def _duplicates(values: tuple[str, ...]) -> tuple[str, ...]:
    counts = Counter(values)
    return tuple(sorted(name for name, count in counts.items() if count > 1))


class ModelClosureValidationReport(BaseModel):
    """Generic structural validation report for closure/system shape."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_valid: bool
    system_shape: Literal["square", "overdetermined", "underdetermined"]
    active_equation_count: int
    free_endogenous_variable_count: int
    fixed_by_closure_count: int
    fixed_by_bounds_only_count: int
    equation_variable_gap: int
    numeraire: str | None = None
    numeraire_is_fixed: bool | None = None
    duplicate_equations: tuple[str, ...] = Field(default_factory=tuple)
    duplicate_free_variables: tuple[str, ...] = Field(default_factory=tuple)
    unsupported_fixed_symbols: tuple[str, ...] = Field(default_factory=tuple)
    unsupported_endogenous_symbols: tuple[str, ...] = Field(default_factory=tuple)
    messages: tuple[str, ...] = Field(default_factory=tuple)


def validate_closure_structure(
    *,
    active_equations: Iterable[str],
    free_endogenous_variables: Iterable[str],
    fixed_by_closure: Iterable[str] | None = None,
    fixed_by_bounds_only: Iterable[str] | None = None,
    numeraire: str | None = None,
    unsupported_fixed_symbols: Iterable[str] | None = None,
    unsupported_endogenous_symbols: Iterable[str] | None = None,
) -> ModelClosureValidationReport:
    """Validate the structural shape implied by the current closure."""

    active_eq = _normalize_names(active_equations)
    free_vars = _normalize_names(free_endogenous_variables)
    fixed_closure = _normalize_names(fixed_by_closure)
    fixed_bounds = _normalize_names(fixed_by_bounds_only)
    unsupported_fixed = _normalize_names(unsupported_fixed_symbols)
    unsupported_endogenous = _normalize_names(unsupported_endogenous_symbols)

    duplicate_equations = _duplicates(active_eq)
    duplicate_free_variables = _duplicates(free_vars)

    active_eq_unique = tuple(dict.fromkeys(active_eq))
    free_vars_unique = tuple(dict.fromkeys(free_vars))
    fixed_closure_unique = tuple(dict.fromkeys(fixed_closure))
    fixed_bounds_unique = tuple(dict.fromkeys(fixed_bounds))

    active_equation_count = len(active_eq_unique)
    free_endogenous_variable_count = len(free_vars_unique)
    equation_variable_gap = active_equation_count - free_endogenous_variable_count

    if equation_variable_gap == 0:
        system_shape: Literal["square", "overdetermined", "underdetermined"] = "square"
    elif equation_variable_gap > 0:
        system_shape = "overdetermined"
    else:
        system_shape = "underdetermined"

    fixed_names = set(fixed_closure_unique) | set(fixed_bounds_unique)
    numeraire_is_fixed = None
    if numeraire is not None:
        numeraire_text = str(numeraire).strip() or None
        numeraire = numeraire_text
        numeraire_is_fixed = numeraire_text in fixed_names if numeraire_text is not None else None

    messages: list[str] = []
    if duplicate_equations:
        messages.append(f"Duplicate active equations: {', '.join(duplicate_equations)}")
    if duplicate_free_variables:
        messages.append(f"Duplicate free endogenous variables: {', '.join(duplicate_free_variables)}")
    if unsupported_fixed:
        messages.append(
            "Unsupported fixed closure symbols: "
            + ", ".join(sorted(dict.fromkeys(unsupported_fixed)))
        )
    if unsupported_endogenous:
        messages.append(
            "Unsupported endogenous closure symbols: "
            + ", ".join(sorted(dict.fromkeys(unsupported_endogenous)))
        )
    if system_shape != "square":
        messages.append(
            "Closure leaves a non-square system "
            f"(active_equations={active_equation_count}, free_variables={free_endogenous_variable_count})."
        )
    if numeraire is not None and not numeraire_is_fixed:
        messages.append(f"Numeraire {numeraire!r} is not fixed by the current closure/bounds.")

    is_valid = (
        system_shape == "square"
        and not duplicate_equations
        and not duplicate_free_variables
        and not unsupported_fixed
        and not unsupported_endogenous
        and (numeraire is None or bool(numeraire_is_fixed))
    )

    return ModelClosureValidationReport(
        is_valid=is_valid,
        system_shape=system_shape,
        active_equation_count=active_equation_count,
        free_endogenous_variable_count=free_endogenous_variable_count,
        fixed_by_closure_count=len(fixed_closure_unique),
        fixed_by_bounds_only_count=len(fixed_bounds_unique),
        equation_variable_gap=equation_variable_gap,
        numeraire=numeraire,
        numeraire_is_fixed=numeraire_is_fixed,
        duplicate_equations=duplicate_equations,
        duplicate_free_variables=duplicate_free_variables,
        unsupported_fixed_symbols=tuple(sorted(dict.fromkeys(unsupported_fixed))),
        unsupported_endogenous_symbols=tuple(sorted(dict.fromkeys(unsupported_endogenous))),
        messages=tuple(messages),
    )
