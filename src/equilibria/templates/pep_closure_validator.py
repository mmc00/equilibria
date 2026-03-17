"""Structural validation for PEP closure/equation/variable shape."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def _normalize_names(values: Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def _duplicates(values: tuple[str, ...]) -> tuple[str, ...]:
    counts = Counter(values)
    return tuple(sorted(name for name, count in counts.items() if count > 1))


class PEPClosureValidationReport(BaseModel):
    """Report for validating whether a closure leaves a well-formed system."""

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


def validate_pep_closure_structure(
    *,
    active_equations: Iterable[str],
    free_endogenous_variables: Iterable[str],
    fixed_by_closure: Iterable[str] | None = None,
    fixed_by_bounds_only: Iterable[str] | None = None,
    numeraire: str | None = None,
    unsupported_fixed_symbols: Iterable[str] | None = None,
    unsupported_endogenous_symbols: Iterable[str] | None = None,
) -> PEPClosureValidationReport:
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

    return PEPClosureValidationReport(
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
