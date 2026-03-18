"""Structural validation for PEP closure/equation/variable shape."""

from __future__ import annotations

from collections.abc import Iterable

from equilibria.contracts import ModelClosureValidationReport, validate_closure_structure


class PEPClosureValidationReport(ModelClosureValidationReport):
    """Report for validating whether a closure leaves a well-formed PEP system."""


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
    """Validate the structural shape implied by the current PEP closure."""

    report = validate_closure_structure(
        active_equations=active_equations,
        free_endogenous_variables=free_endogenous_variables,
        fixed_by_closure=fixed_by_closure,
        fixed_by_bounds_only=fixed_by_bounds_only,
        numeraire=numeraire,
        unsupported_fixed_symbols=unsupported_fixed_symbols,
        unsupported_endogenous_symbols=unsupported_endogenous_symbols,
    )
    return PEPClosureValidationReport.model_validate(report.model_dump(mode="python"))
