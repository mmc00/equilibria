"""Initialization strategies for solver benchmark state construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from equilibria.templates.pep_model_equations import PEPModelVariables


class InitStrategySolverProtocol(Protocol):
    """Minimal solver surface used by init strategies."""

    sets: dict[str, list[str]]

    def _ensure_strict_gams_baseline_compatibility(self) -> None: ...

    def _overlay_with_gams_levels(self, vars: PEPModelVariables) -> None: ...

    def _overlay_with_calibrated_levels(self, vars: PEPModelVariables) -> None: ...

    def _sync_lambda_tr_from_levels(self, vars: PEPModelVariables) -> None: ...

    def _sync_policy_params_from_vars(self, vars: PEPModelVariables) -> None: ...

    def _apply_equation_consistent_adjustments(self, vars: PEPModelVariables) -> None: ...

    def _reconcile_composite_prices(self, vars: PEPModelVariables) -> None: ...

    def _reconcile_tax_identities(self, vars: PEPModelVariables) -> None: ...

    def _recompute_gdp_aggregates(self, vars: PEPModelVariables) -> None: ...

    def _apply_trade_blockwise_flow(self, vars: PEPModelVariables) -> None: ...

    def _apply_trade_blockwise_transformation(self, vars: PEPModelVariables) -> None: ...

    def _apply_production_blockwise_accounting(self, vars: PEPModelVariables) -> None: ...

    def _apply_commodity_balance_blockwise(self, vars: PEPModelVariables) -> None: ...

    def _apply_trade_market_clearing_blockwise(self, vars: PEPModelVariables) -> None: ...

    def _attempt_coupled_trade_reconciliation(self, vars: PEPModelVariables) -> None: ...

    def _apply_macro_closure_blockwise(self, vars: PEPModelVariables) -> None: ...

    def _apply_gams_blockwise_presolve(self, vars: PEPModelVariables) -> None: ...


class InitializationStrategy(ABC):
    """Base class for all initialization strategies."""

    mode: str

    @abstractmethod
    def apply(self, solver: InitStrategySolverProtocol, vars: PEPModelVariables) -> None:
        """Mutate `vars` in-place using solver-specific initialization logic."""
