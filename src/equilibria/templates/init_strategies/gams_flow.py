"""Excel/SAM initialization strategy."""

from __future__ import annotations

from equilibria.templates.init_strategies.base import (
    InitializationStrategy,
    InitStrategySolverProtocol,
)
from equilibria.templates.init_strategies.strict_gams import _set_walras_residual
from equilibria.templates.pep_model_equations import PEPModelVariables


class GAMSFlowInitializationStrategy(InitializationStrategy):
    """
    Mirror the canonical GAMS SAM-calibration flow:
    calibrate from SAM -> set variable.l = variable0.
    """

    mode = "excel"

    def apply(self, solver: InitStrategySolverProtocol, vars: PEPModelVariables) -> None:
        solver._overlay_with_calibrated_levels(vars)
        solver._sync_lambda_tr_from_levels(vars)
        solver._sync_policy_params_from_vars(vars)
        _set_walras_residual(solver, vars)


class GAMSBlockwiseInitializationStrategy(InitializationStrategy):
    """
    Start from calibrated benchmark levels, then run blockwise pre-solve passes.

    This is the closest Python analogue to the GAMS workflow:
    1. initialize `.l` from calibrated `*O` values
    2. apply shocked/fixed closures
    3. let the solver start from that benchmark-consistent point
    """

    mode = "gams_blockwise"

    def apply(self, solver: InitStrategySolverProtocol, vars: PEPModelVariables) -> None:
        solver._overlay_with_calibrated_levels(vars)
        solver._sync_lambda_tr_from_levels(vars)
        solver._sync_policy_params_from_vars(vars)
        _set_walras_residual(solver, vars)
        solver._apply_gams_blockwise_presolve(vars)
        solver._sync_lambda_tr_from_levels(vars)
        solver._sync_policy_params_from_vars(vars)
        _set_walras_residual(solver, vars)
