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
