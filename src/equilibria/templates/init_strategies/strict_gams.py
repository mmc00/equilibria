"""GAMS-level initialization strategy."""

from __future__ import annotations

from equilibria.templates.init_strategies.base import (
    InitializationStrategy,
    InitStrategySolverProtocol,
)
from equilibria.templates.pep_model_equations import PEPModelVariables


class StrictGAMSInitializationStrategy(InitializationStrategy):
    """Mirror GAMS benchmark levels with baseline compatibility checks."""

    mode = "gams"

    def apply(self, solver: InitStrategySolverProtocol, vars: PEPModelVariables) -> None:
        solver._ensure_strict_gams_baseline_compatibility()
        solver._overlay_with_gams_levels(vars)
        solver._sync_lambda_tr_from_levels(vars)
        solver._sync_policy_params_from_vars(vars)
        _set_walras_residual(solver, vars)


def _set_walras_residual(solver: InitStrategySolverProtocol, vars: PEPModelVariables) -> None:
    i_set = solver.sets.get("I", [])
    walras_i = "agr" if "agr" in i_set else (i_set[0] if i_set else None)
    if walras_i is None:
        return
    vars.LEON = (
        vars.Q.get(walras_i, 0.0)
        - sum(vars.C.get((walras_i, h), 0.0) for h in solver.sets.get("H", []))
        - vars.CG.get(walras_i, 0.0)
        - vars.INV.get(walras_i, 0.0)
        - vars.VSTK.get(walras_i, 0.0)
        - vars.DIT.get(walras_i, 0.0)
        - vars.MRGN.get(walras_i, 0.0)
    )
