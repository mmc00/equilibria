"""IPOPT as a recipe solver: solve the model as it stands, return a status."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pyomo.environ as pyo


def ipopt(tol: float = 1e-10) -> Callable[[Any], str]:
    """A recipe solver for a Pyomo period model (``model.pyomo``): IPOPT on the
    square system the steps left free, with a zero objective.

    Returns ``"optimal"`` when IPOPT converged, else its termination condition.
    It never raises on non-convergence: the driver reports the code. On failure
    no solution is loaded: the model keeps the point the solve started from.
    """

    def solve(period_model: Any) -> str:
        model = period_model.pyomo
        if not list(model.component_data_objects(pyo.Objective, active=True)):
            model.add_component("_obj", pyo.Objective(expr=0.0))
        opt = pyo.SolverFactory("ipopt")
        opt.options["tol"] = tol
        opt.options["constr_viol_tol"] = tol
        res = opt.solve(model, load_solutions=False)
        status = str(res.solver.termination_condition)
        if status in ("optimal", "locallyOptimal"):
            model.solutions.load_from(res)
            return "optimal"
        return status

    return solve
