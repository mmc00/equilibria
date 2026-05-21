"""GTAP v6.2 Solver — thin wrapper over the v7 ``gtap.GTAPSolver``.

The solver engine (IPOPT, PATH, aggressive-fixing logic, walras-residual
reporting) is version-agnostic and lives in ``templates.gtap.gtap_solver``.
This wrapper layers v6.2-specific numeraire fixing and closure flag
handling on top.

Phase 2a status: this is a stub that delegates everything to the v7
solver. Phase 2c will add v6.2-specific overrides if needed (e.g. for
the ``pgdpwld`` numeraire vs v7's ``pnum``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from equilibria.templates.gtap.gtap_solver import GTAPSolver, SolverResult, SolverStatus

if TYPE_CHECKING:
    from pyomo.environ import ConcreteModel

    from equilibria.templates.gtap_v62.gtap_v62_contract import GTAPv62ClosureConfig
    from equilibria.templates.gtap_v62.gtap_v62_parameters import GTAPv62Parameters

logger = logging.getLogger(__name__)


class GTAPv62Solver(GTAPSolver):
    """Solver for GTAP v6.2 models.

    Inherits the full v7 solver machinery (PATH C-API, IPOPT,
    aggressive-fixing, residual reporting) and replaces the numeraire-
    fixing step so it points at ``pgdpwld`` (v6.2 default) instead of
    ``pnum`` (v7 default).

    Example::

        solver = GTAPv62Solver(model, closure, solver_name="path-capi")
        result = solver.solve()
        assert result.success
        assert abs(result.walras_value) < 1e-6
    """

    def __init__(
        self,
        model: "ConcreteModel",
        closure: Optional["GTAPv62ClosureConfig"] = None,
        solver_name: str = "ipopt",
        solver_options: Optional[Dict[str, Any]] = None,
        params: Optional["GTAPv62Parameters"] = None,
    ):
        super().__init__(
            model=model,
            closure=closure,
            solver_name=solver_name,
            solver_options=solver_options,
            params=params,
        )

    def fix_numeraire(self) -> None:
        """Fix the numeraire variable for v6.2.

        v6.2 traditionally uses ``pgdpwld`` (world GDP price index) as
        numeraire. The variable is fixed at 1.0 (its calibration value)
        so the price system has a well-defined scale.
        """
        numeraire_name = getattr(self.closure, "numeraire", None) or "pgdpwld"
        if not hasattr(self.model, numeraire_name):
            logger.warning(
                "Numeraire variable %r not found on model; skipping fix",
                numeraire_name,
            )
            return

        var = getattr(self.model, numeraire_name)
        try:
            if var.is_indexed():
                for k in var:
                    var[k].fix(1.0)
            else:
                var.fix(1.0)
        except Exception as exc:
            logger.warning("Could not fix numeraire %s: %s", numeraire_name, exc)


__all__ = [
    "GTAPv62Solver",
    "SolverResult",
    "SolverStatus",
]
