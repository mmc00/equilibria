"""Pyomo backend for equilibria CGE framework.

This module provides a Pyomo-based solver backend that translates
equilibria models into Pyomo format and solves them using
IPOPT, CONOPT, or other Pyomo-compatible solvers.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from equilibria.backends.base import Backend, Solution

try:
    from pyomo.environ import (
        ConcreteModel,
        Param,
        Set,
        SolverFactory,
        Var,
        value,
    )

    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False

if TYPE_CHECKING:
    from equilibria.model import Model as EquilibriaModel


class PyomoBackend(Backend):
    """Pyomo-based solver backend.

    Translates equilibria models into Pyomo format and solves them
    using IPOPT, CONOPT, or other Pyomo-compatible solvers.

    Attributes:
        solver: Solver name (default: 'ipopt')
        pyomo_model: The Pyomo ConcreteModel instance

    Example:
        >>> backend = PyomoBackend(solver='ipopt')
        >>> backend.build(model)
        >>> solution = backend.solve()
        >>> print(solution.status)
    """

    def __init__(self, solver: str = "ipopt") -> None:
        """Initialize Pyomo backend.

        Args:
            solver: Solver name (default: 'ipopt')

        Raises:
            ImportError: If Pyomo is not installed
        """
        if not PYOMO_AVAILABLE:
            msg = "Pyomo is not installed. Install with: uv add pyomo"
            raise ImportError(msg)

        super().__init__(solver)
        self.pyomo_model: ConcreteModel | None = None
        self._solver_results: Any = None

    def build(self, model: EquilibriaModel) -> None:
        """Build Pyomo model from equilibria model.

        Args:
            model: equilibria Model instance
        """
        self._model = model
        self.pyomo_model = ConcreteModel(name=model.name)

        # Build sets
        self._build_sets(model)

        # Build parameters
        self._build_parameters(model)

        # Build variables
        self._build_variables(model)

        # Build constraints (placeholder - would need actual equations)
        # self._build_constraints(model)

        # Build objective (placeholder - CGE models often don't have objectives)
        # self._build_objective(model)

    def _build_sets(self, model: EquilibriaModel) -> None:
        """Build Pyomo sets from equilibria sets."""
        for set_name in model.set_manager.list_sets():
            set_obj = model.set_manager.get(set_name)
            elements = list(set_obj.elements)

            # Create Pyomo Set
            setattr(
                self.pyomo_model,
                set_name,
                Set(initialize=elements, doc=set_obj.description),
            )

    def _build_parameters(self, model: EquilibriaModel) -> None:
        """Build Pyomo parameters from equilibria parameters."""
        for param_name in model.parameter_manager.list_params():
            param = model.parameter_manager.get(param_name)

            if not param.domains:
                # Scalar parameter
                setattr(
                    self.pyomo_model,
                    param_name,
                    Param(initialize=float(param.value)),
                )
            else:
                # Indexed parameter
                # Get Pyomo sets for indexing
                index_sets = [getattr(self.pyomo_model, d) for d in param.domains]

                # Create dictionary of values
                values_dict = {}
                if len(param.domains) == 1:
                    set_obj = model.set_manager.get(param.domains[0])
                    for i, elem in enumerate(set_obj.iter_elements()):
                        values_dict[elem] = float(param.value[i])
                elif len(param.domains) == 2:
                    set1 = model.set_manager.get(param.domains[0])
                    set2 = model.set_manager.get(param.domains[1])
                    for i, e1 in enumerate(set1.iter_elements()):
                        for j, e2 in enumerate(set2.iter_elements()):
                            values_dict[(e1, e2)] = float(param.value[i, j])

                setattr(
                    self.pyomo_model,
                    param_name,
                    Param(*index_sets, initialize=values_dict),
                )

    def _build_variables(self, model: EquilibriaModel) -> None:
        """Build Pyomo variables from equilibria variables."""
        for var_name in model.variable_manager.list_vars():
            var = model.variable_manager.get(var_name)

            # Determine bounds
            lower = var.lower
            upper = var.upper

            if not var.domains:
                # Scalar variable
                setattr(
                    self.pyomo_model,
                    var_name,
                    Var(
                        bounds=(lower, upper),
                        initialize=float(var.value),
                    ),
                )
            else:
                # Indexed variable
                # Get Pyomo sets for indexing
                index_sets = [getattr(self.pyomo_model, d) for d in var.domains]

                # Create initialization dictionary
                init_dict = {}
                if len(var.domains) == 1:
                    set_obj = model.set_manager.get(var.domains[0])
                    for i, elem in enumerate(set_obj.iter_elements()):
                        init_dict[elem] = float(var.value[i])
                elif len(var.domains) == 2:
                    set1 = model.set_manager.get(var.domains[0])
                    set2 = model.set_manager.get(var.domains[1])
                    for i, e1 in enumerate(set1.iter_elements()):
                        for j, e2 in enumerate(set2.iter_elements()):
                            init_dict[(e1, e2)] = float(var.value[i, j])

                setattr(
                    self.pyomo_model,
                    var_name,
                    Var(
                        *index_sets,
                        bounds=(lower, upper),
                        initialize=init_dict,
                    ),
                )

    def solve(self, options: dict[str, Any] | None = None) -> Solution:
        """Solve the Pyomo model.

        Args:
            options: Solver options dictionary

        Returns:
            Solution object with results

        Raises:
            RuntimeError: If model not built or solver not available
        """
        if self.pyomo_model is None:
            msg = "Model not built. Call build() first."
            raise RuntimeError(msg)

        # Get solver
        solver = SolverFactory(self.solver)
        if not solver.available():
            msg = f"Solver '{self.solver}' is not available"
            raise RuntimeError(msg)

        # Set options
        if options:
            for key, val in options.items():
                solver.options[key] = val

        # Solve
        start_time = time.time()
        results = solver.solve(self.pyomo_model, tee=False)
        solve_time = time.time() - start_time

        # Store results
        self._solver_results = results

        # Extract solution
        status = str(results.solver.status)
        termination = str(results.solver.termination_condition)

        # Get variable values
        var_values = {}
        for var_name in self._model.variable_manager.list_vars():
            pyomo_var = getattr(self.pyomo_model, var_name)
            var = self._model.variable_manager.get(var_name)

            if not var.domains:
                # Scalar
                var_values[var_name] = np.array([value(pyomo_var)])
            else:
                # Indexed - extract values
                set_obj = self._model.set_manager.get(var.domains[0])
                values_list = []
                for elem in set_obj:
                    val = value(pyomo_var[elem])
                    values_list.append(val)
                var_values[var_name] = np.array(values_list)

        # Create solution object
        solution = Solution(
            model_name=self._model.name,
            status=f"{status} - {termination}",
            solve_time=solve_time,
            variables=var_values,
        )

        return solution

    def get_solver_status(self) -> dict[str, Any]:
        """Get detailed solver status.

        Returns:
            Dictionary with solver status details
        """
        if self._solver_results is None:
            return {"status": "not_solved"}

        results = self._solver_results

        return {
            "status": str(results.solver.status),
            "termination": str(results.solver.termination_condition),
            "message": str(results.solver.message),
            "time": results.solver.time,
            "iterations": results.solver.iterations,
        }

    def list_available_solvers(self) -> list[str]:
        """List available Pyomo solvers.

        Returns:
            List of available solver names
        """
        available = []
        test_solvers = ["ipopt", "gurobi", "cplex", "cbc", "glpk"]

        for solver_name in test_solvers:
            solver = SolverFactory(solver_name)
            if solver.available():
                available.append(solver_name)

        return available

    def __repr__(self) -> str:
        """String representation."""
        return f"PyomoBackend(solver={self.solver})"
