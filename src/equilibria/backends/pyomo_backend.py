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

        # Build constraints from equations
        self._build_constraints(model)

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
                # Check if it's actually a scalar or an array without domain info
                if param.value.ndim == 0 or param.value.size == 1:
                    # Scalar parameter
                    setattr(
                        self.pyomo_model,
                        param_name,
                        Param(initialize=float(param.value.flatten()[0])),
                    )
                else:
                    # Multi-dimensional parameter without domain info (e.g., FD0)
                    # Skip these for now as they're only used for initialization, not constraints
                    print(
                        f"Warning: Skipping parameter {param_name} - no domains defined"
                    )
                    continue
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
                # Scalar variable - extract scalar value from array if needed
                if hasattr(var.value, "__len__") and len(var.value) == 1:
                    init_val = float(var.value[0])
                else:
                    init_val = float(var.value)
                setattr(
                    self.pyomo_model,
                    var_name,
                    Var(
                        bounds=(lower, upper),
                        initialize=init_val,
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

    def _build_constraints(self, model: EquilibriaModel) -> None:
        """Build Pyomo constraints from equilibria equations.

        Args:
            model: equilibria Model instance
        """
        from pyomo.environ import Constraint
        from equilibria.backends.pyomo_equations import PyomoEquation

        for eq_name in model.equation_manager.list_equations():
            eq = model.equation_manager.get(eq_name)

            # Try to use build_expression method (new API)
            if hasattr(eq, "build_expression"):
                indices_list = eq.get_indices(model.set_manager)

                if not indices_list:
                    continue

                # Create constraint dictionary
                constraint_dict = {}
                for indices in indices_list:
                    try:
                        expr = eq.build_expression(self.pyomo_model, indices)
                        if expr is not None:
                            constraint_dict[indices] = expr
                    except Exception as e:
                        # Skip constraints that fail to build
                        print(
                            f"Warning: Could not build constraint {eq_name}{indices}: {e}"
                        )
                        continue

                if constraint_dict:
                    if eq.domains:
                        # Build index sets from constraint_dict keys
                        # Extract unique index values for each dimension
                        domain_sets = []
                        for dim_idx, domain in enumerate(eq.domains):
                            unique_vals = sorted(
                                set(idx[dim_idx] for idx in constraint_dict.keys())
                            )
                            domain_sets.append((domain, unique_vals))

                        # Create Pyomo sets for indexing if they don't exist
                        for domain, vals in domain_sets:
                            attr_name = f"_{eq_name}_{domain}_idx"
                            if not hasattr(self.pyomo_model, attr_name):
                                setattr(
                                    self.pyomo_model, attr_name, Set(initialize=vals)
                                )

                        index_sets = [
                            getattr(self.pyomo_model, f"_{eq_name}_{domain}_idx")
                            for domain, _ in domain_sets
                        ]

                        # Create a proper constraint rule that captures the dict
                        def make_constraint_rule(constraints):
                            def constraint_rule(m, *idx):
                                if idx in constraints:
                                    return constraints[idx]
                                return Constraint.Skip

                            return constraint_rule

                        setattr(
                            self.pyomo_model,
                            f"{eq_name}_con",
                            Constraint(
                                *index_sets,
                                rule=make_constraint_rule(constraint_dict),
                            ),
                        )
                    else:
                        # Scalar constraint
                        setattr(
                            self.pyomo_model,
                            f"{eq_name}_con",
                            Constraint(
                                rule=lambda m: list(constraint_dict.values())[0]
                            ),
                        )
            else:
                # Legacy equation handling - skip for now
                # These equations use closures and won't work with Pyomo
                pass

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
            elif len(var.domains) == 1:
                # 1D variable
                set_obj = self._model.set_manager.get(var.domains[0])
                values_list = []
                for elem in set_obj:
                    val = value(pyomo_var[elem])
                    values_list.append(val)
                var_values[var_name] = np.array(values_list)
            elif len(var.domains) == 2:
                # 2D variable (e.g., FD[F, J])
                set_obj_0 = self._model.set_manager.get(var.domains[0])
                set_obj_1 = self._model.set_manager.get(var.domains[1])
                values_matrix = []
                for elem0 in set_obj_0:
                    row = []
                    for elem1 in set_obj_1:
                        val = value(pyomo_var[elem0, elem1])
                        row.append(val)
                    values_matrix.append(row)
                var_values[var_name] = np.array(values_matrix)
            else:
                # 3D+ variables not yet supported
                raise NotImplementedError(
                    f"Solution extraction for {len(var.domains)}D variable '{var_name}' not yet implemented"
                )

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
