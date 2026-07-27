"""Pyomo backend for equilibria CGE framework.

This module provides a Pyomo-based solver backend that translates
equilibria models into Pyomo format and solves them using
IPOPT, CONOPT, or other Pyomo-compatible solvers.
"""

from __future__ import annotations

import itertools
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from equilibria.backends.base import Backend, Solution

logger = logging.getLogger(__name__)

try:
    from pyomo.environ import (
        ConcreteModel,
        NonNegativeReals,
        Param,
        Reals,
        Set,
        SolverFactory,
        Var,
        value,
    )

    PYOMO_AVAILABLE = True

    # String → Pyomo domain object. The core Variable carries a plain string
    # (no pyomo import in core/), the bridge maps it here. GTAP uses only these
    # two — Reals (FREE vars xet/xw) and NonNegativeReals (78 price/qty vars).
    _DOMAINS = {"Reals": Reals, "NonNegativeReals": NonNegativeReals}
except ImportError:
    PYOMO_AVAILABLE = False
    _DOMAINS = {}

if TYPE_CHECKING:
    from equilibria.model import Model as EquilibriaModel


class BridgeTranslationError(RuntimeError):
    """Raised when the Pyomo bridge cannot faithfully translate a model.

    The bridge must fail loudly rather than silently dropping an equation
    that fails to build or stubbing a trivially-feasible constraint when no
    constraints survive — a dropped equation is invisible and fatal for
    parity with the GAMS reference.
    """


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
        self._pyomo_model: ConcreteModel | None = None
        self._solver_results: Any = None

    def build(self, model: EquilibriaModel) -> None:
        """Build Pyomo model from equilibria model.

        Args:
            model: equilibria Model instance
        """
        self._model = model
        self.pyomo_model = ConcreteModel(name=model.name)
        self._pyomo_model = self.pyomo_model

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
                    logger.warning(
                        "Skipping parameter %s - no domains defined", param_name
                    )
                    continue
            else:
                # Indexed parameter
                # Get Pyomo sets for indexing
                index_sets = [getattr(self.pyomo_model, d) for d in param.domains]

                # Create dictionary of values (arity-generic: any n >= 1).
                # itertools.product iterates the leftmost domain slowest and the
                # rightmost fastest, in lockstep with np.ndindex over the value
                # array's shape (first axis slowest, last axis fastest), so the
                # label tuple and the numpy index always refer to the same cell.
                # For a 1-D param the label tuple is (elem,); Pyomo indexes a
                # 1-D Param with the bare element, so unwrap the singleton.
                elems = [
                    list(model.set_manager.get(d).iter_elements())
                    for d in param.domains
                ]
                values_dict = {}
                for label_tuple, np_index in zip(
                    itertools.product(*elems),
                    np.ndindex(param.value.shape),
                    strict=True,
                ):
                    key = label_tuple[0] if len(label_tuple) == 1 else label_tuple
                    values_dict[key] = float(param.value[np_index])

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

            # Map the Variable's domain string to a Pyomo domain object. The
            # core Variable defaults to "Reals" so existing callers are
            # unchanged. An unrecognized name is made VISIBLE (logger.warning)
            # and falls back to Reals — a typo must not silently pass, but it
            # must not crash a solve either. Domain and bounds coexist: Pyomo
            # intersects within= with bounds=, matching the monolith which sets
            # both within and .setlb on its NonNegativeReals vars.
            within = _DOMAINS.get(var.domain)
            if within is None:
                logger.warning(
                    "Variable %s: unknown domain %r — falling back to Reals",
                    var_name,
                    var.domain,
                )
                within = Reals

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
                        within=within,
                        initialize=init_val,
                    ),
                )
            else:
                # Indexed variable
                # Get Pyomo sets for indexing
                index_sets = [getattr(self.pyomo_model, d) for d in var.domains]

                # Create initialization dictionary (arity-generic: any n >= 1).
                # itertools.product runs in lockstep with np.ndindex over the
                # value array's shape (leftmost domain / first axis slowest,
                # rightmost / last axis fastest), so the label tuple and the
                # numpy index name the same cell. A 1-D Var is indexed by the
                # bare element in Pyomo, so unwrap the singleton tuple.
                elems = [
                    list(model.set_manager.get(d).iter_elements()) for d in var.domains
                ]
                init_dict = {}
                for label_tuple, np_index in zip(
                    itertools.product(*elems),
                    np.ndindex(var.value.shape),
                    strict=True,
                ):
                    key = label_tuple[0] if len(label_tuple) == 1 else label_tuple
                    init_dict[key] = float(var.value[np_index])

                setattr(
                    self.pyomo_model,
                    var_name,
                    Var(
                        *index_sets,
                        bounds=(lower, upper),
                        within=within,
                        initialize=init_dict,
                    ),
                )

    def _build_constraints(self, model: EquilibriaModel) -> None:
        """Build Pyomo constraints from equilibria equations.

        Args:
            model: equilibria Model instance
        """
        from pyomo.environ import Constraint

        constraint_count = 0
        for eq_name in model.equation_manager.list_equations():
            eq = model.equation_manager.get(eq_name)

            # Try to use build_expression method (new API)
            if hasattr(eq, "build_expression"):
                indices_list = eq.get_indices(model.set_manager)

                if not indices_list:
                    # Legitimate in GTAP: an equation domained over an empty
                    # set contributes zero scalar constraints. Do not raise —
                    # but make the drop visible instead of silent.
                    logger.warning(
                        "Equation %s domained over an empty set — resolved to "
                        "zero index combinations; contributing no constraints",
                        eq_name,
                    )
                    continue

                # Create constraint dictionary
                constraint_dict = {}
                for indices in indices_list:
                    try:
                        expr = eq.build_expression(self.pyomo_model, indices)
                        if expr is not None:
                            constraint_dict[indices] = expr
                    except (ValueError, KeyError, AttributeError, TypeError) as e:
                        logger.warning(
                            "Could not build constraint %s%s: %s", eq_name, indices, e
                        )
                        raise BridgeTranslationError(
                            f"equation {eq_name}{indices} failed to build"
                        ) from e

                if constraint_dict:
                    if eq.domains:
                        # Build index sets from constraint_dict keys
                        # Extract unique index values for each dimension
                        domain_sets = []
                        for dim_idx, domain in enumerate(eq.domains):
                            unique_vals = sorted(
                                {idx[dim_idx] for idx in constraint_dict}
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
                        constraint_count += 1
                    else:
                        # Scalar constraint
                        setattr(
                            self.pyomo_model,
                            f"{eq_name}_con",
                            Constraint(
                                rule=lambda m: list(constraint_dict.values())[0]
                            ),
                        )
                        constraint_count += 1
                else:
                    # Had index combinations but every build_expression
                    # returned None — a true invisible drop of the whole
                    # equation. Fail loudly.
                    raise BridgeTranslationError(
                        f"equation {eq_name} produced no constraints despite "
                        f"having {len(indices_list)} index combination(s) — "
                        "all build_expression calls returned None"
                    )
            else:
                # Legacy closure-based equations cannot be translated to Pyomo.
                raise BridgeTranslationError(
                    f"equation {eq_name} has no build_expression "
                    "(legacy closure form not supported)"
                )
        if constraint_count == 0:
            raise BridgeTranslationError(
                "no constraints were built — model would be trivially feasible"
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
                # Indexed variable of any arity n >= 1. Allocate the array with
                # the shape implied by the domain set sizes, then fill it by
                # zipping itertools.product over the element labels with
                # np.ndindex over the array shape. Both iterate the leftmost
                # domain / first axis slowest and the rightmost / last axis
                # fastest, so each label tuple lands in the matching numpy cell
                # (a transposed fill would corrupt every N-D component). A 1-D
                # Var is indexed by the bare element, so unwrap the singleton.
                elems = [
                    list(self._model.set_manager.get(d).iter_elements())
                    for d in var.domains
                ]
                arr = np.empty(tuple(len(e) for e in elems))
                for label_tuple, np_index in zip(
                    itertools.product(*elems),
                    np.ndindex(arr.shape),
                    strict=True,
                ):
                    key = label_tuple[0] if len(label_tuple) == 1 else label_tuple
                    arr[np_index] = value(pyomo_var[key])
                var_values[var_name] = arr

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
