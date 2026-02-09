"""Symbolic equation system for CGE models with Pyomo support.

This module provides equation classes that build symbolic expressions
compatible with Pyomo and other algebraic modeling languages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from equilibria.core.sets import SetManager
    from equilibria.core.parameters import Parameter
    from equilibria.core.variables import Variable


class SymbolicEquation(BaseModel, ABC):
    """Base class for symbolic equations that can be converted to Pyomo.

    Symbolic equations define mathematical relationships using symbolic
    expressions that can be evaluated with different backends (Pyomo, etc.)

    Attributes:
        name: Equation identifier
        domains: Tuple of set names defining equation indices
        description: Human-readable description
    """

    name: str = Field(..., description="Equation identifier")
    domains: tuple[str, ...] = Field(
        default_factory=tuple, description="Dimension set names"
    )
    description: str = Field(default="", description="Human-readable description")

    model_config = {"frozen": False}

    @abstractmethod
    def build_expression(
        self,
        set_manager: SetManager,
        variables: dict[str, Variable],
        parameters: dict[str, Parameter],
        indices: tuple[str, ...],
    ) -> Callable[..., float]:
        """Build a symbolic expression function.

        This method returns a function that, when called with variable/parameter
        accessors, returns the constraint residual.

        Args:
            set_manager: Set manager for index validation
            variables: Dictionary of all model variables
            parameters: Dictionary of all model parameters
            indices: Index tuple for this constraint instance

        Returns:
            Function that takes variable/parameter accessors and returns residual
        """
        pass

    def get_indices(self, set_manager: SetManager) -> list[tuple[str, ...]]:
        """Generate all index combinations for this equation."""
        if not self.domains:
            return [()]

        sets = [set_manager.get(d) for d in self.domains]

        def _product(sets_list):
            if not sets_list:
                return [()]
            first, *rest = sets_list
            result = []
            for elem in first:
                for combo in _product(rest):
                    result.append((elem,) + combo)
            return result

        return _product(sets)

    def count_equations(self, set_manager: SetManager) -> int:
        """Count total number of scalar equations."""
        if not self.domains:
            return 1

        total = 1
        for domain in self.domains:
            set_obj = set_manager.get(domain)
            total *= len(set_obj)
        return total


class ResidualEquation(SymbolicEquation):
    """Equation defined as a residual expression (lhs - rhs = 0)."""

    def __init__(
        self,
        name: str,
        domains: tuple[str, ...],
        lhs_expr: Callable,
        rhs_expr: Callable,
        description: str = "",
    ):
        """Initialize residual equation.

        Args:
            name: Equation name
            domains: Domain set names
            lhs_expr: Function returning left-hand side expression
            rhs_expr: Function returning right-hand side expression
            description: Human-readable description
        """
        super().__init__(
            name=name,
            domains=domains,
            description=description,
        )
        self.lhs_expr = lhs_expr
        self.rhs_expr = rhs_expr

    def build_expression(self, set_manager, variables, parameters, indices):
        """Build residual expression (lhs - rhs)."""
        lhs = self.lhs_expr(set_manager, variables, parameters, indices)
        rhs = self.rhs_expr(set_manager, variables, parameters, indices)

        def residual(get_var, get_param):
            return lhs(get_var, get_param) - rhs(get_var, get_param)

        return residual


# Helper functions for building expressions


def var(var_name: str, *indices: str) -> Callable:
    """Create a variable accessor function.

    Returns a function that, when given a variable accessor, retrieves
    the specified variable value.
    """

    def accessor(get_var, get_param):
        return get_var(var_name, *indices)

    return accessor


def param(param_name: str, *indices: str) -> Callable:
    """Create a parameter accessor function."""

    def accessor(get_var, get_param):
        return get_param(param_name, *indices)

    return accessor


def const(value: float) -> Callable:
    """Create a constant accessor function."""

    def accessor(get_var, get_param):
        return value

    return accessor


def add(*expressions: Callable) -> Callable:
    """Add multiple expressions."""

    def result(get_var, get_param):
        return sum(expr(get_var, get_param) for expr in expressions)

    return result


def multiply(*expressions: Callable) -> Callable:
    """Multiply multiple expressions."""

    def result(get_var, get_param):
        import functools
        import operator

        return functools.reduce(
            operator.mul, (expr(get_var, get_param) for expr in expressions), 1.0
        )

    return result


def power(base: Callable, exponent: Callable) -> Callable:
    """Raise base to power."""

    def result(get_var, get_param):
        return base(get_var, get_param) ** exponent(get_var, get_param)

    return result


def divide(numerator: Callable, denominator: Callable) -> Callable:
    """Divide two expressions."""

    def result(get_var, get_param):
        return numerator(get_var, get_param) / denominator(get_var, get_param)

    return result


def log(expr: Callable) -> Callable:
    """Natural logarithm."""
    import numpy as np

    def result(get_var, get_param):
        return np.log(expr(get_var, get_param))

    return result


def exp(expr: Callable) -> Callable:
    """Exponential."""
    import numpy as np

    def result(get_var, get_param):
        return np.exp(expr(get_var, get_param))

    return result


def sum_over(expr: Callable, set_name: str, set_manager) -> Callable:
    """Sum expression over a set.

    Creates an expression that sums over all elements of a set.
    """
    set_obj = set_manager.get(set_name)
    elements = list(set_obj)

    def result(get_var, get_param):
        return sum(expr(get_var, get_param, elem) for elem in elements)

    return result
