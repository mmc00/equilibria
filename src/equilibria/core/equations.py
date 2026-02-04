"""Equation definitions for CGE models.

Equations represent mathematical relationships between variables
and parameters in the model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from equilibria.core.parameters import Parameter
    from equilibria.core.sets import SetManager
    from equilibria.core.variables import Variable


class Equation(BaseModel, ABC):
    """Base class for model equations.

    Equations define mathematical relationships that must hold in the
    model solution. They are declared over sets and can be indexed.

    Attributes:
        name: Equation identifier
        domains: Tuple of set names defining equation indices
        description: Human-readable description
        expression: Mathematical expression (as string or callable)

    Example:
        >>> class MarketClearing(Equation):
        ...     def define(self, set_manager, variables, parameters):
        ...         # Returns constraint expression
        ...         pass
    """

    name: str = Field(..., min_length=1, description="Equation identifier")
    domains: tuple[str, ...] = Field(
        default_factory=tuple, description="Dimension set names"
    )
    description: str = Field(default="", description="Human-readable description")

    model_config = {"frozen": False}

    @abstractmethod
    def define(
        self,
        set_manager: SetManager,
        variables: dict[str, Variable],
        parameters: dict[str, Parameter],
    ) -> dict[tuple[str, ...], Callable[[], float]]:
        """Define the equation for all index combinations.

        This method must be implemented by subclasses to define the
        mathematical relationship for each index combination.

        Args:
            set_manager: Set manager for index validation
            variables: Dictionary of all model variables
            parameters: Dictionary of all model parameters

        Returns:
            Dictionary mapping index tuples to constraint functions
        """
        ...

    def get_indices(self, set_manager: SetManager) -> list[tuple[str, ...]]:
        """Generate all index combinations for this equation.

        Args:
            set_manager: Set manager containing the domain sets

        Returns:
            List of index tuples
        """
        if not self.domains:
            return [()]

        # Get cartesian product of all domain sets
        sets = [set_manager.get(d) for d in self.domains]

        def _product(sets_list: list) -> list[tuple[str, ...]]:
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
        """Count total number of scalar equations.

        Args:
            set_manager: Set manager for set sizes

        Returns:
            Total number of indexed equations
        """
        if not self.domains:
            return 1

        total = 1
        for domain in self.domains:
            set_obj = set_manager.get(domain)
            total *= len(set_obj)
        return total

    def __repr__(self) -> str:
        """String representation."""
        domain_str = f"[{', '.join(self.domains)}]" if self.domains else "scalar"
        return f"Equation {self.name}{domain_str}"


class EquationManager:
    """Manages all equations in a model.

    Provides centralized access to equations and tracks equation counts.
    """

    def __init__(self, set_manager: SetManager | None = None) -> None:
        """Initialize equation manager.

        Args:
            set_manager: Optional set manager for validation
        """
        self._equations: dict[str, Equation] = {}
        self._set_manager = set_manager

    def add(self, eq: Equation) -> None:
        """Add an equation.

        Args:
            eq: Equation to add

        Raises:
            ValueError: If equation already exists
        """
        if eq.name in self._equations:
            msg = f"Equation '{eq.name}' already exists"
            raise ValueError(msg)

        # Validate domains exist in set manager
        if self._set_manager is not None:
            for domain in eq.domains:
                if domain not in self._set_manager:
                    msg = f"Domain set '{domain}' not found for equation '{eq.name}'"
                    raise ValueError(msg)

        self._equations[eq.name] = eq

    def get(self, name: str) -> Equation:
        """Get equation by name.

        Args:
            name: Equation name

        Returns:
            The Equation object

        Raises:
            KeyError: If equation not found
        """
        if name not in self._equations:
            msg = f"Equation '{name}' not found"
            raise KeyError(msg)
        return self._equations[name]

    def __getitem__(self, name: str) -> Equation:
        """Get equation by name using bracket notation."""
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        """Check if equation exists."""
        return name in self._equations

    def list_equations(self) -> list[str]:
        """Return list of all equation names."""
        return list(self._equations.keys())

    def get_total_count(self) -> int:
        """Return total number of scalar equations.

        Requires set_manager to be set for indexed equations.
        """
        if self._set_manager is None:
            return len(self._equations)

        total = 0
        for eq in self._equations.values():
            total += eq.count_equations(self._set_manager)
        return total

    def summary(self) -> dict[str, Any]:
        """Return summary of all equations."""
        total_scalar = (
            self.get_total_count() if self._set_manager else len(self._equations)
        )

        return {
            "total_equations": len(self._equations),
            "total_scalar_eqs": total_scalar,
            "equations": {
                name: {
                    "domains": eq.domains,
                    "description": eq.description,
                    "scalar_count": (
                        eq.count_equations(self._set_manager)
                        if self._set_manager
                        else 1
                    ),
                }
                for name, eq in self._equations.items()
            },
        }


class Constraint:
    """A single constraint instance (indexed equation).

    Represents one scalar constraint from an indexed equation.

    Attributes:
        equation: Parent equation
        indices: Index tuple for this instance
        expression: Constraint expression function
        lower: Lower bound (default: 0)
        upper: Upper bound (default: 0 for equality)
    """

    def __init__(
        self,
        equation: Equation,
        indices: tuple[str, ...],
        expression: Callable[[], float],
        lower: float = 0.0,
        upper: float = 0.0,
    ):
        """Initialize constraint.

        Args:
            equation: Parent equation
            indices: Index tuple
            expression: Function returning constraint residual
            lower: Lower bound
            upper: Upper bound
        """
        self.equation = equation
        self.indices = indices
        self.expression = expression
        self.lower = lower
        self.upper = upper

    def evaluate(self) -> float:
        """Evaluate the constraint expression."""
        return self.expression()

    def is_equality(self) -> bool:
        """Check if this is an equality constraint."""
        return self.lower == self.upper

    def __repr__(self) -> str:
        """String representation."""
        idx_str = f"[{', '.join(self.indices)}]" if self.indices else ""
        bound_str = "=" if self.is_equality() else f"[{self.lower}, {self.upper}]"
        return f"Constraint {self.equation.name}{idx_str} {bound_str}"
