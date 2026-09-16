"""Pyomo-compatible equation expressions for CGE models.

This module provides equation classes that build Pyomo expressions directly,
allowing proper symbolic constraint construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pyomo.environ import ConcreteModel


class PyomoEquation(BaseModel):
    """Equation that builds Pyomo expressions.

    Unlike the base Equation class that returns constraint functions,
    this class builds Pyomo constraint expressions directly.

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

    def build_pyomo_constraint(
        self,
        pyomo_model: ConcreteModel,
        indices: tuple[str, ...],
    ) -> Any:
        """Build a Pyomo constraint expression.

        Args:
            pyomo_model: The Pyomo model with variables and parameters
            indices: Index tuple for this constraint instance

        Returns:
            Pyomo expression (e.g., pyomo_model.var1 - pyomo_model.var2 == 0)
        """
        raise NotImplementedError("Subclasses must implement build_pyomo_constraint")

    def get_indices(self, set_manager) -> list[tuple[str, ...]]:
        """Generate all index combinations for this equation."""
        if not self.domains:
            return [()]

        # Get cartesian product of all domain sets
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
