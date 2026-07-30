"""Symbolic equation system for CGE models with Pyomo support.

This module provides the equation ABC that concrete blocks implement to
build Pyomo constraint expressions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from equilibria.core.sets import SetManager


class SymbolicEquation(BaseModel, ABC):
    """Base class for symbolic equations that can be converted to Pyomo.

    Symbolic equations define mathematical relationships as Pyomo
    expressions built directly against a live ``pyomo_model``.

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
        pyomo_model: Any,
        indices: tuple[str, ...],
    ) -> Any:
        """Build a Pyomo expression for one constraint instance.

        Args:
            pyomo_model: The live Pyomo model (variables/parameters already
                attached as model components) to build the expression against
            indices: Index tuple for this constraint instance

        Returns:
            A Pyomo expression for this constraint instance, or ``None`` to
            signal that this particular index combination contributes no
            constraint (the bridge skips it rather than erroring, as long as
            at least one other index combination for the equation builds).
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
