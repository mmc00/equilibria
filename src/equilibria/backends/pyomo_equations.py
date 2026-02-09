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


class CESAggregationEquation(PyomoEquation):
    """CES aggregation equation: VA = B * (sum beta * FD^rho)^(1/rho)"""

    def build_pyomo_constraint(self, pyomo_model, indices):
        """Build CES aggregation constraint for a sector."""
        j = indices[0]  # Sector index

        # Get Pyomo variables and parameters
        VA = getattr(pyomo_model, "VA")
        FD = getattr(pyomo_model, "FD")
        B_VA = getattr(pyomo_model, "B_VA")
        beta_VA = getattr(pyomo_model, "beta_VA")
        sigma_VA = getattr(pyomo_model, "sigma_VA")

        # For simplicity, use a linearized version or Cobb-Douglas
        # Full CES requires careful handling of the exponent
        # For now: VA[j] = B_VA[j] * prod(FD[f,j]^beta_VA[f,j])

        # Get the set of factors
        F_set = pyomo_model.F

        # Build product expression
        from pyomo.environ import log, exp

        # log(VA) = log(B_VA) + sum(beta * log(FD))
        # This is the Cobb-Douglas case (sigma = 1)
        log_expr = log(VA[j])
        rhs_expr = log(B_VA[j])

        for f in F_set:
            rhs_expr = rhs_expr + beta_VA[f, j] * log(FD[f, j])

        return log_expr == rhs_expr


class LeontiefEquation(PyomoEquation):
    """Leontief equation: XST = a_io * Z"""

    def build_pyomo_constraint(self, pyomo_model, indices):
        """Build Leontief constraint for commodity-sector pair."""
        i, j = indices

        XST = getattr(pyomo_model, "XST")
        Z = getattr(pyomo_model, "Z")
        a_io = getattr(pyomo_model, "a_io")

        return XST[i, j] == a_io[i, j] * Z[j]


class MarketClearingEquation(PyomoEquation):
    """Market clearing: supply = demand"""

    def build_pyomo_constraint(self, pyomo_model, indices):
        """Build market clearing constraint."""
        i = indices[0]

        # Get all relevant variables
        QA = getattr(pyomo_model, "QA", None)
        QD = getattr(pyomo_model, "QD", None)
        QM = getattr(pyomo_model, "QM", None)

        if QA is None:
            return None

        # Supply = QA[i]
        # Demand = QD[i] + QM[i] (simplified)
        supply = QA[i]
        demand = QD[i] if QD is not None else 0
        if QM is not None:
            demand = demand + QM[i]

        return supply == demand
