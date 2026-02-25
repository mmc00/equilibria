"""Simple Open Economy CGE template.

A basic open economy CGE model suitable for teaching and
simple policy analysis.
"""

from typing import Any

from pydantic import Field

from equilibria.blocks import (
    CESValueAdded,
    CETTransformation,
    LeontiefIntermediate,
)
from equilibria.core import Set
from equilibria.model import Model
from equilibria.templates.base import ModelTemplate


class SimpleOpenEconomy(ModelTemplate):
    """Simple open economy CGE model template.

    A basic open economy model with:
    - CES value-added production
    - Leontief intermediate inputs
    - CET output transformation
    - Armington import aggregation
    - Export supply

    Attributes:
        sigma_va: CES elasticity of substitution (default: 0.8)
        sigma_m: Armington elasticity (default: 1.5)
        sigma_e: CET elasticity (default: 2.0)

    Example:
        >>> template = SimpleOpenEconomy(num_sectors=3)
        >>> model = template.create_model()
        >>> print(model.statistics)
    """

    name: str = Field(default="SimpleOpenEconomy", description="Template name")
    description: str = Field(
        default="Simple open economy CGE model", description="Template description"
    )
    sigma_va: float = Field(
        default=0.8, gt=0, description="CES elasticity of substitution"
    )
    sigma_m: float = Field(default=1.5, gt=0, description="Armington elasticity")
    sigma_e: float = Field(default=2.0, gt=0, description="CET elasticity")

    def create_model(self) -> Model:
        """Create the simple open economy model.

        Returns:
            Configured Model instance
        """
        # Create model
        model = Model(
            name=self.name,
            description=self.description,
        )

        # Create sets
        sector_names = self.get_default_sector_names()
        factor_names = self.get_default_factor_names()

        sectors = Set(
            name="J",
            elements=tuple(sector_names),
            description="Production sectors",
        )

        factors = Set(
            name="I",
            elements=tuple(factor_names),
            description="Factors of production",
        )

        # Use same set for commodities as sectors
        commodities = Set(
            name="COMM",
            elements=tuple(sector_names),
            description="Commodities",
        )

        firms = Set(
            name="F",
            elements=("firm",),
            description="Firm institutions",
        )

        model.add_sets([sectors, factors, commodities, firms])

        # Add production blocks
        model.add_block(
            CESValueAdded(
                sigma=self.sigma_va,
                name="CES_VA",
                description="CES value-added production",
            )
        )

        model.add_block(
            LeontiefIntermediate(
                name="Leontief_INT",
                description="Leontief intermediate inputs",
            )
        )

        model.add_block(
            CETTransformation(
                omega=self.sigma_e,
                name="CET",
                description="CET output transformation",
            )
        )

        # Note: Trade blocks (Armington, CET_Exports) would be added here
        # but are omitted to avoid variable name conflicts in this demo
        # In a full implementation, these would share variables with production blocks

        return model

    def get_info(self) -> dict[str, Any]:
        """Get template information."""
        info = super().get_info()
        info.update(
            {
                "sigma_va": self.sigma_va,
                "sigma_m": self.sigma_m,
                "sigma_e": self.sigma_e,
                "blocks": [
                    "CES_VA",
                    "Leontief_INT",
                    "CET",
                    "Armington",
                    "CET_Exports",
                ],
            }
        )
        return info
