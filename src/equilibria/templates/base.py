"""Template base class for equilibria CGE models.

Templates provide pre-configured models with sensible defaults
for common CGE model types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from equilibria.model import Model


class ModelTemplate(BaseModel, ABC):
    """Base class for CGE model templates.

    Templates provide pre-configured models with standard sets,
    blocks, and default parameter values for common CGE model types.

    Attributes:
        name: Template name
        description: Template description
        num_sectors: Number of production sectors
        num_factors: Number of factors of production

    Example:
        >>> template = SimpleOpenEconomy(num_sectors=3)
        >>> model = template.create_model()
        >>> print(model.statistics)
    """

    name: str = Field(..., description="Template name")
    description: str = Field(default="", description="Template description")
    num_sectors: int = Field(default=3, gt=0, description="Number of sectors")
    num_factors: int = Field(default=2, gt=0, description="Number of factors")

    model_config = {"arbitrary_types_allowed": True}

    @abstractmethod
    def create_model(self) -> Model:
        """Create a model from this template.

        Returns:
            Configured Model instance
        """
        ...

    def get_default_sector_names(self) -> list[str]:
        """Get default sector names.

        Returns:
            List of sector identifiers
        """
        if self.num_sectors <= 5:
            defaults = ["AGR", "MFG", "SRV", "ENE", "CON"]
            return defaults[: self.num_sectors]
        else:
            return [f"SEC{i + 1}" for i in range(self.num_sectors)]

    def get_default_factor_names(self) -> list[str]:
        """Get default factor names.

        Returns:
            List of factor identifiers
        """
        defaults = ["LAB", "CAP", "LAND", "NAT"]
        return defaults[: min(self.num_factors, len(defaults))]

    def get_info(self) -> dict[str, Any]:
        """Get template information.

        Returns:
            Dictionary with template metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "num_sectors": self.num_sectors,
            "num_factors": self.num_factors,
            "sector_names": self.get_default_sector_names(),
            "factor_names": self.get_default_factor_names(),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"{self.num_sectors} sectors, "
            f"{self.num_factors} factors)"
        )
