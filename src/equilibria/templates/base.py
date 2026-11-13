"""Template base class for equilibria CGE models.

Templates provide pre-configured models with sensible defaults
for common CGE model types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, PrivateAttr

from equilibria.model import Model

if TYPE_CHECKING:
    from equilibria.blocks.base import Block
    from equilibria.core.recipe import Recipe


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
        >>> template = SimpleOpenEconomy()
        >>> model = template.create_model()
        >>> print(model.statistics)
    """

    name: str = Field(..., description="Template name")
    description: str = Field(default="", description="Template description")
    num_sectors: int = Field(default=3, gt=0, description="Number of sectors")
    num_factors: int = Field(default=2, gt=0, description="Number of factors")

    model_config = {"arbitrary_types_allowed": True}

    # reflected models, keyed by the period list and the template's fields: a
    # template changed in place (e.g. an elasticity sweep) gets a new model
    _built: dict[tuple[tuple[str, ...], str], Any] = PrivateAttr(default_factory=dict)

    @abstractmethod
    def create_model(self) -> Model:
        """Create a model from this template.

        Returns:
            Configured Model instance
        """
        ...

    def period_model(self, overwrite: Sequence[Block] = ()) -> Any:
        """The single-period Pyomo model, each ``overwrite`` block replacing the
        template's block of the same name (ADR-0004). Templates that run under
        a recipe implement it."""
        raise NotImplementedError(f"{type(self).__name__} has no period model")

    def default_recipe(self) -> Recipe:
        """The recipe an ``Exercise`` uses when it is given none."""
        raise NotImplementedError(f"{type(self).__name__} has no default recipe")

    def build(self, periods: Sequence[str], overwrite: Sequence[Block] = ()) -> Any:
        """A fresh model with ``periods``, never shared with another call.

        Without ``overwrite`` the reflected model is built once per period list
        and per value of the template's fields, and every call returns a copy
        of it; with ``overwrite`` it is built fresh and kept nowhere. The key
        is the fields, not the files they point to: a data file edited on disk
        needs a new template.
        """
        from equilibria.core.reflection import reflect

        if overwrite:
            return reflect(self.period_model(overwrite), periods)
        key = (tuple(periods), self.model_dump_json())
        if key not in self._built:
            self._built[key] = reflect(self.period_model(), periods)
        return self._built[key].clone()

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
