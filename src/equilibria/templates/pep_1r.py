"""PEP Single Region Template.

This module provides the PEP single region (1R) template,
a complete implementation of the PEP standard CGE model.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from equilibria.blocks import (
    ArmingtonCES,
    CETExports,
    CETTransformation,
    CESValueAdded,
    CobbDouglasConsumer,
    FactorMarketClearing,
    Government,
    Household,
    LeontiefIntermediate,
    LESConsumer,
    MarketClearing,
    PriceNormalization,
    RestOfWorld,
)
from equilibria.core import Set
from equilibria.model import Model
from equilibria.templates.pep_base import PEPBaseTemplate


class PEP1R(PEPBaseTemplate):
    """PEP Single Region CGE Model Template.

    Full implementation of PEP-1-1_v2_1_modular.gms with:
    - 4 household types (hrp, hrr, hup, hur)
    - Multiple labor and capital types
    - Complete tax system (ti, tm, tx, td)
    - Trade margins and transport costs
    - LES demand system with Frisch parameters

    Example:
        >>> template = PEP1R()
        >>> model = template.create_model()
        >>> print(model.statistics)
    """

    name: str = Field(default="PEP-1R", description="Template name")
    description: str = Field(
        default="PEP Single Region CGE Model", description="Template description"
    )

    def create_model(self) -> Model:
        """Create the PEP single region model.

        Returns:
            Configured Model instance with all PEP blocks
        """
        # Create model
        model = Model(name=self.name, description=self.description)

        # Extract sets from SAM if not already done
        if not self.sectors:
            self.extract_sets_from_sam()

        # Create sets
        self._create_sets(model)

        # Add production blocks
        self._add_production_blocks(model)

        # Add trade blocks
        self._add_trade_blocks(model)

        # Add demand blocks
        self._add_demand_blocks(model)

        # Add institution blocks
        self._add_institution_blocks(model)

        # Add equilibrium blocks
        self._add_equilibrium_blocks(model)

        return model

    def _create_sets(self, model: Model) -> None:
        """Create all PEP sets in the model."""
        # Industries/Commodities
        sectors_set = Set(
            name="J",
            elements=tuple(self.sectors),
            description="Industries/sectors",
        )
        commodities_set = Set(
            name="I",
            elements=tuple(self.commodities),
            description="Commodities",
        )

        # Factors
        labor_set = Set(
            name="L",
            elements=tuple(self.labor_types),
            description="Labor types",
        )
        capital_set = Set(
            name="K",
            elements=tuple(self.capital_types),
            description="Capital types",
        )

        # Agents
        households_set = Set(
            name="H",
            elements=tuple(self.households),
            description="Households",
        )

        # All agents (households + firms + government + ROW + taxes)
        all_agents = (
            list(self.households)
            + ["firm", "gvt", "row"]
            + (["ti", "tm", "tx", "td"] if self.include_all_taxes else [])
        )
        agents_set = Set(
            name="AG",
            elements=tuple(all_agents),
            description="All agents",
        )

        model.add_sets(
            [
                sectors_set,
                commodities_set,
                labor_set,
                capital_set,
                households_set,
                agents_set,
            ]
        )

    def _add_production_blocks(self, model: Model) -> None:
        """Add production-related blocks."""
        # CES Value-Added production
        model.add_block(
            CESValueAdded(
                sigma=0.8,  # Will be overridden by calibration
                name="CES_VA",
                description="CES value-added production",
            )
        )

        # Leontief intermediate inputs
        model.add_block(
            LeontiefIntermediate(
                name="Leontief_INT",
                description="Leontief intermediate inputs",
            )
        )

        # CET output transformation
        model.add_block(
            CETTransformation(
                omega=2.0,  # Will be overridden by calibration
                name="CET",
                description="CET output transformation",
            )
        )

    def _add_trade_blocks(self, model: Model) -> None:
        """Add trade-related blocks."""
        # Armington import aggregation
        model.add_block(
            ArmingtonCES(
                sigma_m=1.5,  # Will be overridden by calibration
                name="Armington",
                description="Armington import aggregation",
            )
        )

        # CET export transformation
        model.add_block(
            CETExports(
                sigma_e=2.0,  # Will be overridden by calibration
                name="CET_Exports",
                description="CET export transformation",
            )
        )

    def _add_demand_blocks(self, model: Model) -> None:
        """Add consumer demand blocks."""
        # Add LES consumer for each household type
        for household in self.households:
            model.add_block(
                LESConsumer(
                    name=f"LES_{household}",
                    description=f"LES consumer for {household}",
                )
            )

    def _add_institution_blocks(self, model: Model) -> None:
        """Add institution blocks."""
        # Household income aggregation
        model.add_block(
            Household(
                name="Household",
                description="Household income and expenditure",
            )
        )

        # Government budget
        if self.include_all_taxes:
            model.add_block(
                Government(
                    name="Government",
                    description="Government budget",
                )
            )

        # Rest of world
        model.add_block(
            RestOfWorld(
                name="ROW",
                description="Rest of world (foreign sector)",
            )
        )

    def _add_equilibrium_blocks(self, model: Model) -> None:
        """Add equilibrium condition blocks."""
        # Commodity market clearing
        model.add_block(
            MarketClearing(
                name="MarketClearing",
                description="Commodity market clearing",
            )
        )

        # Factor market clearing
        model.add_block(
            FactorMarketClearing(
                name="FactorMarket",
                description="Factor market clearing",
            )
        )

        # Price normalization
        model.add_block(
            PriceNormalization(
                name="PriceNorm",
                description="Price normalization (numeraire)",
            )
        )

    def get_info(self) -> dict[str, Any]:
        """Get template information."""
        info = super().get_info()
        info.update(
            {
                "variant": "1R (Single Region)",
                "blocks": [
                    "CES_VA",
                    "Leontief_INT",
                    "CET",
                    "Armington",
                    "CET_Exports",
                    f"LES_Consumer (x{len(self.households)})",
                    "Household",
                    "Government",
                    "ROW",
                    "MarketClearing",
                    "FactorMarketClearing",
                    "PriceNorm",
                ],
            }
        )
        return info
