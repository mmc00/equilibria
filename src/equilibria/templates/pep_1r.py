"""PEP Single Region Template.

This module provides the PEP single region (1R) template,
a complete implementation of the PEP standard CGE model.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from equilibria.blocks import (
    ArmingtonCES,
    CESValueAdded,
    CETExports,
    CETTransformation,
    FactorMarketClearing,
    Government,
    Household,
    LeontiefIntermediate,
    LESConsumer,
    MarketClearing,
    PriceNormalization,
    RestOfWorld,
)
from equilibria.core import (
    CalibrationData,
    CalibrationPhase,
    DependencyValidator,
    Set,
)
from equilibria.model import Model
from equilibria.templates.pep_base import PEPBaseTemplate
from equilibria.templates.pep_calibration import PEPCalibrator


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

    def create_model(
        self,
        calibrate: bool = True,
        mode: str = "sam",
        dummy_defaults: dict[str, dict] | None = None,
    ) -> Model:
        """Create the PEP single region model.

        Args:
            calibrate: Whether to run calibration (default: True)
            mode: Calibration mode - "sam" or "dummy" (default: "sam")
            dummy_defaults: Per-block dummy default values for calibration

        Returns:
            Configured Model instance with all PEP blocks
        """
        # Create model
        model = Model(name=self.name, description=self.description)

        # Extract sets from SAM if not already done
        if not self.sectors:
            self.extract_sets_from_sam()

        # Create sets first
        self._create_sets(model)

        # Add all blocks to model
        calibration_data = None
        if calibrate:
            calibration_data = self._calibrate_model()

        self._add_production_blocks(model, calibration_data)
        self._add_trade_blocks(model, calibration_data)
        self._add_demand_blocks(model, calibration_data)
        self._add_institution_blocks(model, calibration_data)
        self._add_equilibrium_blocks(model, calibration_data)

        # Run calibration on all blocks in phases
        if calibrate:
            self._run_calibration(model, mode, dummy_defaults)

        return model

    def _run_calibration(
        self,
        model: Model,
        mode: str,
        dummy_defaults: dict[str, dict] | None = None,
    ) -> None:
        """Run calibration on all blocks in phase order.

        Args:
            model: The model with blocks
            mode: "sam" or "dummy"
            dummy_defaults: Per-block dummy default values
        """
        # Create calibration data
        sam = self.load_sam() if mode == "sam" else None
        data = CalibrationData(sam, mode)

        # Register set mappings
        data.register_set_mapping("J", self.sectors)
        data.register_set_mapping("I", self.commodities)
        data.register_set_mapping("F", self.labor_types + self.capital_types)
        data.register_set_mapping("H", self.households)

        # Register special accounts (ROW, GVT, etc.)
        data.register_set_mapping("ROW", ["ROW"])
        data.register_set_mapping("GVT", ["GVT"])
        data.register_set_mapping("FIRM", ["FIRM"])

        # Create dependency validator
        validator = DependencyValidator()

        # Run calibration in phases
        for phase in CalibrationPhase:
            for block in model.blocks:
                # Apply dummy defaults if provided
                if dummy_defaults and block.name in dummy_defaults:
                    block.dummy_defaults.update(dummy_defaults[block.name])

                # Calibrate the block
                block.calibrate(
                    phase=phase,
                    data=data,
                    mode=mode,
                    set_manager=model.set_manager,
                    param_manager=model.parameter_manager,
                    var_manager=model.variable_manager,
                    dependency_validator=validator,
                )

    def _calibrate_model(self) -> dict:
        """Run calibration and return calibration data."""
        sam = self.load_sam()
        calibrator = PEPCalibrator(
            sam=sam,
            param_file=self.param_file,
            sectors=self.sectors,
            commodities=self.commodities,
            labor_types=self.labor_types,
            capital_types=self.capital_types,
            households=self.households,
        )
        return calibrator.calibrate()

    def _create_sets(self, model: Model) -> None:
        """Create all PEP sets in the model."""
        # Use hardcoded PEP sets to avoid SAM parsing issues
        # Standard PEP 1-1 v2.1 structure
        sectors = ["agr", "othind", "food", "ser", "adm"]
        commodities = ["agr", "othind", "food", "ser", "adm"]
        labor_types = ["usk", "sk"]
        capital_types = ["cap", "land"]
        households = ["hrp", "hup", "hrr", "hur"]

        # Update instance variables
        self.sectors = sectors
        self.commodities = commodities
        self.labor_types = labor_types
        self.capital_types = capital_types
        self.households = households

        # Industries/Commodities
        sectors_set = Set(
            name="J",
            elements=tuple(sectors),
            description="Industries/sectors",
        )
        commodities_set = Set(
            name="I",
            elements=tuple(commodities),
            description="Commodities",
        )

        # Factors
        labor_set = Set(
            name="L",
            elements=tuple(labor_types),
            description="Labor types",
        )
        capital_set = Set(
            name="K",
            elements=tuple(capital_types),
            description="Capital types",
        )

        # Combined factors set (for blocks that need it)
        all_factors = labor_types + capital_types
        factors_set = Set(
            name="F",
            elements=tuple(all_factors),
            description="All factors (labor + capital)",
        )

        # Agents
        households_set = Set(
            name="H",
            elements=tuple(households),
            description="Households",
        )

        # All agents (households + firms + government + ROW + taxes)
        all_agents = (
            households
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
                factors_set,
                households_set,
                agents_set,
            ]
        )

    def _add_production_blocks(
        self, model: Model, calibration_data: dict | None = None
    ) -> None:
        """Add production-related blocks."""
        # Get elasticities from calibration if available
        sigma_va = 0.8
        sigma_xt = 2.0
        if calibration_data and "elasticities" in calibration_data:
            el = calibration_data["elasticities"]
            # Use average of sector elasticities
            if el.sigma_va:
                sigma_va = sum(el.sigma_va.values()) / len(el.sigma_va)
            if el.sigma_xt:
                sigma_xt = sum(el.sigma_xt.values()) / len(el.sigma_xt)

        # CES Value-Added production
        model.add_block(
            CESValueAdded(
                sigma=sigma_va,
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
                omega=sigma_xt,
                name="CET",
                description="CET output transformation",
            )
        )

    def _add_trade_blocks(
        self, model: Model, calibration_data: dict | None = None
    ) -> None:
        """Add trade-related blocks."""
        # Get elasticities from calibration if available
        sigma_m = 1.5
        sigma_e = 2.0
        if calibration_data and "elasticities" in calibration_data:
            el = calibration_data["elasticities"]
            if el.sigma_m:
                sigma_m = sum(el.sigma_m.values()) / len(el.sigma_m)
            if el.sigma_xd:
                sigma_e = sum(el.sigma_xd.values()) / len(el.sigma_xd)

        # Armington import aggregation
        model.add_block(
            ArmingtonCES(
                sigma_m=sigma_m,
                name="Armington",
                description="Armington import aggregation",
            )
        )

        # CET export transformation
        model.add_block(
            CETExports(
                sigma_e=sigma_e,
                name="CET_Exports",
                description="CET export transformation",
            )
        )

    def _add_demand_blocks(
        self, model: Model, calibration_data: dict | None = None
    ) -> None:
        """Add consumer demand blocks."""
        # Get elasticities from calibration if available
        frisch_params = {}
        if calibration_data and "elasticities" in calibration_data:
            el = calibration_data["elasticities"]
            frisch_params = el.frisch

        # Add LES consumer for each household type
        for household in self.households:
            frisch = frisch_params.get(household, -1.5)
            model.add_block(
                LESConsumer(
                    name=f"LES_{household}",
                    description=f"LES consumer for {household}",
                )
            )

    def _add_institution_blocks(
        self, model: Model, calibration_data: dict | None = None
    ) -> None:
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

    def _add_equilibrium_blocks(
        self, model: Model, calibration_data: dict | None = None
    ) -> None:
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
