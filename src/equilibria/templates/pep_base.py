"""PEP Base Template for equilibria CGE framework.

This module provides the base class for all PEP model variants,
including shared functionality for set management, block assembly,
and GAMS comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from equilibria.babel import SAM
from equilibria.templates.base import ModelTemplate


class PEPBaseTemplate(ModelTemplate):
    """Base class for all PEP model variants.

    Provides shared functionality for:
    - Set management (auto-detect from SAM)
    - Block assembly
    - Calibration orchestration
    - GAMS comparison utilities

    Attributes:
        sam_file: Path to SAM Excel file
        param_file: Path to VAL_PAR Excel file
        sectors: List of sector names (auto-populated)
        commodities: List of commodity names (auto-populated)
        labor_types: List of labor type names (auto-populated)
        capital_types: List of capital type names (auto-populated)
        households: List of household names (auto-populated)
    """

    name: str = Field(default="PEPBase", description="Template name")
    description: str = Field(
        default="PEP Base Template", description="Template description"
    )

    # File paths
    sam_file: Path | None = Field(default=None, description="Path to SAM Excel file")
    param_file: Path | None = Field(
        default=None, description="Path to VAL_PAR Excel file"
    )

    # Dynamic sets (auto-populated from SAM)
    sectors: list[str] = Field(default_factory=list, description="Industries/sectors")
    commodities: list[str] = Field(default_factory=list, description="Commodities")
    labor_types: list[str] = Field(default_factory=list, description="Labor types")
    capital_types: list[str] = Field(default_factory=list, description="Capital types")
    households: list[str] = Field(default_factory=list, description="Households")

    # Features (all enabled by default)
    include_margins: bool = Field(default=True, description="Include trade margins")
    include_all_taxes: bool = Field(
        default=True, description="Include all tax instruments (ti, tm, tx, td)"
    )
    include_transfers: bool = Field(default=True, description="Include transfers")
    include_inventory: bool = Field(default=True, description="Include inventory")

    model_config = {"arbitrary_types_allowed": True}

    def load_sam(self) -> SAM:
        """Load SAM from file.

        Returns:
            SAM object

        Raises:
            FileNotFoundError: If sam_file not set or file doesn't exist
        """
        if self.sam_file is None:
            from equilibria.templates.data.pep import load_default_pep_sam

            return load_default_pep_sam()

        from equilibria.templates.data.pep import load_pep_sam

        return load_pep_sam(self.sam_file)

    def load_parameters(self) -> dict[str, Any]:
        """Load parameters from VAL_PAR file.

        Returns:
            Dictionary of parameters

        Raises:
            FileNotFoundError: If param_file not set or file doesn't exist
        """
        if self.param_file is None:
            from equilibria.templates.data.pep import load_default_pep_parameters

            return load_default_pep_parameters()

        from equilibria.templates.data.pep import load_pep_parameters

        return load_pep_parameters(self.param_file)

    def extract_sets_from_sam(self, sam: SAM | None = None) -> None:
        """Extract PEP sets from SAM structure.

        Populates sectors, commodities, labor_types, capital_types,
        and households from SAM account structure.

        Args:
            sam: SAM object (loads default if not provided)
        """
        if sam is None:
            sam = self.load_sam()

        if sam is None:
            raise ValueError("No SAM provided")

        from equilibria.templates.pep_sets import PEPSetManager

        manager = PEPSetManager(sam)
        sets = manager.sets

        self.sectors = list(sets["J"].elements)
        self.commodities = list(sets["I"].elements)
        self.labor_types = list(sets["L"].elements)
        self.capital_types = list(sets["K"].elements)
        self.households = list(sets["H"].elements)

    def extract_sets_from_gdx_data(self, gdx_data: dict) -> None:
        """Extract PEP sets from GDX data structure.

        Populates sectors, commodities, labor_types, capital_types,
        and households from GDX UEL (Unique Element List).

        Args:
            gdx_data: Dictionary from read_gdx() containing GDX data
        """
        elements = gdx_data.get("elements", [])
        
        # Categorize elements based on PEP naming conventions
        # This uses heuristics based on standard PEP element names
        
        # Sectors/Commodities: typically agr, food, othind, ser, adm
        sector_keywords = ["agr", "food", "othind", "ser", "adm", "ind", "manuf"]
        self.sectors = [e for e in elements if any(kw in e.lower() for kw in sector_keywords)]
        self.commodities = self.sectors.copy()  # In PEP, I = J
        
        # Labor types: usk (unskilled), sk (skilled), and variants
        labor_keywords = ["usk", "sk", "lab", "labor", "worker"]
        self.labor_types = [e for e in elements if any(kw in e.lower() for kw in labor_keywords)]
        
        # Capital types: cap, land, and variants
        capital_keywords = ["cap", "land", "capital"]
        self.capital_types = [e for e in elements if any(kw in e.lower() for kw in capital_keywords)]
        
        # Households: hrp, hrr, hup, hur (PEP standard household types)
        household_keywords = ["hrp", "hrr", "hup", "hur", "household", "hh"]
        self.households = [e for e in elements if any(kw in e.lower() for kw in household_keywords)]
        
        # If no matches found, use default PEP sets
        if not self.sectors:
            self.use_default_pep_sets()

    def use_default_pep_sets(self) -> None:
        """Use default PEP sets (hardcoded for standard PEP model).

        This is a temporary solution until full SAM parsing is implemented.
        Uses the standard PEP V2.0 structure.
        """
        self.sectors = ["agr", "othind", "food", "ser", "adm"]
        self.commodities = ["agr", "othind", "food", "ser", "adm"]
        self.labor_types = ["usk", "sk"]
        self.capital_types = ["cap", "land"]
        self.households = ["hrp", "hup", "hrr", "hur"]

    def sync_sets_to_gams(self, output_path: Path | None = None) -> Path:
        """Generate GAMS include file from current sets.

        Args:
            output_path: Where to write include file (optional)

        Returns:
            Path to generated file
        """
        from equilibria.templates.pep_sets import PEPSetManager

        if not self.sectors:
            self.extract_sets_from_sam()

        # Create temporary SAM with current sets
        # This is a simplified version - full version would reconstruct SAM
        manager = PEPSetManager()

        # Create sets manually
        from equilibria.core import Set

        manager.sets = {
            "J": Set(name="J", elements=tuple(self.sectors)),
            "I": Set(name="I", elements=tuple(self.commodities)),
            "L": Set(name="L", elements=tuple(self.labor_types)),
            "K": Set(name="K", elements=tuple(self.capital_types)),
            "H": Set(name="H", elements=tuple(self.households)),
        }

        if output_path is None:
            output_path = Path("sets_definition.inc")

        manager.generate_gams_include(output_path)
        return output_path

    def get_info(self) -> dict[str, Any]:
        """Get template information."""
        info = super().get_info()
        info.update(
            {
                "sam_file": str(self.sam_file) if self.sam_file else None,
                "param_file": str(self.param_file) if self.param_file else None,
                "n_sectors": len(self.sectors),
                "n_commodities": len(self.commodities),
                "n_labor_types": len(self.labor_types),
                "n_capital_types": len(self.capital_types),
                "n_households": len(self.households),
                "features": {
                    "margins": self.include_margins,
                    "taxes": self.include_all_taxes,
                    "transfers": self.include_transfers,
                    "inventory": self.include_inventory,
                },
            }
        )
        return info
