"""PEP Set Manager for dynamic set generation.

Manages dynamic set generation from SAM data and synchronization
between Python and GAMS representations.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from equilibria.babel import SAM
from equilibria.core import Set


class PEPSetManager:
    """Manages PEP model sets with dynamic generation from SAM.

    Extracts set definitions from SAM structure and provides
    synchronization with GAMS include files.

    Attributes:
        sam: SAM object containing model data
        sets: Dictionary of set name to Set objects
    """

    def __init__(self, sam: SAM | None = None) -> None:
        """Initialize set manager.

        Args:
            sam: Optional SAM object to extract sets from
        """
        self.sam = sam
        self.sets: dict[str, Set] = {}

        if sam is not None:
            self.extract_sets_from_sam()

    def extract_sets_from_sam(self, sam: SAM | None = None) -> dict[str, Set]:
        """Extract PEP sets from SAM structure.

        Identifies:
        - J: Industries/sectors
        - I: Commodities
        - K: Capital types
        - L: Labor types
        - AG: All agents
        - H: Households (subset of AG)

        Args:
            sam: SAM object (uses self.sam if not provided)

        Returns:
            Dictionary of set name to Set objects
        """
        if sam is None:
            sam = self.sam

        if sam is None:
            raise ValueError("No SAM provided")

        # Get all accounts from SAM
        accounts = sam.data.index.tolist()

        # Categorize accounts based on PEP naming conventions
        # This is a simplified version - full version would use
        # pattern matching or configuration

        # Industries/Commodities (typically 5 sectors in PEP)
        industry_names = self._identify_sectors(accounts)

        # Factors (capital and labor)
        capital_names = self._identify_capital(accounts)
        labor_names = self._identify_labor(accounts)

        # Agents
        agent_names = self._identify_agents(accounts)
        household_names = self._identify_households(agent_names)

        # Create sets
        self.sets = {
            "J": Set(
                name="J",
                elements=tuple(industry_names),
                description="Industries/sectors",
            ),
            "I": Set(
                name="I",
                elements=tuple(industry_names),  # Same as J in PEP
                description="Commodities",
            ),
            "K": Set(
                name="K",
                elements=tuple(capital_names) if capital_names else ("cap",),
                description="Capital types",
            ),
            "L": Set(
                name="L",
                elements=tuple(labor_names) if labor_names else ("lab",),
                description="Labor types",
            ),
            "AG": Set(
                name="AG",
                elements=tuple(agent_names),
                description="All agents",
            ),
            "H": Set(
                name="H",
                elements=tuple(household_names),
                description="Households",
            ),
        }

        return self.sets

    def _identify_sectors(self, accounts: list[str]) -> list[str]:
        """Identify sector/industry accounts from SAM."""
        # In PEP, sectors are typically: agr, othind, food, ser, adm
        # They appear as both rows and columns with intermediate transactions

        # For now, use heuristics based on common patterns
        # Full implementation would use configuration or pattern matching

        sector_candidates = []
        for acc in accounts:
            # Skip obvious non-sectors
            if acc.lower() in ["ti", "tm", "tx", "td", "marg", "inv", "vstk"]:
                continue
            if acc.lower().startswith(("h", "firm", "gvt", "row", "lab", "cap")):
                continue
            sector_candidates.append(acc)

        return sector_candidates if sector_candidates else ["sec1", "sec2"]

    def _identify_capital(self, accounts: list[str]) -> list[str]:
        """Identify capital type accounts."""
        capital_types = []
        for acc in accounts:
            if "cap" in acc.lower() or "land" in acc.lower():
                capital_types.append(acc)
        return capital_types

    def _identify_labor(self, accounts: list[str]) -> list[str]:
        """Identify labor type accounts."""
        labor_types = []
        for acc in accounts:
            if any(x in acc.lower() for x in ["lab", "usk", "sk", "worker"]):
                labor_types.append(acc)
        return labor_types

    def _identify_agents(self, accounts: list[str]) -> list[str]:
        """Identify all agent accounts."""
        # Agents include: households, firms, government, ROW, taxes
        agents = []

        for acc in accounts:
            # Tax accounts
            if acc.lower() in ["ti", "tm", "tx", "td"]:
                agents.append(acc)
            # Households (typically start with 'h')
            elif acc.lower().startswith("h") and len(acc) <= 4:
                agents.append(acc)
            # Other agents
            elif acc.lower() in ["firm", "gvt", "row"]:
                agents.append(acc)

        return agents if agents else ["hh", "firm", "gvt", "row"]

    def _identify_households(self, agents: list[str]) -> list[str]:
        """Identify household accounts from agents."""
        households = [a for a in agents if a.lower().startswith("h") and len(a) <= 4]
        return households if households else ["hh"]

    def generate_gams_include(self, output_path: Path) -> None:
        """Generate GAMS sets_definition.inc file.

        Args:
            output_path: Path to write include file
        """
        lines = [
            "* Auto-generated sets from SAM structure",
            "* Generated by equilibria PEP template",
            "",
        ]

        for set_name, set_obj in self.sets.items():
            elements_str = "\n".join(f"  {elem}" for elem in set_obj.elements)
            lines.append(f"{set_name} {set_obj.description} /")
            lines.append(elements_str)
            lines.append("/")
            lines.append("")

        output_path.write_text("\n".join(lines))

    def validate_consistency(self, gams_include_path: Path) -> bool:
        """Validate that Python sets match GAMS sets.

        Args:
            gams_include_path: Path to GAMS include file

        Returns:
            True if consistent
        """
        # TODO: Implement comparison logic
        return True

    def get_set(self, name: str) -> Set:
        """Get a set by name."""
        return self.sets[name]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            name: {
                "elements": list(s.elements),
                "description": s.description,
            }
            for name, s in self.sets.items()
        }
