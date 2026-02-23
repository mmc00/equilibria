"""PEP Set Manager for dynamic set generation.

Manages dynamic set generation from SAM data and synchronization
between Python and GAMS representations.
"""

import re
from pathlib import Path
from typing import Any

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
            if acc.lower() in ["ti", "tm", "tx", "td"] or acc.lower().startswith("h") and len(acc) <= 4 or acc.lower() in ["firm", "gvt", "row"]:
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
        if not gams_include_path.exists():
            return False

        gams_sets = self._parse_gams_sets_include(gams_include_path)

        for set_name, set_obj in self.sets.items():
            expected = {str(elem).strip().lower() for elem in set_obj.elements}
            actual = gams_sets.get(set_name.upper())
            if actual is None:
                return False
            if expected != actual:
                return False

        return True

    def _parse_gams_sets_include(self, include_path: Path) -> dict[str, set[str]]:
        """Parse a GAMS include file and extract set memberships.

        Supports both common formats used in this repo:
        1) ``J Description / ... /``
        2) ``SET`` blocks where the opening slash is on the next line.
        """
        text = include_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        parsed: dict[str, set[str]] = {}
        current_set: str | None = None
        in_elements = False

        def add_elements(set_name: str, fragment: str) -> None:
            tokens = re.split(r"\s+", fragment.strip())
            for token in tokens:
                cleaned = token.strip().strip(",;").strip("'\"")
                if cleaned and cleaned not in {"/", ";"}:
                    parsed.setdefault(set_name, set()).add(cleaned.lower())

        for raw_line in lines:
            line = raw_line.strip()

            if not line or line.startswith("*"):
                continue

            if line.upper() == "SET" or line == ";":
                continue

            if in_elements and current_set is not None:
                if "/" in line:
                    before, _, _ = line.partition("/")
                    add_elements(current_set, before)
                    in_elements = False
                    current_set = None
                else:
                    add_elements(current_set, line)
                continue

            if current_set is not None:
                if "/" not in line:
                    continue
                _, _, after_open = line.partition("/")
                if "/" in after_open:
                    inside, _, _ = after_open.partition("/")
                    add_elements(current_set, inside)
                    current_set = None
                    in_elements = False
                else:
                    add_elements(current_set, after_open)
                    in_elements = True
                continue

            decl_match = re.match(
                r"^([A-Za-z][A-Za-z0-9_]*)(?:\([^)]*\))?",
                line,
            )
            if decl_match is None:
                continue

            set_name = decl_match.group(1).upper()
            parsed.setdefault(set_name, set())
            current_set = set_name

            if "/" in line:
                _, _, after_open = line.partition("/")
                if "/" in after_open:
                    inside, _, _ = after_open.partition("/")
                    add_elements(current_set, inside)
                    current_set = None
                    in_elements = False
                else:
                    add_elements(current_set, after_open)
                    in_elements = True

        return parsed

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
