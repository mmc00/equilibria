"""Templates module for equilibria CGE framework.

Provides pre-configured model templates for common CGE model types.
"""

from __future__ import annotations

import warnings
from typing import Any

__all__ = [
    "ModelTemplate",
    "SimpleOpenEconomy",
    "PEPScenarioParityRunner",
    "PEPExportTaxParityRunner",
    "PEPGovernmentSpendingParityRunner",
    "PEPImportPriceParityRunner",
    "PEPImportShockParityRunner",
    "EquilibriaLevelsExtractor",
    "GAMSLevelsExtractor",
    "LevelsComparator",
    # GAMS Comparison
    "GAMSComparisonResult",
    "GAMSComparisonReport",
    "GAMSRunner",
    "SolutionComparator",
    "PEPGAMSComparator",
    "run_gams_comparison",
]


def __getattr__(name: str) -> Any:
    """Lazy-load template exports to avoid import cycles."""
    if name == "ModelTemplate":
        from equilibria.templates.base import ModelTemplate

        return ModelTemplate

    if name == "SimpleOpenEconomy":
        from equilibria.templates.simple_open import SimpleOpenEconomy

        return SimpleOpenEconomy

    if name == "PEPScenarioParityRunner":
        warnings.warn(
            "equilibria.templates.PEPScenarioParityRunner is deprecated. "
            "Use equilibria.simulations.PepSimulator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from equilibria.templates.pep_scenario_parity import PEPScenarioParityRunner

        return PEPScenarioParityRunner

    if name in {
        "PEPExportTaxParityRunner",
        "PEPGovernmentSpendingParityRunner",
        "PEPImportPriceParityRunner",
        "PEPImportShockParityRunner",
    }:
        warnings.warn(
            f"equilibria.templates.{name} is deprecated. "
            "Use equilibria.simulations.PepSimulator presets/methods instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from equilibria.templates.pep_scenario_parity import (
            PEPExportTaxParityRunner,
            PEPGovernmentSpendingParityRunner,
            PEPImportPriceParityRunner,
            PEPImportShockParityRunner,
        )

        return {
            "PEPExportTaxParityRunner": PEPExportTaxParityRunner,
            "PEPGovernmentSpendingParityRunner": PEPGovernmentSpendingParityRunner,
            "PEPImportPriceParityRunner": PEPImportPriceParityRunner,
            "PEPImportShockParityRunner": PEPImportShockParityRunner,
        }[name]

    if name in {"EquilibriaLevelsExtractor", "GAMSLevelsExtractor", "LevelsComparator"}:
        from equilibria.templates.pep_levels import (
            EquilibriaLevelsExtractor,
            GAMSLevelsExtractor,
            LevelsComparator,
        )

        return {
            "EquilibriaLevelsExtractor": EquilibriaLevelsExtractor,
            "GAMSLevelsExtractor": GAMSLevelsExtractor,
            "LevelsComparator": LevelsComparator,
        }[name]

    if name in {
        "GAMSComparisonResult",
        "GAMSComparisonReport",
        "GAMSRunner",
        "SolutionComparator",
        "PEPGAMSComparator",
        "run_gams_comparison",
    }:
        from equilibria.templates.gams_comparison import (
            GAMSComparisonReport,
            GAMSComparisonResult,
            GAMSRunner,
            PEPGAMSComparator,
            SolutionComparator,
            run_gams_comparison,
        )

        return {
            "GAMSComparisonResult": GAMSComparisonResult,
            "GAMSComparisonReport": GAMSComparisonReport,
            "GAMSRunner": GAMSRunner,
            "SolutionComparator": SolutionComparator,
            "PEPGAMSComparator": PEPGAMSComparator,
            "run_gams_comparison": run_gams_comparison,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
