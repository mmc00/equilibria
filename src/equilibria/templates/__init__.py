"""Templates module for equilibria CGE framework.

Provides pre-configured model templates for common CGE model types.
"""

from equilibria.templates.base import ModelTemplate
from equilibria.templates.gams_comparison import (
    GAMSComparisonReport,
    GAMSComparisonResult,
    GAMSRunner,
    PEPGAMSComparator,
    SolutionComparator,
    run_gams_comparison,
)
from equilibria.templates.simple_open import SimpleOpenEconomy

__all__ = [
    "ModelTemplate",
    "SimpleOpenEconomy",
    # GAMS Comparison
    "GAMSComparisonResult",
    "GAMSComparisonReport",
    "GAMSRunner",
    "SolutionComparator",
    "PEPGAMSComparator",
    "run_gams_comparison",
]
