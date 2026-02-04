"""Templates module for equilibria CGE framework.

Provides pre-configured model templates for common CGE model types.
"""

from equilibria.templates.base import ModelTemplate
from equilibria.templates.pep import PEP1R, PEPBaseTemplate, PEPSetManager
from equilibria.templates.simple_open import SimpleOpenEconomy

__all__ = [
    "ModelTemplate",
    "SimpleOpenEconomy",
    # PEP Templates
    "PEPBaseTemplate",
    "PEP1R",
    "PEPSetManager",
]
