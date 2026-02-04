"""PEP (Poverty and Equity Program) CGE model templates.

This module provides PEP standard CGE model templates for equilibria,
including single region, multi-region, and dynamic variants.

The PEP model is a comprehensive CGE framework developed by:
- Veronique Robichaud
- Andre Lemelin
- Helene Maisonnave
- Bernard Decaluwe

Example:
    >>> from equilibria.templates import PEP1R
    >>> template = PEP1R()
    >>> model = template.create_model()
    >>> print(model.statistics)
"""

from equilibria.templates.pep_1r import PEP1R
from equilibria.templates.pep_base import PEPBaseTemplate
from equilibria.templates.pep_sets import PEPSetManager

__all__ = [
    "PEPBaseTemplate",
    "PEP1R",
    "PEPSetManager",
]
