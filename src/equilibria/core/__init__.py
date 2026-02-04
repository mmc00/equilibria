"""Core data structures for equilibria CGE modeling framework.

This module provides the fundamental building blocks for CGE models:
- Sets: Index definitions for multi-dimensional data
- Parameters: Constant values (calibrated from SAM)
- Variables: Endogenous model variables
- Equations: Mathematical relationships
"""

from equilibria.core.equations import Equation
from equilibria.core.parameters import Parameter
from equilibria.core.sets import Set, SetManager
from equilibria.core.variables import Variable

__all__ = [
    "Set",
    "SetManager",
    "Parameter",
    "Variable",
    "Equation",
]
