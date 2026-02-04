"""equilibria - A Modern Python Framework for CGE Modeling."""

from equilibria.blocks import Block, register_block
from equilibria.core import (
    Equation,
    Parameter,
    Set,
    SetManager,
    Variable,
)
from equilibria.model import Model
from equilibria.version import __version__

__all__ = [
    "__version__",
    "Model",
    "Block",
    "register_block",
    "Set",
    "SetManager",
    "Parameter",
    "Variable",
    "Equation",
]
