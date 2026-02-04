"""Blocks module for equilibria CGE modeling.

Blocks are modular equation components that define economic behavior.
Each block is a Pydantic model that declares its required sets,
parameters, variables, and equations.
"""

from equilibria.blocks.base import (
    Block,
    BlockRegistry,
    EquationSpec,
    ParameterSpec,
    VariableSpec,
    get_registry,
    register_block,
)
from equilibria.blocks.production import (
    CESValueAdded,
    CETTransformation,
    LeontiefIntermediate,
)
from equilibria.blocks.trade import ArmingtonCES, CETExports

__all__ = [
    "Block",
    "BlockRegistry",
    "ParameterSpec",
    "VariableSpec",
    "EquationSpec",
    "get_registry",
    "register_block",
    # Production blocks
    "CESValueAdded",
    "LeontiefIntermediate",
    "CETTransformation",
    # Trade blocks
    "ArmingtonCES",
    "CETExports",
]
