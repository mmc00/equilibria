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
from equilibria.blocks.demand import CobbDouglasConsumer, LESConsumer
from equilibria.blocks.equilibrium import (
    FactorMarketClearing,
    MarketClearing,
    PEPMacroClosureInit,
    PriceNormalization,
)
from equilibria.blocks.institutions import Government, Household, RestOfWorld
from equilibria.blocks.production import (
    CESValueAdded,
    CETTransformation,
    LeontiefIntermediate,
    PEPProductionAccountingInit,
)
from equilibria.blocks.trade import (
    ArmingtonCES,
    CETExports,
    PEPCommodityBalanceInit,
    PEPTradeFlowInit,
    PEPTradeMarketClearingInit,
    PEPTradeTransformationInit,
)

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
    "PEPProductionAccountingInit",
    # Trade blocks
    "ArmingtonCES",
    "CETExports",
    "PEPCommodityBalanceInit",
    "PEPTradeFlowInit",
    "PEPTradeMarketClearingInit",
    "PEPTradeTransformationInit",
    # Demand blocks
    "LESConsumer",
    "CobbDouglasConsumer",
    # Institution blocks
    "Household",
    "Government",
    "RestOfWorld",
    # Equilibrium blocks
    "MarketClearing",
    "PriceNormalization",
    "FactorMarketClearing",
    "PEPMacroClosureInit",
]
