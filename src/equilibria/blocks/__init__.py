"""Blocks module for equilibria CGE modeling.

Blocks are modular equation components that define economic behavior.
Each block is a Pydantic model that declares its required sets,
parameters, variables, and equations.
"""

from equilibria.blocks.base import (
    Block,
    EquationSpec,
    ParameterSpec,
    VariableSpec,
)
from equilibria.blocks.demand import (
    CobbDouglasConsumer,
    CobbDouglasHousehold,
    LESConsumer,
)
from equilibria.blocks.equilibrium import (
    FactorMarket,
    FactorMarketClearing,
    GoodsMarket,
    MarketClearing,
    Numeraire,
    PEPMacroClosureInit,
    PriceNormalization,
)
from equilibria.blocks.institutions import (
    ExternalSector,
    Government,
    Household,
    RestOfWorld,
    TaxGovernment,
)
from equilibria.blocks.production import (
    CESValueAdded,
    CETOutput,
    CETTransformation,
    CobbDouglasValueAdded,
    LeontiefComposite,
    LeontiefIntermediate,
    PEPProductionAccountingInit,
)
from equilibria.blocks.trade import (
    ArmingtonCES,
    ArmingtonComposite,
    CETExports,
    PEPCommodityBalanceInit,
    PEPTradeFlowInit,
    PEPTradeMarketClearingInit,
    PEPTradeTransformationInit,
)

__all__ = [
    "ArmingtonComposite",
    "CETOutput",
    "CobbDouglasHousehold",
    "CobbDouglasValueAdded",
    "ExternalSector",
    "FactorMarket",
    "GoodsMarket",
    "LeontiefComposite",
    "Numeraire",
    "TaxGovernment",
    "Block",
    "ParameterSpec",
    "VariableSpec",
    "EquationSpec",
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
