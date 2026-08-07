"""Log-value GTAPv7 blocks (parallel to equilibria.blocks.gtap, in log form).

Same model / economics / calibration as the gtap_julia port, re-expressed as
composable Blocks whose equations are log(lhs)==log(rhs) with multiplicative-power
taxes. GTAP_LOGVALUE_BLOCK_ORDER is dependency order (leaf → closure); shared vars
dedup by name, first registration wins.
"""

from .closure import ClosureLVBlock
from .demand_utility import DemandUtilityLVBlock
from .factor import FactorLVBlock
from .income import IncomeLVBlock
from .production_supply import ProductionSupplyLVBlock
from .trade_armington import ArmingtonBilateralLVBlock
from .trade_cet import TradeCETLVBlock

GTAP_LOGVALUE_BLOCK_ORDER = [
    ProductionSupplyLVBlock,
    FactorLVBlock,
    ArmingtonBilateralLVBlock,
    TradeCETLVBlock,
    DemandUtilityLVBlock,
    IncomeLVBlock,
    ClosureLVBlock,
]

__all__ = [
    "GTAP_LOGVALUE_BLOCK_ORDER",
    "ProductionSupplyLVBlock",
    "FactorLVBlock",
    "ArmingtonBilateralLVBlock",
    "TradeCETLVBlock",
    "DemandUtilityLVBlock",
    "IncomeLVBlock",
    "ClosureLVBlock",
]
