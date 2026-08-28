"""GTAP6 symbolic Block units.

5 units (leaf -> closure), fewer than GTAP7's 7 because v6.2 has no
make-matrix / MRIO / output-CET split: TradeArmington, Production, Factor,
DemandUtility, IncomeClosure.

Mirrors ``equilibria.blocks.gtap`` (GTAP7's Block-extraction package) but for
the v6.2 monolith (``scripts/gtap6/_v62_monolith_oracle.py``,
``GTAP6MonolithOracle``). Each block is form+domain gated against that
oracle at the benchmark point (``tests/blocks/gtap6/test_gtap6_blocks_form.py``).

``GTAP6_BLOCK_ORDER`` is the leaf-to-closure composition order Task 10's
composer imports to assemble the full solvable model.
"""

from __future__ import annotations

from equilibria.blocks.gtap6.demand_utility import DemandUtilityBlock
from equilibria.blocks.gtap6.factor import FactorBlock
from equilibria.blocks.gtap6.income_closure import IncomeClosureBlock
from equilibria.blocks.gtap6.production import ProductionBlock
from equilibria.blocks.gtap6.trade_armington import TradeArmingtonBlock

GTAP6_BLOCK_ORDER = [
    TradeArmingtonBlock,
    ProductionBlock,
    FactorBlock,
    DemandUtilityBlock,
    IncomeClosureBlock,
]

__all__ = [
    "DemandUtilityBlock",
    "FactorBlock",
    "GTAP6_BLOCK_ORDER",
    "IncomeClosureBlock",
    "ProductionBlock",
    "TradeArmingtonBlock",
]
