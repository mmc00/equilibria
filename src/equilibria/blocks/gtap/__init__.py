"""GTAP symbolic Block units (F3 Task 4). WIP.

Dependency order (leaf → closure). ``GTAP_BLOCK_ORDER`` is the order the composer
(Task 5) registers the blocks in; shared vars dedup by name (first registration
wins), so a var owned by an earlier block is skipped in a later one.

ifSUB wiring the composer MUST apply (mirrors the monolith post-block 7970-8036):
  * when ``closure.if_sub`` is True, DEACTIVATE the 9 defining eqs
    (eq_pp_rai, eq_xwmg, eq_xmgm, eq_pwmg, eq_pefobeq, eq_pmcifeq, eq_pmeq,
     eq_pfaeq, eq_pfyeq) and FIX their paired vars (pp_rai, xwmg, xmgm, pwmg,
     pefob, pmcif, pm, pfa, pfy) to the ``_m_*`` macro value at the built point.
  * the blocks always DEFINE all these eqs in the ``if_sub=False`` form (plain
    vars — ``_m_pp`` -> pp_rai, ``_m_pfa`` -> pfa, ...), which is what the Task-4
    form gate compares against the comp-stat oracle.
  * eq_pfact is DEFINED in CLOSURE (the monolith registers it there, 7831) though
    its rule lives conceptually in FACTOR.
  * eq_pmuv is created only if closure.rmuv & imuv; pmuv is a Var (bnd 0.001,None)
    under that switch, else a Param -- the block handles the Var case.

CARRY (Blocker C, Task 5): after registering all blocks the composer must run
``apply_production_scaling``'s recompute of gd_share/ge_share/gw_share (they drift
from the params.shares seed the blocks read). On gtap7_3x3 gd/ge are inert
(omegax=inf) and gw enters only unit-4 bodies.
"""

from equilibria.blocks.gtap.production_supply import ProductionSupplyBlock
from equilibria.blocks.gtap.trade_cet import TradeCETBlock

# Dependency order (leaf first). Extended as units 3-7 land.
GTAP_BLOCK_ORDER = [
    TradeCETBlock,
    ProductionSupplyBlock,
]

__all__ = [
    "GTAP_BLOCK_ORDER",
    "ProductionSupplyBlock",
    "TradeCETBlock",
]
