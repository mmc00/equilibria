"""GTAP symbolic Block units (F3 Task 4). WIP.

Dependency order (leaf → closure). ``GTAP_BLOCK_ORDER`` is the order the composer
(Task 5) registers the blocks in; shared vars dedup by name (first registration
wins), so a var owned by an earlier block is skipped in a later one.

============================================================================
COMPOSER CHECKLIST (Task 5) — the ONE authoritative list of what the composer
must do AFTER registering the blocks, mirroring the monolith's build_model
ordering (_add_variables -> _add_parameters -> apply_production_scaling ->
_align_xi_xaa_post_scaling -> _add_equations -> ifSUB post-block). The blocks
DEFINE all equations in the ``if_sub=False`` comp-stat form (the Task-4 gate
oracle); everything MODE- or SCALING-dependent below is the composer's.
============================================================================

1. ifSUB (mirrors the monolith post-block 7970-8036). When ``closure.if_sub`` is
   True: DEACTIVATE the 9 defining eqs (eq_pp_rai, eq_xwmg, eq_xmgm, eq_pwmg,
   eq_pefobeq, eq_pmcifeq, eq_pmeq, eq_pfaeq, eq_pfyeq) and FIX their paired vars
   (pp_rai, xwmg, xmgm, pwmg, pefob, pmcif, pm, pfa, pfy) to the ``_m_*`` macro
   value at the built point. The blocks port the plain-var ``if_sub=False`` form
   (``_m_pp`` -> pp_rai, ``_m_pfa`` -> pfa, ``_m_pfy`` -> pfy, ...).

2. 6 globally-deactivated eqs are NEVER ported active by any block (prf_y 5585,
   eq_ps 5901, eq_pe 5962, eq_pe_route 5979, eq_xet_agg 5994, eq_xe_xw 6007).

3. eq_pfact is DEFINED in CLOSURE (the monolith registers it there, 7831) though
   its rule lives conceptually in FACTOR. It (and eq_pwfact) reference the
   SNAPSHOT Params pf0/xf0/mqfactr_bb/mqfactw_bb (monolith 7762-7827), computed
   from the model's xf/pf/xscale Var levels AFTER apply_production_scaling — NOT
   purely from self.params. The composer must snapshot pf0=pf.l, xf0=xf.l and
   compute mqfactr_bb=sum_{f,a} pf0*xf0/xscale from the SCALED model, fill those
   Params, THEN build eq_pfact/eq_pwfact. The CLOSURE block exposes the rule
   forms referencing those Params.

4. eq_pmuv is created only if ``closure.rmuv`` & ``closure.imuv``; ``pmuv`` is a
   Var (bnd 0.001,None) under that switch, else a Param. The blocks handle the
   Var case; the composer applies the switch.

5. gd_share/ge_share/gw_share recompute (Blocker C). After registration the
   composer must run ``apply_production_scaling``'s share recompute (gd/ge/gw
   drift from the params.shares seed the blocks read). On gtap7_3x3 gd/ge are
   inert (omegax=inf); gw_share (drift ~8.7) enters ARMINGTON+BILATERAL (unit 4)
   bodies — see that block's notes.

6. POST-SCALING LEVEL/BOUND SNAPSHOT (kstock/kapEnd/arent/rorc/rore and every
   price var). apply_production_scaling RE-VALUES several level vars from scaled
   quantities (e.g. kapEnd = (1-depr)*kstock + xiagg with xiagg=yi/pi, monolith
   1268) and the runtime price/level floor sweep (5298-5385) then sets each
   var's lower bound to max(1e-8, 1e-3*value) from the SCALED value. The blocks
   seed levels from the benchmark and apply the same floor formula to that seed,
   so a level var re-valued by scaling shows a ~1e-8-relative BOUND difference
   vs the oracle (measured: the 3 kapEnd cells on gtap7_3x3). This is the same
   snapshot family as pf0/xf0/mqfactr_bb (item 3) — the composer re-values and
   re-floors these vars after scaling. Blocks own the vars + the floor formula;
   the composer owns the post-scaling value. (Domain-gate exception documented
   in tests/templates/gtap/test_gtap_blocks_form.py:_DOMAIN_CARRY_VARS.)
"""

from equilibria.blocks.gtap.demand_utility import DemandUtilityBlock
from equilibria.blocks.gtap.factor import FactorBlock
from equilibria.blocks.gtap.income import IncomeBlock
from equilibria.blocks.gtap.production_supply import ProductionSupplyBlock
from equilibria.blocks.gtap.trade_armington_bilateral import ArmingtonBilateralBlock
from equilibria.blocks.gtap.trade_cet import TradeCETBlock

# Dependency order (leaf first). Extended as units 3-7 land.
GTAP_BLOCK_ORDER = [
    TradeCETBlock,
    ProductionSupplyBlock,
    FactorBlock,
    ArmingtonBilateralBlock,
    DemandUtilityBlock,
    IncomeBlock,
]

__all__ = [
    "ArmingtonBilateralBlock",
    "DemandUtilityBlock",
    "FactorBlock",
    "GTAP_BLOCK_ORDER",
    "IncomeBlock",
    "ProductionSupplyBlock",
    "TradeCETBlock",
]
