"""GTAP symbolic Block units (F3 Task 4). All 7 units migrated.

Dependency order (leaf → closure). ``GTAP_BLOCK_ORDER`` is the order the composer
(Task 5) registers the blocks in; shared vars dedup by name (first registration
wins), so a var owned by an earlier block is skipped in a later one.

The 7 units: TradeCET, ProductionSupply, Factor, ArmingtonBilateral,
DemandUtility, Income, Closure. Each is form+domain+index-set clean vs the
gtap7_3x3 ``if_sub=False`` comp-stat oracle (tests/templates/gtap/
test_gtap_blocks_form.py).

Several blocks take a ``residual_region`` constructor field (DemandUtility,
Income, Closure) — the composer must pass the SAME residual region the model is
built with (mirrors GTAPModelEquations(residual_region=...)); it drives the
eq_savf / eq_yi / eq_walras Skip masks and the savf_bar rebalance. Default
``NAmerica`` (the monolith default); the gtap7_3x3 comp-stat build uses ``ROW``.

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
   (``_m_pp`` -> pp_rai, ``_m_pfa`` -> pfa, ``_m_pfy`` -> pfy, ...). Also apply
   pa.fix(1.0) on the no-demand cells (monolith 3595-3602 — value/flag, not bounds).

2. 6 globally-deactivated eqs are NEVER ported active by any block (prf_y 5585,
   eq_ps 5901, eq_pe 5962, eq_pe_route 5979, eq_xet_agg 5994, eq_xe_xw 6007).

3. eq_pfact is DEFINED in CLOSURE (the monolith registers it there, 7831) though
   its rule lives conceptually in FACTOR. It (and eq_pwfact) reference the
   SNAPSHOT Params pf0/xf0/mqfactr_bb/mqfactw_bb (monolith 7762-7827), computed
   from the model's xf/pf/xscale Var levels AFTER apply_production_scaling — NOT
   purely from self.params. On the base build pf0=pf.l matches the block's
   _pf_init exactly, but xf0=xf.l is the SCALED xf (benchmark / xscale), so the
   folded eq_pfact/eq_pwfact coefficient differs from the block's benchmark seed
   by the xscale factor (up to ~10x on Mnfcs/Svces). The composer MUST snapshot
   pf0=pf.l, xf0=xf.l and recompute mqfactr_bb=sum_{f,a} pf0*xf0/xscale (and
   mqfactw_bb globally) from the SCALED model, OVERWRITE the CLOSURE block's
   pf0/xf0/mqfactr_bb/mqfactw_bb Params, THEN the eq_pfact/eq_pwfact expressions
   are correct. (Form-gate exception: _STRUCT_CARRY_EQS — the block's skeleton is
   byte-identical; only the composer-owned magnitude is exempt.)

4. eq_pmuv + pmuv Var-or-Param switch (monolith 5096-5118, 7870-7942). Only when
   ``closure.rmuv`` & ``closure.imuv`` are BOTH non-empty is ``pmuv`` a Var (bnd
   0.001,None) and ``eq_pmuv`` registered (against composer-snapshot pefob0/xw0/
   mqmuv_bb Params). Otherwise (the gate oracle) ``pmuv`` is a mutable Param
   frozen at 1.0 and eq_pmuv is NOT created. The CLOSURE block declares the
   Param form (matching the gate oracle); the composer flips pmuv to a Var and
   registers eq_pmuv under the switch.

5. SHARE / ALPHA recompute (Blocker C) after registration via
   apply_production_scaling — the blocks read the params.shares / benchmark SEED:
     - gd_share/ge_share/gw_share (CET/Armington-bilateral shares; on gtap7_3x3
       gd/ge inert (omegax=inf), gw inert (omegaw=inf); gw drift ~8.7 bites
       finite-omega datasets).
     - top-Armington alphad/alpham (eq_paa/eq_xda/eq_xma — computed from
       post-scaling xaa/xda/xma/pdp/pmp Var levels, monolith 6122-6229; last-bit
       noise on the base build).
     - alphaa(r,i,gov|inv) i.e. g_share/i_share (monolith recomputes at the
       betaCal solution when t0_snapshot is set, 2660-2725; iterloop each period).
   All read the seed in-block; the composer runs the recompute post-registration.

6. POST-SCALING LEVEL/BOUND SNAPSHOT (kstock/kapEnd/arent/rorc/rore, xiagg,
   xigbl, rorg and every price var). apply_production_scaling RE-VALUES several
   level vars from scaled quantities (e.g. kapEnd = (1-depr)*kstock + xiagg with
   xiagg=yi/pi, monolith 1268; rorg from the rate-of-return chain; xigbl from
   _align_xi_xaa_post_scaling) and the runtime price/level floor sweep
   (5298-5385) then sets each var's lb = max(1e-8, 1e-3*SCALED value). The blocks
   seed levels from the benchmark and apply the same floor formula to that seed,
   so a re-valued var shows a BOUND difference vs the oracle (measured on
   gtap7_3x3: kapEnd (3 cells, ~1e-8), pm/pmcif/pefob (price re-value), xigbl
   (~2e-8), rorg (1e-3 -> ~0.0039)). Same snapshot family as pf0/xf0 (item 3):
   the composer re-values + re-floors after scaling. Blocks own the vars + the
   floor formula; the composer owns the post-scaling value. (Domain-gate
   exception: test_gtap_blocks_form.py:_DOMAIN_CARRY_VARS.)

7. DEMAND/INCOME CALIBRATION-LOOP composer carries (from _derived_params.
   demand_income_params, a verbatim transcription of the monolith 2258-2785).
   The base-build pass is byte-exact to the oracle (validated 0-diff on all 27
   Params + zcons init). TWO conditional overrides are OMITTED from the block
   transcription because they read model/dump state the block lacks — the
   composer applies them (both base-None on the gate oracle):
     (a) gams_calibration_dump alphaa(r,i,'inv') override (monolith 2638-2653) —
         when a GAMS cal_dump is supplied, override i_share from alphaa.
     (b) t0_snapshot recompute (monolith 2660-2760): i_share/g_share recomputed
         at converged BASELINE values (via t0.xaa/xg/pa), and betap/betag/betas
         inherited from the base snapshot. Only for counterfactual/shock periods.
   FISHER base snapshots (INCOME eq_pabs/eq_gdpmp/eq_rgdpmp, monolith 7519-7616):
   base_pa/base_xaa/base_pefob/base_pmcif/base_xw/base_pabs/base_rgdpmp are read
   from t0_snapshot (or the model itself on base). The block seeds them from the
   benchmark (base_pa=1, base_pabs=1, base_xaa=purchaser value); on a SHOCK build
   the composer MUST pass t0_snapshot so these reference the solved base period
   (matters for eq_pabs/eq_rgdpmp under shocks). On the base build eq_rgdpmp is
   rgdpmp==gdpmp (is_counterfactual False) — pass is_counterfactual/t0_snapshot to
   the INCOME block for the Fisher chain-volume form on shock periods. eq_pabs's
   base_xaa also drifts ~1e-7 on the 6 inv cells (_align_xi_xaa_post_scaling) —
   coefficient-only carry, composer re-snapshots (test _FORM_CARRY_EQS).
"""

from equilibria.blocks.gtap.closure import ClosureBlock
from equilibria.blocks.gtap.demand_utility import DemandUtilityBlock
from equilibria.blocks.gtap.factor import FactorBlock
from equilibria.blocks.gtap.income import IncomeBlock
from equilibria.blocks.gtap.production_supply import ProductionSupplyBlock
from equilibria.blocks.gtap.trade_armington_bilateral import ArmingtonBilateralBlock
from equilibria.blocks.gtap.trade_cet import TradeCETBlock

# Dependency order (leaf first). All 7 units migrated.
GTAP_BLOCK_ORDER = [
    TradeCETBlock,
    ProductionSupplyBlock,
    FactorBlock,
    ArmingtonBilateralBlock,
    DemandUtilityBlock,
    IncomeBlock,
    ClosureBlock,
]

__all__ = [
    "ArmingtonBilateralBlock",
    "ClosureBlock",
    "DemandUtilityBlock",
    "FactorBlock",
    "GTAP_BLOCK_ORDER",
    "IncomeBlock",
    "ProductionSupplyBlock",
    "TradeCETBlock",
]
