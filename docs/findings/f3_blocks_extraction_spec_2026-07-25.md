# F3 — GTAP Blocks Extraction — Design Spec

**Date:** 2026-07-25
**Status:** Design (approved for planning)
**Roadmap:** F3 (modularization to blocks), prerequisite of F3.5 (no-check variant), F5 (GEMPACK), F7 (GTAP6), F9 (PEP blocks).

## Goal

Extract the GTAP model monolith `src/equilibria/templates/gtap/gtap_model_equations.py`
(8076 lines) into `src/equilibria/blocks/gtap/` — one block at a time, each a
declarative `Block` subclass with its variables and equations — maintaining
**0-diff parity vs GAMS** at every step. This makes GTAP composable (needed for
F3.5's calibrated-base variant, F7's GTAP6 on shared blocks, F8's block swapping).

## Background — what the dependency map found

An Explore pass over the monolith (2026-07-25) established:

1. **All 105 `Var` declarations live in ONE central block (lines 3509–5106), BEFORE
   any equation block.** The banner-delimited blocks (5574+) contain only
   `Constraint` definitions. Coupling is measured by which var each block's
   equations *determine* (LHS/complementary var) vs which it *reads* from others.
2. **~10 banner blocks collapse to 7 extraction units** because of **5 circular
   dependencies** — those pairs must be extracted as one bundle:
   - PRODUCTION ↔ SUPPLY (irreducible CES make/supply nest)
   - TRADE_ARMINGTON ↔ BILATERAL_TRADE (import-sourcing loop: xw ↔ pwmg/xda/xmt)
   - FACTOR ↔ INCOME (FACTOR reads pabs; INCOME reads kstock/xf)
   - UTILITY_SAVINGS ↔ INCOME (util reads regy/rsav/yg; income reads phip/pi/savf/uh)
   - DEMAND ↔ UTILITY_SAVINGS (DEMAND is a 4-eq shell fused to UTILITY)
3. **Several cross-block edges come from DEACTIVATED legacy constraints**
   (`.deactivate()`) that are not in the active MCP — these must be filtered when
   measuring dependencies (most notably TRADE_CET's pe/xw edges are all legacy).

## Design decisions (all user-approved 2026-07-25)

### Form: declarative `Block` subclasses (option B), with vars distributed to their owning block
Each extracted block is a subclass of `equilibria.blocks.base.Block`
(pydantic, `VariableSpec`/`EquationSpec`, `setup`/`get_calibration_phases`/
`_extract_calibration`). Variables are **moved to their owning block** (px/pva/nd →
ProductionBlock, xw/pe → BilateralBlock, …), not left in a shared central block.
This is the fully-modular form and gives F3.5 the calibration hooks natively. The
risk (moving 105 var declarations with their domains/bounds — where a mis-set
`Reals` vs `NonNegativeReals` fabricates a spurious complementarity corner, cf.
project_gtap7_5x5_ifsub1_fabricated_corner) is controlled by the per-block gate below.

### Strategy: one block at a time, green gate between each, closure last
Extract in dependency order (leaf → coupled). Each block is a small, bisectable PR.
If parity breaks, the failing block and the failing gate layer are immediately known.

### Extraction order (from the dependency map)
1. **TRADE_CET** (5903–6008) — the safe leaf: ZERO active outbound edges (all pe/xw
   uses are in deactivated legacy constraints); reads only xs/ps/pd/pet + params.
2. **PRODUCTION + SUPPLY** (5574–5902) — one merged module (PROD↔SUPPLY cycle).
3. **FACTOR** (6633–6919) — single clean inbound from PROD; one back-edge (pabs) to
   INCOME handled via a forward-declared interface.
4. **ARMINGTON + BILATERAL** (6009–6632) — one merged trade module (sourcing cycle).
5. **DEMAND + UTILITY_SAVINGS** (6920–7314) — one merged household module.
6. **INCOME** (7315–7740) — highest fan-out (aggregates 6 blocks); near-last.
7. **CLOSURE / EQUILIBRIUM** (7741–8079) — last (pre-decided): market clearing,
   numeraire, Walras, pfact/pwfact — the most delicate (MCP pairing, NLP numeraire,
   multi-period holdfix, per-mode PATH options).

### Per-block fidelity gate (all four, cheap → expensive; ALL must be green to accept)
1. **Equation-form diff** (`scripts/gtap/diff_equation_form.py`, cascade tool 5):
   the expanded Pyomo expression of the new declarative block == the monolith's,
   cell-by-cell. Catches any form drift BEFORE a solve. Instant.
2. **Var domain+bounds diff**: each moved Var keeps EXACTLY its `domain`
   (Reals/NonNegativeReals) and bounds — a direct var-by-var check, no solver.
   This is where the spurious-corner risk lives (xw/xet → Reals).
3. **Canary (gtap7_3x3)**: a single small dataset solved first — if it breaks, no
   need to spend the full sweep. Seconds.
4. **Full NLP-vs-NLP + MCP-vs-MCP sweep** (`scripts/gtap/run_parity_gates.py`) vs
   native GAMS. The roadmap's real gate. ~12min. Refreshes the parity stamp.

## Architecture

- `src/equilibria/blocks/gtap/` — new package, one module per extraction unit.
- Each module: a `Block` subclass declaring its `VariableSpec`s + `EquationSpec`s,
  with a `setup(model, params, ...)` that builds them on the Pyomo model.
- A `gtap_block_model.py` (or extension of `gtap_model_multiperiod.py`) composes the
  blocks via the `BlockRegistry`, replacing the monolith's `build_equations_*` calls
  block by block as each is extracted. Until a block is extracted, the monolith path
  is still used for it (incremental cutover — the model always builds and solves).
- The monolith `gtap_model_equations.py` shrinks block by block; it is NOT deleted
  until the last block (closure) is extracted and green.

## Acceptance gates (parity floors, per the roadmap convention)

- **Per block:** all four gate layers green (form-diff clean, var domain/bounds
  identical, canary 3x3 code=1 + measured match unchanged, full sweep green + stamp).
- **Global (F3 done):** the full monolith is extracted; `gtap_model_equations.py`
  no longer builds equations (or is a thin shim); the coverage matrix is unchanged
  (0-diff) on every dataset × kind; parity stamp fresh.
- **Non-regression:** at no intermediate commit may the matrix drop below its
  current measured values on any cell.

## Non-goals / YAGNI

- No behavior change — F3 is a pure refactor. Any result change is a bug, not F3.
- No F3.5 work here (the calibrated-base variant is a separate phase; F3 only
  ensures the block form gives it the calibration hooks).
- No new solver/closure logic — the closure block moves as-is, last.
- No GTAP6/PEP block work (F7/F9 reuse these blocks later).

## Risks

- **Var domain/bounds drift** on the moved 105 declarations → spurious corners.
  Mitigation: gate layer 2 (var domain+bounds diff) is mandatory per block.
- **Circular-dependency bundles** can't be split → extracted as merged modules
  (5 bundles identified). Attempting to split them would break the build.
- **Legacy deactivated constraints** create phantom edges → filtered in the
  dependency measurement; the form-diff gate compares only active constraints.
- **Closure block** is the highest-risk (solver/period/mode-parameterized) →
  extracted LAST, with both gates, per the roadmap.
