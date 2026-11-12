# One driver for every model, fed by a recipe: design

Decided with the maintainer on 2026-11-11/12. Items marked **(builder's call)**
were not decided by the maintainer and may be changed by whoever implements this.

## Context

What the driver file holds today (measured 2026-11-11, `0d02b205`):

| Content | ~Lines | Belongs in the driver? |
|---|---|---|
| Period orchestration (`_solve_multiperiod_inner`, `solve_multiperiod`) | 430 | yes |
| Per-period equation rewrites (`_recalibrate_*`, altertax only) | 600 | no: likely deleted (step 0) |
| GTAP squaring fixes (`_collapse_pft_pfteq`, `_holdfix_fnm_pf`, `_apply_gams_bounds`, `_recompute_ifsub_report_vars`, welfare tail, Fisher duplicates) | 545 | no: GTAP recipe steps or GTAP blocks |
| Single-period reference model and its snapshot (`_build_sp_reference`, `SPSnapshot`, `_replicate_*`) | 280 | no: GTAP recipe step |
| Shock methods (in steps, Johansen, Richardson) | 135 | as pluggable steps |
| Freeze and seed periods | 140 | yes |
| GC and PATH-option context managers | 140 | no: solver |

A recipe already exists, privately: `_Recipe` and `_RECIPES` in
`templates/gtap/period_prep.py`, a table of steps per `(period, mode)` run by
`PeriodPreparer.prepare`. Its steps call back into the driver file (`_drv()`).
The multi-period model is built by reflecting a single-period (SP) model: every
Var family gets a trailing `t` and every equation is copied per period
(`GTAPMultiPeriodModel` in `templates/gtap/gtap_model_multiperiod.py`; ADR-0006).
That reflection is GTAP-specific today.

PEP (`templates/pep_pyomo/`) has no periods: a scenario edits the calibrated
state and the model is rebuilt (`pep_pyomo_scenarios.py`). It uses its own
blocks (`pep_pyomo_blocks.py`), not `blocks/`. `simple_open` is a small
single-country template on the generic blocks (`blocks/production`, `trade`,
`demand`, ...) with its own GAMS parity pipeline.

Constraints:

- Parity is not negotiable: every gate that is green today stays green with the
  same numbers (GTAP `mcp`, `nlp`, `gempack`, coverage matrix, the Burfisher
  notebooks; PEP 314/314 when Ricardo moves it).
- ADR-0006 stays: blocks write equations for one period, without `t`.
- Equations are never rewritten between periods. If a coefficient must change
  between periods (a future recursive dynamic model), that is a block's own
  update rule run as a step, never an equation rebuilt by the driver.

## Approach

### Three responsibilities

- **Blocks: the economics.** Equations, variables, parameters, and what each
  block needs from others (`Block.uses`, ADR-0005). Nobody else edits a block's
  equations.
- **Recipe: the what.** Per model: its periods, the kind of each period, the
  blocks it composes, the closure per period, and the steps per kind of period.
  It solves nothing.
- **Driver: the how.** One for every model. For each period of the recipe: run
  its "before" steps, solve, run its "after" steps, freeze. It imports nothing
  from any template.

```
RECIPE (per model)            DRIVER (one)              BLOCKS
templates/<model>/recipe.py   core/driver.py            blocks/...
periods + kinds          ->   for each period:          equations written once,
blocks                        1. steps before           for one period (ADR-0006)
closure per period            2. solve          <--     
steps per kind                3. steps after
                              4. freeze
```

### Periods

The recipe declares its periods as an ordered list, each with a kind:

- `calibrate` (GTAP `base`), `replicate` (GTAP `check`), `shock`
  (GTAP `shock`, PEP `sim1`), `step` (one year of a recursive dynamic model).

Examples: GTAP comparative statics `[base:calibrate, check:replicate,
shock:shock]`; GTAP recursive dynamic `[base:calibrate, 2014:step, ...,
2030:step]`; PEP `[base:calibrate, sim1:shock]`. There is one way to have
periods: the `t` index added by the reflection. A model is never rebuilt per
period.

The reflection that adds `t` moves from `GTAPMultiPeriodModel` to the core so any
recipe can use it **(builder's call: how much of it generalizes; GTAP-only
pieces stay in a GTAP subclass)**. Every model goes through the reflection,
including PEP once Ricardo gives it periods: there is no "no periods" path.
Every Var family gets the trailing `t`, as today (ADR-0006).

### Block order

By default the order comes from the dependencies each block declares
(`Block.uses`). A recipe may also pin an explicit order (for example, GAMS's
calibration order). If the pinned order contradicts the declared dependencies,
building fails with an error naming the block and what it is missing. A cycle
in the declared dependencies is also an error. `Block.uses` is already checked
for consistency by `add_block` (#139); a dependency a block forgot to declare is
a bug in that block, not something the ordering guesses.

Order matters for building and calibrating, not for solving: the system of a
period is solved simultaneously.

### Steps

A step is a small callable with one job (seed from the prior period, freeze
inactive periods, apply the shock, fix the numeraire, ...). Generic steps live
in the core; model steps live with their model (`templates/gtap/steps.py`).
GTAP's squaring fixes that survive step 0 become GTAP steps. The single-period
reference model becomes one GTAP step. How the shock is solved (one go, in
steps, Johansen) is a pluggable step. The GC and PATH-option context managers
move to `solver/`.

The existing `_RECIPES` table is the starting point for `GTAPRecipe`; its
`mode` (`gtap` / `altertax`) becomes a recipe parameter or two recipes
**(builder's call)**. Its steps are rewritten as steps of the new contract, not
loaded as they are: today they reach into the driver file through `_drv()`.

**Step contract (builder's call, proposed):** a step is
`step(ctx) -> None`, where `ctx` carries the model, the parameters, the recipe,
the current period and the prior one (as `_Ctx` in `period_prep.py` does
today). A step mutates the model in place and returns nothing; a step that
needs to report something logs it. Freezing and seeding are steps like any
other; the driver only guarantees the order "steps before, solve, steps after,
freeze".

**Closure (builder's call, proposed):** the closure stays the model's own
object (`GTAPClosureConfig` for GTAP, with `@overwrite` per ADR-0004). The
recipe names which closure each period uses and a model step applies it after
the "before" steps and right before the solve. The driver treats the closure as
opaque and never reads its fields.

### Where things go

- `src/equilibria/core/recipe.py`: the abstract `Recipe` (periods with kinds,
  blocks, optional pinned order, closure per period, steps per kind).
- `src/equilibria/core/driver.py`: the generic driver.
- `src/equilibria/templates/gtap/recipe.py` and `steps.py`.
- `src/equilibria/templates/simple_open/recipe.py` **(builder's call: the exact
  file layout of simple_open)**.
- Not in `blocks/`: blocks are the pieces, the recipe is the instructions.

This follows the R-ARCH rule: what serves any model goes in the core ("build"),
what serves one model goes in `templates/<model>`.

### Migration: a new driver next to the old one

1. **Step 0, cleanup**, on the old driver: one PR per deletion, failing test
   first, measured against GAMS on every gate that exercises it (for the
   altertax recalibrations: the altertax rows of Table A1, ifSUB 0 and 1, NLP
   and MCP). A deletion is accepted when every cell stays equal or moves
   closer to GAMS, with the gates' existing tolerances; if any cell moves away,
   the piece stays and the reason is written next to it. Precedent: removing
   the io/af recalibration raised the shock gate 98.21 % -> 99.93 %, and
   removing alphad/alpham (#222) took 15x10 altertax 99.45 % -> 100 %.
2. Build `Recipe`, the core driver and the generic reflection, with
   `SimpleOpenRecipe` as their first user.
3. Build `GTAPRecipe` and its steps from `_RECIPES` and the surviving pieces of
   the old driver.
4. Run every GTAP gate through both drivers and compare (same codes, same
   values within the gates' tolerances).
5. Switch `solve_multiperiod` to the new driver in one PR, and delete the old
   driver and `period_prep.py` in the next.

The public entry point `solve_multiperiod(m, params, closure, ...)` keeps its
signature through the switch **(builder's call: whether it becomes a thin
wrapper over `GTAPRecipe` + the core driver)**.

### Out of scope

- PEP: periods, its recipe and moving its scenarios to the shock period are
  Ricardo's (R-PEP, after E1).
- The end-of-solve summary by default (`R-PROGRESS`): it hooks into the new
  driver's "after solve" once this lands.
- Recursive dynamic GTAP (`R-DYN`): this change only makes it a recipe with
  more periods; the dynamics block is its own change.

## Alternatives considered

- **A driver with two modes** (one model indexed by `t`, or one model per
  period, for PEP). Rejected: it gives the driver more responsibilities, the
  opposite of the goal.
- **Change the existing driver little by little.** Rejected by the maintainer:
  a new driver next to the old one, switched in one move once it matches.
- **Keep per-period coefficients as `t`-indexed parameters inside blocks.**
  Rejected: contradicts ADR-0006, and step 0 is expected to remove the need.
- **Ricardo reviews the recipe before the GTAP side is built.** Rejected by the
  maintainer: go ahead; `simple_open` as a second model lowers the risk that the
  recipe only fits GTAP.

## Verification

- Every GTAP parity gate and the Burfisher notebooks give the same numbers
  through the new driver as through the old one, before the switch.
- `simple_open` keeps its GAMS parity through the new driver.
- The core driver and `Recipe` import nothing from `templates/` (a layering test,
  like `tests/core/test_layering.py`).
- The core driver stays small: a test or check that fails if it grows past
  ~300 lines **(builder's call: whether to enforce it)**.
- Block order: a test with a pinned order that contradicts `Block.uses` fails
  with the expected error.
