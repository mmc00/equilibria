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

### Three concepts, plus the pieces

The words follow `docs/glossary.md`; the glossary is updated with the new
meanings. The picture the maintainer chose: the blocks are the pieces of a
rocket, the template says which pieces it has, the recipe says what we do with
the rocket, the driver is the pilot, and the exercise is one mission.

| Concept | Rocket | GAMS analogue | Responsibility |
|---|---|---|---|
| **Block** | a piece | the equations of one part | The economics: parameters, variables, equations for one period (ADR-0006), and what it reads from other blocks (`Block.uses`, ADR-0005). Nobody else edits a block's equations. |
| **Template** (the plan of the set) | which pieces and how they fit | `model gtap / ... /` | Which blocks the model has, the order they are built in, the default closure, its default recipe, and **its data**: it is created with them (`GTAP(params)`), so nothing else receives the parameters separately. |
| **Recipe** (the flight plan) | what we do with the rocket | the `loop(tsim, ...)` with its assignments | The periods and their kinds; per kind: the steps before and after the solve, the solve strategy and the solver. It builds and solves nothing itself. |
| **Driver** (the pilot) | flies the plan | the `solve` inside the loop | For each period of the recipe: steps before, solve with the period's strategy and solver, steps after, freeze. It knows no model, no shock and no closure, and keeps no state between runs. |
| **Exercise** (the mission) | this rocket, this plan, this shock | one `.gms` run | Template + recipe + shock + `@overwrite`. `Exercise(...).run()` is the public way to run one. |

```python
gtap = GTAP(params)                                  # the template carries its data
ex = Exercise(template=gtap, recipe=GTAPStatic(),
              shock={"lambdava": {("USA", "SER"): 10}},
              overwrite=[my_unemployment_block])     # optional, this exercise only
results = ex.run()                                   # {period: {"code", "residual"}}, as today
```

### Template

- Extends the existing `ModelTemplate` (`templates/base.py`, used today only by
  `simple_open`); GTAP moves onto it. No new base class.
- Declares its blocks. The build order comes by default from what each block
  declares in `Block.uses`; a template may pin an explicit order (for example
  GAMS's calibration order). A pinned order that contradicts the declared
  dependencies, or a cycle, fails the build with an error naming the block and
  what it is missing. `Block.uses` is already checked by `add_block` (#139); a
  dependency a block forgot to declare is a bug in that block.
- Order matters for building and calibrating, not for solving: each period's
  system is solved simultaneously.
- Brings a default recipe (GTAP: comparative statics, `GTAPStatic`). Other
  recipes (recursive dynamic, Johansen) are chosen explicitly. One template can
  run under several recipes.
- Building = template + the recipe's period list, so the reflection can add `t`.
  The reflection moves from `GTAPMultiPeriodModel` to the core **(builder's
  call: how much of it generalizes; GTAP-only pieces stay in a GTAP subclass)**.
  Every model goes through it, including PEP once Ricardo gives it periods:
  there is no "no periods" path.
- Calibration is unchanged in this change (`R-CALIB` is its own change): GTAP
  keeps calibrating from `blocks/gtap/_derived_params.py`.

### Recipe

- **Periods** are an ordered list, each with a kind. The core ships four kinds:
  `calibrate` (GTAP `base`), `replicate` (GTAP `check`), `shock` (GTAP `shock`,
  PEP `sim1`) and `step` (one year of a recursive dynamic model). A recipe uses
  them as they are; it may add a kind of its own only when none fits. The driver
  uses the kind only to look up the steps; it behaves the same for every kind.
- Examples: GTAP comparative statics `[base:calibrate, check:replicate,
  shock:shock]`; GTAP recursive dynamic `[base:calibrate, 2014:step, ...,
  2030:step]`; PEP `[base:calibrate, sim1:shock]`.
- **Steps** per kind: small functions with one job, in order (see below).
- **Solve strategy** per kind, a slot of its own, not a step, because it wraps
  the solve: in one go (default), in steps (Gragg-like), Johansen.
- **Solver and problem type** per kind (default PATH, MCP; also IPOPT/NLP and
  Newton). The driver only calls what the recipe names.
- The existing `_RECIPES` table is the starting point for `GTAPStatic`; its
  `mode` (`gtap` / `altertax`) becomes a recipe parameter or two recipes
  **(builder's call)**. Its steps are rewritten to the new contract, not loaded
  as they are: today they reach into the driver file through `_drv()`.

### Steps

- The contract lives in the core: `Step` is a protocol, `step(ctx) -> None`,
  where `ctx` carries the model, the template (and through it the data), the
  recipe, the current period and the prior one (as `_Ctx` in `period_prep.py`
  does today). A step mutates the model in place and returns nothing; a step
  that needs to report something logs it.
- Generic steps live in `core/steps.py` (freeze the other periods, seed from the
  prior period). Model steps live with their model (`templates/gtap/steps.py`:
  GTAP's squaring fixes that survive step 0, the single-period reference model,
  `regy`, the shock numeraire). The driver defines no step.
- The closure stays the model's own object (`GTAPClosureConfig` for GTAP, with
  `@overwrite` per ADR-0004). A model step applies it after the "before" steps
  and right before the solve. The driver never reads its fields.
- The shock stays in the `ShockBlock` (ADR-0003). The recipe only names which
  period receives it; the exercise says what it is.
- The GC and PATH-option context managers move to `solver/`.

### Exercise

- Carries its shock and its `@overwrite` blocks. Nothing is registered globally,
  so nothing leaks from one exercise to the next: this is the structural cure of
  #214 (the quick fix of #214 lands first, on the old driver).
- Each exercise gets **its own copy** of the built model. The model is built
  once and kept in the existing content-keyed model cache (5.5x on 20x41); each
  exercise receives a fresh copy from it, never a shared object.
- Returns what `solve_multiperiod` returns today, per period `{code, residual}`,
  so tests and notebooks do not change. The end-of-solve summary (`R-PROGRESS`)
  hooks in later as an "after solve" step.

### Where things go

- `src/equilibria/core/`: `recipe.py` (`Recipe`, period kinds), `steps.py`
  (`Step` and the generic steps), `driver.py`, `exercise.py`, the reflection.
- `src/equilibria/templates/base.py`: `ModelTemplate`, extended.
- `src/equilibria/templates/gtap/`: the `GTAP` template, `GTAPStatic`,
  `steps.py`.
- `src/equilibria/templates/simple_open/`: its template and recipe
  **(builder's call: exact file layout)**.
- Not in `blocks/`. This follows the R-ARCH rule: what serves any model goes in
  the core ("build"), what serves one model goes in `templates/<model>`.

### Documentation

Decided with the maintainer on mockups (the throwaway page is
https://claude.ai/artifact/GP5WJ63qZtFymNAjekZGKS; the references are in
dev-tools `equilibria-tools/plans/research/docs-lego-references/`).

- **Style:** 3D bricks seen from the front with studs on top, as SVG that works
  in light and dark themes. Drawn from scratch; references are inspiration only
  and never enter the repo. Replaces today's flat tower of rectangles in
  `docs/site/guide/architecture.md`.
- **Parts table** at the top of the architecture page, one fixed drawing per
  concept, used on every page:

  | Piece | Concept |
  |---|---|
  | brick | block |
  | plate | data (SAM, parameters) |
  | smooth tile on top (no studs: nothing stacks on it) | closure (`ClosureBlock`) |
  | Technic Shock Absorber 9L, blue (part 79717c01), pinned into Technic bricks with holes | shock (`ShockBlock`) |
  | the set's instruction sheet | template |
  | flight plan (`base → check → shock`) | recipe |
  | pilot minifig | driver |
  | rocket | exercise |

- **Colors in two levels:** generic blocks in one grey (they only say the role:
  households, production, trade...). In a built model, the color is the
  functional form: CES orange, CET teal, Leontief brown, Cobb-Douglas yellow,
  LES blue, CDE pink. Swapping households CDE for LES is drawn as taking one
  piece out and putting another in, same place, same studs.
- **Shock** is the model's suspension: the shock block is a purple Technic brick
  with holes, two 9L struts pinned between it and a grey Technic chassis that
  carries the economy, the closure tile on top. Compressed, a strut shows the
  size of the shock (+10 %). In the rocket, the struts are its landing legs
  **(builder's call, proposed by Claude)**.
- **Pages** (English, as the whole repo): the architecture page rewritten around
  the concepts; a new guide "Run an exercise" with `Exercise(...).run()` step by
  step; the GTAP quickstart moved to `Exercise`. All told with the rocket. "How
  to write a new template" waits for a second real template (PEP, Ricardo).
- **When:** the drawings are designed in step 3 (they also guide the build); the
  pages are published in step 7, when the code exists, because the CI runs the
  docs examples (#183).

### Migration: a new driver next to the old one

1. **Fix #214** on the old driver (exercises leaking `@overwrite` hooks).
2. **Step 0, cleanup**, on the old driver: one PR per deletion, failing test
   first, measured against GAMS on every gate that exercises it (for the
   altertax recalibrations: the altertax rows of Table A1, ifSUB 0 and 1, NLP
   and MCP). A deletion is accepted when every cell stays equal or moves closer
   to GAMS, with the gates' existing tolerances; if any cell moves away, the
   piece stays and the reason is written next to it. Precedent: removing the
   io/af recalibration raised the shock gate 98.21 % -> 99.93 %, and removing
   alphad/alpham (#222) took 15x10 altertax 99.45 % -> 100 %. First candidates:
   `_recalibrate_and_ava`, `_gx_ax`, `_alphaa_gov_inv`, `_alphaa_hhd` (GAMS
   calibrates every share once, in `loop(tsim$t0)`).
3. **New ADR** superseding ADR-0002, the glossary updated (Template, Recipe,
   Driver, Exercise), and the drawings designed.
4. **Core** (`Recipe`, `Step`, driver, `Exercise`, reflection) with
   `simple_open` as its first user.
5. **GTAP**: the `GTAP` template, `GTAPStatic` and its steps, from `_RECIPES` and
   the surviving pieces of the old driver.
6. **Compare** every GTAP gate through both drivers (same codes, same values
   within the gates' tolerances).
7. **Switch** `solve_multiperiod` to the new driver in one PR (it keeps its
   signature during the transition **(builder's call: whether it becomes a thin
   wrapper)**), regenerate the Burfisher notebooks with their generator so they
   use `Exercise(...).run()` (same numbers against GEMPACK), and publish the
   docs pages.
8. **Delete** the old driver and `period_prep.py`.

### Out of scope

- PEP: periods, its recipe and moving its scenarios to the shock period are
  Ricardo's (`R-PEP`, stage E3, after E1).
- Calibration per block (`R-CALIB`).
- The end-of-solve summary by default (`R-PROGRESS`): an "after solve" step once
  this lands.
- Recursive dynamic GTAP (`R-DYN`): this change only makes it another recipe;
  the dynamics block is its own change.

## Alternatives considered

- **A driver with two modes** (one model indexed by `t`, or one model per
  period, for PEP). Rejected: it gives the driver more responsibilities, the
  opposite of the goal.
- **Change the existing driver little by little.** Rejected by the maintainer:
  a new driver next to the old one, switched in one move once it matches.
- **Keep per-period coefficients as `t`-indexed parameters inside blocks.**
  Rejected: contradicts ADR-0006, and step 0 is expected to remove the need.
- **One object that both builds and runs** (the recipe also listing the
  blocks). Rejected: one template runs under several recipes (static, dynamic,
  Johansen), so building and running are two responsibilities.
- **Period kinds defined by each recipe only**, or no kinds (steps per period
  name). Rejected: four kinds in the core make every recipe read the same; the
  dynamic model would repeat its steps per year without kinds.
- **Ricardo reviews the recipe before the GTAP side is built.** Rejected by the
  maintainer: go ahead; `simple_open` as a second model lowers the risk that the
  recipe only fits GTAP.
- **Docs:** one color family in shades (hides the families), one color per role
  (says nothing about the form), a Technic pin or a padlock for the closure, and
  a lever, an engine, a control panel, a dial, a pneumatic cylinder or a spring
  shooter for the shock. All shown on the mockup and rejected by the maintainer.

## Verification

Approved by the maintainer on 2026-11-12 with one explicit condition: **every
check and every Burfisher exercise gives the same result** as on `main` before
the change. That is the gate for each step of the migration, not only for the
switch.

- Every check: the full test suite (CI and the local-only parity tests, with
  `gdxdump` on the PATH so none is skipped silently), compared by test name and
  skip reason, not only by count.
- Every Burfisher exercise: each notebook and each `run_burfisher.py`
  experiment against GEMPACK, cell by cell, with the same numbers as on `main`.
- Every GTAP parity gate and the Burfisher notebooks give the same numbers
  through the new driver as through the old one, before the switch.
- `simple_open` keeps its GAMS parity through the new driver.
- The core (`Recipe`, `Step`, driver, `Exercise`) imports nothing from
  `templates/` (a layering test, like `tests/core/test_layering.py`).
- The core driver stays small: a test or check that fails if it grows past
  ~300 lines **(builder's call: whether to enforce it)**.
- Block order: a pinned order that contradicts `Block.uses`, and a cycle, fail
  with the expected error.
- Two exercises in one process, the first with an `@overwrite`, the second
  without: the second's closure is the default one (the #214 scenario).
- Two exercises from one cached model: changing the first's model leaves the
  second's untouched.
- The docs examples run in the CI (#183), and the drawings render in light and
  dark themes.
