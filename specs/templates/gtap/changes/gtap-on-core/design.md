# GTAP on the core driver, one recipe for Pyomo and JAX: design

Status: approved, 2026-11-13. Continues `specs/templates/gtap/changes/driver-recipe/design.md`
(ADR-0011) after step 5 (#251). It covers steps 6 to 9 of that migration in one
design, with a change decided by the maintainer: the same GTAP recipe flies on both
backends (Pyomo and family/JAX), and every old driver is deleted at the end.

## Context

After step 5, `equilibria.core` has `Recipe`, `Step`, the driver (`fly`) and
`Exercise`. Steps and the driver touch a model only through `PeriodModel`
(`core/period_model.py`); `backends/pyomo_periods.py` implements it for Pyomo.
`simple_open` runs on them. GTAP does not: it runs on two separate stacks.

| | Pyomo route (default) | family route (opt-in) |
|---|---|---|
| Driver | the old Pyomo multiperiod driver (1,845 lines) | the old family driver (1,090 lines), `family/altertax/run.py` |
| Period recipe | the old Pyomo period recipe (its `_RECIPES` table) | the old family period recipe (its own `_RECIPES`) |
| Model | reflected Pyomo `ConcreteModel` | `MPArrays` (numpy masks) + JAX-compiled equation families |
| Solver | PATH (`run_gtap._run_path_capi_nonlinear_full`, closure applied inside) | Newton-TR (`solver/newton_tr.py`) + MUMPS |
| Covers | modes `gtap` and `altertax`, base solved or skipped, F3.5 settled seed, `solve_check`, `settle_only`, shock in steps, Johansen, `@overwrite`, welfare tail, ifSUB report vars | mode `gtap` only, base skipped, check solved, no steps, no Johansen, no `@overwrite` (refused with `NotImplementedError`); plus a settle cache and the ratio-bordered nominal-savf shock |

The same thirteen-odd preparation steps are written twice by hand, once per stack.
The 42 Burfisher notebooks call `build_block_model` + `apply_shock` +
the old Pyomo multiperiod driver; 186 files in `src`, `tests`, `scripts`, `examples` and `docs`
name the old drivers or the old Pyomo period recipe.

Measured times on the Mac (memory notes, to re-measure on `main` before starting):
gtap7_20x41 ~7 min on the Pyomo route with the model cache; ~81 s on the family route.

## Goals

- One GTAP recipe (`GTAPStatic`) whose steps are written once, against `PeriodModel`,
  and fly on Pyomo and on family/JAX.
- The recipe covers everything the Pyomo route does today (table above).
- At the end only the core driver exists: the old Pyomo multiperiod driver,
  the old Pyomo period recipe, the old family driver, the old family period recipe (and what only they
  use) are deleted, with every reference to them.
- The public way to run GTAP is `Exercise(template=GTAP(params), recipe=GTAPStatic(...),
  shock=..., overwrite=[...]).run()`.

## Non-goals

- New economics. No equation, calibration or closure changes; `R-CALIB`, `R-DYN`,
  `R-PROGRESS` and PEP (`R-PEP`) stay out.
- Making the family route support what it refuses today (altertax on family,
  `@overwrite` on family...). The shared recipe may make some of it free; a family
  run that cannot honour a setting still refuses it before any work, naming it.
- The docs pages with the Lego drawings (they wait for their own change).

## Anti-goals (a failure even if everything "works")

- Any check or Burfisher exercise giving a different result from `main`.
- gtap7_20x41 slower than today on either backend (a hard gate, confirmed by the
  maintainer: equal results do not save a slower run).
- A step that reaches into a backend (Pyomo objects, `MPArrays` fields) instead of
  `PeriodModel`: the two recipes would drift apart again.
- An exception swallowed into a different answer (`except ...: pass`): a step that
  cannot do its job raises.

## Approach

### The contract grows bulk operations

`PeriodModel` today works one cell at a time. GTAP's steps touch ~395k cells on
20x41, and the family route is fast because it works on whole arrays. The contract
gains operations over a set of cells or rows at once (names are the builder's call),
covering what the GTAP steps need:

- fix / unfix / set the level of many cells of a period (selected by variable name
  and, optionally, an index predicate), keeping exogenous cells untouched;
- set lower/upper bounds of cells (GAMS bounds, the single-period reference's bounds);
- activate / deactivate single rows or a family of rows of a period (not only a whole
  period: `_deactivate_redundant_xft`, ifSUB report rows);
- copy levels from one period to another for many cells at once (seed, copy check
  from base);
- read a table of levels (for the Fisher refresh, the welfare tail, the endowment
  check).

The per-cell operations stay. Both implementations must give the same model state for
the same calls: a contract test runs every operation on both and compares.

### Two implementations

- `backends/pyomo_periods.PyomoPeriodModel` (exists): extended with the bulk
  operations, vectorised over Pyomo components where it matters for speed.
- A family implementation over `MPArrays` (where it lives is the builder's call;
  likely `family/`), mapping each operation to the numpy masks the old family period recipe uses
  today.

### The GTAP template and recipe

- `templates/gtap/`: a `GTAP` template (on `ModelTemplate`) that carries its data
  (`GTAP(params, closure=..., ...)`), builds the single-period model from the blocks,
  and builds the period model for a backend. The GTAP-only parts of the reflection
  (Fisher cross-period rows, `tmarg`/`amgm`, shared ifSUB macro copies) live in GTAP,
  not in the core reflection.
- `GTAPStatic(mode="gtap" | "altertax", ...)`: periods `base:calibrate`,
  `check:replicate`, `shock:shock`; for each kind the steps of today's `_RECIPES` in
  the same order, rewritten as core steps over `PeriodModel`
  (`templates/gtap/steps.py`). `mode` is a parameter of one recipe, not two recipes:
  today the old Pyomo period recipe already shares one `_RECIPES` table between the
  modes and branches only inside a few steps (altertax recalibration, the altertax
  closure), so the steps keep that shape. The options of the old Pyomo multiperiod driver (`skip_base_solve`,
  `mute_welfare`, `seed_from_prior`, `solve_check`, `settle_only`, `shock_steps`,
  `shock_method`) become recipe parameters, plan choices or strategies:
  - shock in steps and Johansen are solve **strategies** of the shock kind;
  - `settle_only` is a recipe without the shock period;
  - `skip_base_solve` / F3.5 are the calibrate kind with or without a solver;
  - the welfare tail, the ifSUB report vars and the endowment-shock check are
    **after** steps.
- The closure stays `GTAPClosureConfig` (+ `@overwrite`, ADR-0004); a GTAP step
  applies it right before the solve. Today PATH receives it inside
  `_run_path_capi_nonlinear_full`; where it goes so both solvers see it is the
  builder's call, but the driver never reads it.
- The shock stays in the `ShockBlock` (ADR-0003); the default +10 % tariff when no
  shock is given stays a GTAP step of the shock kind.
- Per-mode PATH options (today an env var with a sentinel) become an argument of the
  PATH solver the recipe names; no environment variable is written by the run.

### Solvers

- PATH (`solver/`): the current `_run_path_capi_nonlinear_full` behaviour behind the
  core `Solver` signature, for `PyomoPeriodModel`.
- Newton-TR (`solver/`): the family route's `_newton` behind the same signature, for
  the family `PeriodModel`. Its settle cache and the ratio-bordered nominal-savf
  shock are kept (as a strategy or solver option, builder's call).
- The GC and PATH-option context managers move to `solver/` (design of ADR-0011).

### Choosing the backend

The template carries it: `GTAP(params, ..., backend="pyomo" | "family")`, default
`"pyomo"`. `Exercise.run()` only calls `template.build(periods, overwrite)` (the
`Template` protocol of step 5), so the backend has to live in the template; the core
does not change. Running the same data on both backends is two `GTAP` objects; a
family run asked for something it cannot do (altertax, `@overwrite`...) refuses in
`build`, before any work. `EQUILIBRIA_GTAP_BACKEND` and the other `EQUILIBRIA_GTAP_*` switches that only
route between the old drivers go away; the ones that tune a solver become solver
arguments or stay as documented solver options.

### `FactorBlock.calibrate_base`

It builds and solves a whole model from inside a block (the known layering exception
in `tests/core/test_layering.py`). It must run through the new driver; injecting the
builder (the fix the exception names) is in scope only if the switch needs it.

## Migration (one design, four PRs, in order)

Each PR keeps the gate below.

1. **GTAP on the core, both backends.** Contract bulk operations + contract test,
   family `PeriodModel`, `GTAP` template, `GTAPStatic` and its steps, PATH and
   Newton-TR solvers. The old drivers keep running and stay the default.
2. **Compare.** Every GTAP parity gate and every Burfisher exercise through the old
   route and the new one, per backend: same codes, same values within each gate's
   existing tolerance. The family route is compared on what it supports today.
3. **Switch.** Every caller (`build_block_model`/`solve_block_model`, the simulator,
   `gtap_logvalue`, `gtap_julia`, scripts, tests) moves to `Exercise`; the Burfisher
   notebooks are regenerated by their generator (same numbers against GEMPACK).
4. **Delete.** the old Pyomo multiperiod driver, the old Pyomo period recipe, the old family driver,
   the old family period recipe and what only they use; every reference in `src`, `tests`,
   `scripts`, `examples`, `docs`, `README`, specs, and `.github` lists. A test fails if
   any of those module names reappears.

## Verification

- The migration gate (ADR-0011): every check and every Burfisher exercise gives the
  same result as on `main` before the change. Full suite compared by test name and
  skip reason (local-only parity tests included, `gdxdump` on the PATH, and the local
  reference files linked: PATH library, GEMPACK `.sl4`, pep2 GDX), not only by count.
- gtap7_20x41: wall time on each backend no worse than on `main` (median of three on
  the same machine).
- Contract test: every `PeriodModel` operation gives the same state on Pyomo and on
  family.
- Layering: `core/` run modules import no backend (exists); GTAP steps import no
  backend module (new, same kind of test).
- After PR 4: no file names the old drivers, their period recipes or their entry
  points (`tests/test_no_old_driver_names.py`).

## Left to the builder

Names of the bulk operations; where the family `PeriodModel` lives; where the
closure is applied so both solvers see it; how the settle cache and the ratio border are expressed; the
exact split of `gtap_block_model.py` helpers between template, steps and solver.
