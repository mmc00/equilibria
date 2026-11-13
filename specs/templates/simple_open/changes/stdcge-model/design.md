# simple_open becomes Hosoe's standard CGE model, built from blocks

Migration step 4 of `specs/templates/gtap/changes/driver-recipe/design.md`.
Decided with the maintainer on 2026-11-13.

## Why

`simple_open` is equilibria's small single-country template. It has never been
solved: `create_model` builds a three-block demo (CES value added, Leontief
intermediates, CET) with no trade, households or government, and its "GAMS
parity" (`simple_open_parity_pipeline.py`, `simple_open_parity_spec.py`)
compares nine hand-written benchmark levels and six parameters against a GDX,
with no solve, shock or period. It was invented in February 2026 for teaching
and never followed a published model.

The driver-recipe migration needs a second real model besides GTAP, so the
recipe is not shaped around GTAP alone (step 5 builds the core with
`simple_open` as its first user). A model that does not solve cannot play
that part.

## What it becomes

**`simple_open` is the standard CGE model of Hosoe, Gasawa and Hashimoto,
*Handbook of Computable General Equilibrium Modeling* (University of Tokyo
Press, 2004), chapter 6**, as cited in the `.gms` header, shipped
with GAMS as `stdcge` (GAMS model library, SEQ=276; the local copy is
`<GAMS>/Resources/gamslib_ml/stdcge.276`). It is a published model with a
`.gms`, so it has a source of truth like PEP and GTAP (ADR-0008).

stdcge in short (two goods BRD and MLK, two factors CAP and LAB, ten SAM
accounts):

- production: Cobb-Douglas composite factor `Y(j)` from `F(h,j)`, fixed
  (Leontief) intermediates `X(i,j)` and composite factor per unit of output
  `Z(j)`;
- trade: Armington CES between imports `M` and domestic goods `D` (σ = 2), CET
  between exports `E` and domestic goods (ψ = 2), world prices fixed, the
  exchange rate `epsilon` clears the balance of payments with foreign saving
  `Sf` fixed;
- government: direct tax, production tax and import tariffs; fixed saving
  propensity; Cobb-Douglas consumption shares;
- household: Cobb-Douglas demand, fixed saving propensity;
- investment: fixed shares of total saving (private, government, foreign);
- numeraire: `pf("LAB") = 1`;
- GAMS solves it as an NLP that maximizes utility `UU`; the shock is the
  abolition of all import tariffs (`taum(i) = 0`); welfare is measured by the
  Hicksian equivalent variation `EV`.

## Goals

1. **The book is the starting point, not a cage** (maintainer). The model
   reads its goods and factors from the SAM: BRD and MLK are the data of the
   first case, not names in the code. Any SAM with the same accounts (goods,
   factors, indirect tax, tariff, household, government, investment, rest of
   the world) builds and solves.
2. **Built from blocks** (maintainer). The template declares generic blocks
   from `blocks/` wherever they fit; what serves only this model goes in
   `templates/simple_open/` (the R-ARCH rule in the driver-recipe design).
3. **Solved and checked against GAMS** in three points:
   - **base**: the calibrated SAM;
   - **check**: solve with no shock; it must return the base;
   - **shock**: import tariffs set to zero.
4. **Replaces the old parity entirely.** Everything from the March attempt is
   deleted (maintainer); the new parity uses the existing comparison tool.

## Non-goals (this step)

- No `Recipe`, `Step`, new driver or `Exercise`: that is step 5, which will
  use `simple_open` as its first user. This step only needs the model to
  build, calibrate, solve and be compared.
- No multiple households, no dynamics, no shocks other than the book's.
- No GAMS reference for SAMs other than the book's.

## Anti-goals (would make this a failure even if tests pass)

- A test that cannot fail: the reference must come from running the real
  `.gms`, never from levels written by hand (that is exactly what the old
  parity did).
- Cobb-Douglas treated as a CES with σ = 1, or approximated by a log-linear
  form. Issue #192 showed how much that costs. The production function and the
  household demand must be the exact Cobb-Douglas of the `.gms`.
  (`blocks/production` `CESValueAdded` today switches to a "log-linearized
  Cobb-Douglas form" when σ = 1; that is not acceptable here.)
- Goods or factor names hard-coded anywhere in the template or its blocks. The
  numeraire factor belongs to the data: `closure.yaml` next to each SAM
  (`numeraire: LAB` for the book's), or an explicit `numeraire=` (decided with
  the maintainer after the final review).

## Decisions

- **Solve form** *(Claude's call, accepted by the maintainer)*: equilibria
  solves the square system (as for GTAP and PEP), not the NLP. At the GAMS
  optimum the NLP's first-order conditions are the same equilibrium, so the
  levels must match. The objective `UU` is computed as a reported quantity, not
  maximized.
- **Reference and its generator** *(maintainer: regenerate on the Mac with GAMS
  53; the Demo license covers this size)*. Following ADR-0010 and the rule that
  generators live with the data:
  - the `.gms` is copied from GAMS's model library into `data/` unchanged, with
    a small driver `.gms` (or `$include` wrapper) that writes one GDX with the
    three points on a period axis `t ∈ {base, check, shock}` and every
    variable level, plus the calibrated parameters;
  - the generator script lives under the data's `tools/` folder and records a
    `source.yaml` (dataset, `.gms`, GAMS version, solver, command, sha256 of
    inputs);
  - exact folder names are the builder's call (for example
    `data/stdcge/hosoe/` for the SAM and results, `data/stdcge/gams/` for the
    `.gms` and `tools/`).
- **"check" in GAMS** *(Claude's call)*: the book's `.gms` solves base, then
  shock. The wrapper adds a check solve between them (same data, no shock), as
  the GTAP references do, so all three points come from GAMS.
- **Parity tool** *(Claude's call, accepted)*: `equilibria.compare`
  (`from_gdx`, `from_pyomo`, a `Mapping` in a `map.yaml`), not a new tool. The
  gate is a test that fails if any cell of base, check or shock falls outside
  tolerance. The diagnostic scripts in `scripts/gtap/` are used only if a gap
  has to be debugged.
- **Generality test** *(Claude's call, accepted)*: a second, invented SAM with
  three goods builds, calibrates, solves and passes its check (returns its own
  base). It has no GAMS reference (non-goal), so it proves "not tied to bread
  and milk", not parity.
- **Equation form** *(Claude's call)*: every equation is written in levels
  exactly as in the `.gms` (`Y = b·Π F^β`, `Xp = α·(income − Sp − Td)/pq`,
  ...). A log transform of an identity is exact in value but changes the
  Jacobian and the root PATH or Newton finds, so it is not used; the
  log-linearized Cobb-Douglas branch of `CESValueAdded` and the log form of
  `CobbDouglasConsumer` are either replaced by level forms or not used.
- **Squaring the system** *(Claude's call)*: without the objective, the
  `.gms` has as many equations as variables (24 blocks each), and fixing
  `pf("LAB")` leaves one equation redundant (Walras' law). Following GTAP, a
  free `walras` slack variable enters one market-clearing equation and is
  checked to be ≈ 0 at every solution; no equation is dropped. `UU` is a
  defined variable (`UU = Π Xp^α`) so the system stays square and `UU` and
  `EV` can be compared with GAMS.
- **Calibration is independent** *(Claude's call)*: Python calibrates every
  parameter from the SAM itself, with its own code. The calibrated parameters
  in the GDX are a second comparison (Python vs GAMS), never an input. Reading
  GAMS's parameters into the model would make the parity unable to fail.
- **Investment** *(Claude's call)*: no generic block fits `eqXv` today; it
  goes in a model-specific block in `templates/simple_open/` unless a generic
  investment block turns out to serve PEP or GTAP as well.
- **Generic blocks are tested for the first time.** None of the blocks listed
  below has ever been solved inside a complete model. Fixing them is in scope;
  each fix must keep every GTAP and PEP gate as it is on main. If a block
  cannot express the `.gms` faithfully without bending it for other users, a
  model-specific block goes in `templates/simple_open/` instead.

## Starting map: stdcge equations to existing blocks

A starting point for the builder, not a contract. The fit has not been
checked equation by equation.

| stdcge equations | candidate block today |
|---|---|
| `eqpy`, `eqF` (CD composite factor) | `blocks/production` `CESValueAdded` (needs an exact CD form) |
| `eqX`, `eqY`, `eqpzs` (Leontief, unit cost) | `blocks/production` `LeontiefIntermediate` |
| `eqpzd`, `eqE`, `eqDs` (CET, with `(1+tauz)`) | `blocks/production` `CETTransformation` or `blocks/trade` `CETExports` |
| `eqpqs`, `eqM`, `eqD` (Armington, with `(1+taum)`) | `blocks/trade` `ArmingtonCES` |
| `eqXp`, `eqSp` (household) | `blocks/demand` `CobbDouglasConsumer`, `blocks/institutions` `Household` |
| `eqTd`, `eqTz`, `eqTm`, `eqXg`, `eqSg` (government) | `blocks/institutions` `Government` |
| `eqXv` (investment) | none obvious |
| `eqpe`, `eqpm`, `eqepsilon` (world prices, balance of payments) | `blocks/institutions` `RestOfWorld` |
| `eqpqd`, `eqpf` (market clearing) | `blocks/equilibrium` `MarketClearing`, `FactorMarketClearing` |
| `pf("LAB") = 1` | `blocks/equilibrium` `PriceNormalization` |

Watch the calibration details of the `.gms`: `deltam` includes `(1+taum)`;
`D0 = (1+tauz)·Z0 − E0`; the CET first-order conditions use `(1+tauz)·pz`;
`lambda` divides by total saving `Sp0+Sg0+Sf`; `ssg` divides by total tax
revenue.

## What gets deleted

All of the March 2026 "parity" and the demo template it served (maintainer):

- `src/equilibria/templates/simple_open_parity_pipeline.py`,
  `simple_open_parity_spec.py`, `simple_open_constraint_jacobian.py`,
  `simple_open_contract.py`, `simple_open_runtime_config.py`;
- `src/equilibria/baseline/simple_open_gams_reference.py`;
- `scripts/parity/generate_simple_open_gams_reference.py`,
  `run_simple_open_gams_parity.py`, `measure_simple_open_jacobian_modes.py`;
- `output/simple_open_*` (GDX, JSON and `simple_open_gams_reference/`);
- their tests: `tests/templates/test_simple_open_contract.py`,
  `test_simple_open_parity_spec.py`,
  `tests/qa/test_simple_open_gams_reference_manifest.py`,
  `tests/scripts/test_generate_simple_open_gams_reference_cli.py`,
  `test_measure_simple_open_jacobian_modes_cli.py`,
  `test_run_simple_open_gams_parity_cli.py`;
- `notes/epic_simple_open_gams_parity_2026-03-19.md`.

Rewritten, not deleted: `examples/cge/example_07_templates.py` (shows the new
model: build, solve, shock, EV) and the `SimpleOpenEconomy` cases in
`tests/templates/test_templates.py`. Anything else that imports the deleted
modules (for example `templates/__init__.py` re-exports, docs pages) is
updated. The builder confirms the full list with a search before deleting.

## Success criteria

1. `simple_open` built from blocks, on the book's SAM, matches the GAMS GDX in
   **100 % of the cells** of base, check and shock (`equilibria.compare`),
   including `EV` and `UU`.
2. The three-good invented SAM builds, calibrates, solves and returns its base
   on check.
3. No goods or factor name appears in the template or block code.
4. The reference regenerates from its `source.yaml` command and reproduces the
   committed GDX; the `source.yaml` test is green.
5. Every GTAP and PEP gate and the fast suite are as on main (same results,
   same skips: compare test names and skip reasons, not only counts).
6. The old parity files are gone and nothing imports them.

## Left to the builder

- Exact file layout under `templates/simple_open/` and `data/`.
- Whether `ModelTemplate` (`templates/base.py`) needs changes now or in step 5
  (prefer the minimum now; step 5 extends it).
- Tolerances for `equilibria.compare` (start from what the GTAP gates use).
- Which solver runs the square system (whatever the template stack already
  uses; PATH and Newton must both be tried once and the result recorded).
- Whether the spec system needs a new area in `specs/areas.toml` for
  `simple_open`.

## Sources

- GAMS model library, `stdcge` (SEQ=276):
  https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_stdcge.html
- Hosoe, N., Gasawa, K., Hashimoto, H. (2004). *Handbook of Computable
  General Equilibrium Modeling*. University of Tokyo Press (citation as given
  in the `.gms` header; the English edition and its chapter numbering are not
  verified here).
- `specs/templates/gtap/changes/driver-recipe/design.md` (migration step 4)
- ADR-0008 (reference models are the source of truth), ADR-0010 (one data
  folder)
