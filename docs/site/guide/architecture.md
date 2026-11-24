# Architecture overview

```{raw} html
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600&family=Atkinson+Hyperlegible:wght@400;700&family=JetBrains+Mono:wght@400;600&display=swap">
<link rel="stylesheet" href="../_static/jax_backend.css">
```

`equilibria` builds computable general equilibrium (CGE) models in Python from
reusable **blocks**, and checks every model against its reference
implementation in GAMS and GEMPACK. This page explains how the pieces fit,
first with a picture and then with the real code. The words in **bold** are
the ones the code uses; they are defined in `docs/glossary.md`.

## A model is a stack of bricks

Think of a box of building bricks. Each brick does one job, and you build a
model by snapping bricks on top of each other. In equilibria a brick is a
**block**: a class that declares its parameters, variables and equations, and
says which variables it reads from the bricks below it.

This is GTAP, as it is built today:

```{raw} html
:file: ../../assets/architecture/tower.svg
```

Three rules keep the tower standing:

- **The bricks are the model** ([ADR-0001](#decisions)). GTAP *is* these eight
  blocks. A new closure, feature or exercise goes into a block; nothing new is
  added to the old single-file implementation (the **monolith**).
- **Bricks only snap if they agree** ([ADR-0005](#decisions)). Two blocks may
  declare the same name only if both declarations are identical (value,
  domain, bounds); otherwise `Model.add_block` raises an error naming both.
  Each block also lists in `uses` what it reads from other blocks, and the
  build fails if nobody declares it.
- **No catalogue of bricks** ([ADR-0007](#decisions)). There is no registry.
  The template imports its blocks directly, in dependency order:
  `GTAP_BLOCK_ORDER`, composed by `templates/gtap/gtap_block_model.py`.

| Brick | Class | File in `blocks/gtap/` |
|---|---|---|
| Instruments | `ShockBlock` | `shock.py` |
| Domestic vs export | `TradeCETBlock` | `trade_cet.py` |
| Production and supply | `ProductionSupplyBlock` | `production_supply.py` |
| Factor markets | `FactorBlock` | `factor.py` |
| Imports by origin | `ArmingtonBilateralBlock` | `trade_armington_bilateral.py` |
| Final demand and utility | `DemandUtilityBlock` | `demand_utility.py` |
| Income and taxes | `IncomeBlock` | `income.py` |
| Closure | `ClosureBlock` | `closure.py` |

## Build one period, copy it three times

A brick has no calendar. Each block writes its equations for **one period**,
with no time index. That one-period model is the **SP** (single-period) model.

To run a shock, GTAP needs three periods: `base` (the benchmark data),
`check` (the benchmark must replicate) and `shock`. equilibria does not
rewrite the bricks for that: it photocopies the SP model once per period and
adds the `t` index to every variable ([ADR-0006](#decisions)).

```{raw} html
:file: ../../assets/architecture/periods.svg
```

Because the bricks never saw `t`, changing the time structure (for example,
a list of years for a recursive dynamic model) does not touch any block's
equations. The copying lives in `templates/gtap/gtap_model_multiperiod.py`,
and the periods in `blocks/gtap/periods.py`.

## A shock is one cell that differs

The `ShockBlock` declares every policy and technology **instrument** (`imptx`,
`lambdava`, `aft`, …) as a variable fixed in every period. A **shock** is an
instrument whose `shock` cell differs from its `check` cell, exactly as in
GAMS ([ADR-0003](#decisions)). You give it to the exercise in GEMPACK's terms:
the number is the percentage you would read in a `.EXP` file, and the exercise
writes it in the `shock` period (with `apply_shock`) before solving.

<!-- fragmento -->
```python
from equilibria.core.exercise import Exercise

# +10 % on the tariff power (1 + tms) of one route, as in a GEMPACK .EXP
ex = Exercise(gtap, recipe, shock={"imptx": {("EU_28", "Mnfcs", "USA"): 10.0}})
```

Base and check do not move, so they always replicate the benchmark, whatever
the shock is. The full walk-through is in {doc}`gtap_quickstart`.

## An exercise swaps one brick, in one period

Some exercises need a different closure or an extra equation: a real-wage
target, a quantity target. Instead of copying the model, the exercise
changes **one** block, and only in the shock period, with `@overwrite`
([ADR-0004](#decisions)):

```{raw} html
:file: ../../assets/architecture/overwrite.svg
```

The check and shock towers are the same set of bricks except the top one: in
the shock period the closure tile is lifted off and a closure with one more
equation goes in its place. In code:

<!-- fragmento -->
```python
from equilibria.blocks.gtap import ClosureBlock, ShockBlock
from equilibria.blocks.gtap.overwrite import overwrite

@overwrite(ShockBlock, period="shock")
def unemployment(b):
    b.endogenous("aft", ("USA", "LABOR"))  # free one instrument cell

@overwrite(ClosureBlock, period="shock")
def real_wage(b):
    b.equation("eq_wreal", ("USA", "LABOR"), ...)  # and add the row that pins it
```

The three verbs (`endogenous`, `equation`, `target`) express what GEMPACK
calls a `swap`. The hook is registered on the block class, so every builder
that uses that block sees it.

## The driver follows a recipe

```{raw} html
:file: ../../assets/architecture/recipe.svg
```

The **driver** (`core/driver.py`) is the person assembling the kit: it does
not design bricks, it follows the instructions ([ADR-0011](#decisions)).
`Exercise(template, recipe, shock).run()` hands it the three pieces. The
**recipe** (`gtap_static` in `templates/gtap/recipe.py`) lists the periods
(`base`, `check`, `shock`) and, for each kind of period, the steps before and
after the solve (freeze the other periods, seed, recalibrate, fix and bound),
how to solve (in one go, in steps, Johansen) and with which solver. For each
period the driver runs the steps, solves, and moves on. Equations, instruments
and the shock belong to blocks, never to the driver.

## The instruction booklet is GAMS and GEMPACK

A model built from bricks is only right if it matches the original. The
reference implementations in GAMS and GEMPACK define what is correct
([ADR-0008](#decisions)):

- A change is judged by its fidelity to the reference, not by what it does
  to a match percentage. Divergent cells are never excluded to raise the
  match.
- Every name equilibria shares with a reference model is declared once, in
  a dictionary that ships with the package
  (`compare/maps/gtap7_names.yaml`, [ADR-0009](#decisions)). A name that
  appears on either side without a row turns its test red instead of being
  skipped.
- The parity gates (`scripts/gtap/run_parity_gates.py`) re-measure the match
  against GAMS and GEMPACK, same engine on both sides (NLP vs NLP, MCP vs
  MCP), so the solver tolerance cancels and the number reflects the model.

## Where each template is

| Template | How it is built today |
|---|---|
| GTAP Standard 7 | Blocks (`blocks/gtap/`). The driver no longer solves the monolith `gtap_model_equations.py` (it requires a `ShockBlock`); the monolith survives as the GAMS oracle of the `.nl` structural gate until that gate measures blocks. |
| PEP-1-1 v2.1 | Not yet on blocks. Two engines: the monolith `pep_model_equations.py` and the Pyomo rebuild `pep_pyomo/` (100 % vs GAMS). |
| simple_open | Generic blocks in `blocks/production`, `trade`, `demand`, `institutions`, `equilibrium`. |

`blocks/gtap_logvalue` and `blocks/gtap_loglevels` are research variants of
GTAP written in log form; they are not the production path.

## The pipeline

```{raw} html
:file: ../../assets/architecture/pipeline.svg
```

1. **Data I/O** (`equilibria.babel`, `equilibria.sam_tools`) reads and
   balances the input data: SAM matrices, GEMPACK HAR files (native
   pure-Python reader and writer), GAMS GDX containers, and raw MIP/IO
   tables ({doc}`MIP → SAM <mip_to_sam>`, {doc}`HAR I/O <har_io>`).
2. **Calibration** fits the parameters so the benchmark reproduces the data.
   For GTAP it lives in one place today, `blocks/gtap/_derived_params.py`;
   PEP calibrates from the SAM with its own code.
3. **Blocks → model**: a template composes its blocks into a Pyomo model
   (`equilibria.model`, `equilibria.backends`).
4. **Driver + solver**: the same model solves as an **MCP** (PATH through
   its C API, {doc}`path_capi`) or as an **NLP** (IPOPT), mirroring GAMS's
   `ifMCP` switch.
5. **Validation** compares every solved cell against the reference
   ({doc}`benchmarks`, {doc}`gtap7_coverage_matrix`,
   {doc}`pep_coverage_matrix`), through `equilibria.compare`.

## Package map

| Package | Responsibility |
|---|---|
| `equilibria.sam_tools` | SAM audit, balancing, MIP → SAM transforms |
| `equilibria.babel` | File formats: HAR reader/writer, GDX reader, SAM loaders |
| `equilibria.model` | `Model`: `add_block` (one name, one declaration) and the `uses` check |
| `equilibria.blocks` | `Block` (`base.py`) and the blocks: `gtap/` (GTAP), generic `production`, `trade`, `demand`, `institutions`, `equilibrium` |
| `equilibria.core` | Model assembly core: sets, parameters, variables, equations, calibration phases |
| `equilibria.templates.gtap` | GTAP Standard 7: composer, multi-period model, driver, recipes, shocks |
| `equilibria.templates.pep_pyomo` | PEP-1-1 v2.1 ported to Pyomo (NLP + MCP) |
| `equilibria.compare` | Cell-by-cell comparison against GAMS/GEMPACK, and the name dictionaries (`maps/`) |
| `equilibria.baseline` | Reference manifests and strict compatibility checks |
| `equilibria.calibration` | Generic calibration utilities (CES, Leontief) |
| `equilibria.backends` | Pyomo backend (model construction) |
| `equilibria.solver` | Solver support: PATH C API, guards, Jacobian tooling |
| `equilibria.simulations` | Scenario runtime, adapters, multi-model wrappers |
| `equilibria.contracts`, `equilibria.qa` | Validation contracts and SAM/QA checks |

## Decisions

The rules above are written down as Architecture Decision Records in
`docs/adr/` in the repository. Each one is short and says what was decided,
why, and what was rejected.

| ADR | Decision |
|---|---|
| 0001 | Blocks are the model; a monolith is only a starting point and an oracle |
| 0002 | The driver prepares and solves; it holds no model logic (superseded by 0011) |
| 0003 | The shock lives in the ShockBlock and is written as in GEMPACK |
| 0004 | An exercise changes one block with `@overwrite` and leaves the others alone |
| 0005 | One name, one declaration; each block declares what it uses |
| 0006 | Blocks do not know about periods; the SP copy adds the time index |
| 0007 | No block registry |
| 0008 | GAMS and GEMPACK are the source of truth; fidelity outranks match % |
| 0009 | One name dictionary per model, checked for completeness against both sides |
| 0011 | One driver for every model, fed by a recipe (supersedes 0002) |

See the {doc}`templates overview <templates>` for per-template status and
the {doc}`API reference <../api/index>` for the generated reference.
