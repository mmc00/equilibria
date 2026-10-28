# Architecture overview

`equilibria` builds computable general equilibrium (CGE) models in Python from
reusable **blocks**, and checks every model against its reference
implementation in GAMS and GEMPACK. This page explains how the pieces fit,
first with a picture and then with the real code. The words in **bold** are
the ones the code uses; they are defined in `GLOSSARY.md`.

## A model is a stack of bricks

Think of a box of building bricks. Each brick does one job, and you build a
model by snapping bricks on top of each other. In equilibria a brick is a
**block**: a class that declares its parameters, variables and equations, and
says which variables it reads from the bricks below it.

This is GTAP, as it is built today:

```{raw} html
<svg viewBox="0 0 760 500" role="img" aria-label="GTAP as a tower of eight blocks, ShockBlock at the bottom and ClosureBlock on top"
     style="max-width:100%;height:auto;font-family:sans-serif">
  <rect x="0" y="0" width="760" height="500" rx="14" fill="#f7f6f2"/>
  <g font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="15" font-weight="600" fill="#ffffff">
    <g fill="#14213D"><rect x="110" y="20" width="36" height="10" rx="3"/><rect x="262" y="20" width="36" height="10" rx="3"/><rect x="414" y="20" width="36" height="10" rx="3"/><rect x="60" y="30" width="440" height="44" rx="5"/></g>
    <text x="78" y="58">ClosureBlock</text>
    <g fill="#1E40AF"><rect x="110" y="74" width="36" height="10" rx="3"/><rect x="262" y="74" width="36" height="10" rx="3"/><rect x="414" y="74" width="36" height="10" rx="3"/><rect x="60" y="84" width="440" height="44" rx="5"/></g>
    <text x="78" y="112">IncomeBlock</text>
    <g fill="#1D4ED8"><rect x="110" y="128" width="36" height="10" rx="3"/><rect x="262" y="128" width="36" height="10" rx="3"/><rect x="414" y="128" width="36" height="10" rx="3"/><rect x="60" y="138" width="440" height="44" rx="5"/></g>
    <text x="78" y="166">DemandUtilityBlock</text>
    <g fill="#1E40AF"><rect x="110" y="182" width="36" height="10" rx="3"/><rect x="262" y="182" width="36" height="10" rx="3"/><rect x="414" y="182" width="36" height="10" rx="3"/><rect x="60" y="192" width="440" height="44" rx="5"/></g>
    <text x="78" y="220">ArmingtonBilateralBlock</text>
    <g fill="#1D4ED8"><rect x="110" y="236" width="36" height="10" rx="3"/><rect x="262" y="236" width="36" height="10" rx="3"/><rect x="414" y="236" width="36" height="10" rx="3"/><rect x="60" y="246" width="440" height="44" rx="5"/></g>
    <text x="78" y="274">FactorBlock</text>
    <g fill="#1E40AF"><rect x="110" y="290" width="36" height="10" rx="3"/><rect x="262" y="290" width="36" height="10" rx="3"/><rect x="414" y="290" width="36" height="10" rx="3"/><rect x="60" y="300" width="440" height="44" rx="5"/></g>
    <text x="78" y="328">ProductionSupplyBlock</text>
    <g fill="#1D4ED8"><rect x="110" y="344" width="36" height="10" rx="3"/><rect x="262" y="344" width="36" height="10" rx="3"/><rect x="414" y="344" width="36" height="10" rx="3"/><rect x="60" y="354" width="440" height="44" rx="5"/></g>
    <text x="78" y="382">TradeCETBlock</text>
    <g fill="#C2410C"><rect x="110" y="398" width="36" height="10" rx="3"/><rect x="262" y="398" width="36" height="10" rx="3"/><rect x="414" y="398" width="36" height="10" rx="3"/><rect x="60" y="408" width="440" height="44" rx="5"/></g>
    <text x="78" y="436">ShockBlock</text>
  </g>
  <g font-size="13" fill="#bfcbe0" text-anchor="end">
    <text x="486" y="57">numeraire, market clearing, Walras</text>
    <text x="486" y="111">regional income, taxes, GDP</text>
    <text x="486" y="165">final demand (CDE), utility</text>
    <text x="486" y="219">imports by origin</text>
    <text x="486" y="273">factor markets</text>
    <text x="486" y="327">production nests, supply</text>
    <text x="486" y="381">domestic vs export (CET)</text>
    <text x="486" y="435" fill="#ffe8d9">the instruments: imptx, lambdava…</text>
  </g>
  <rect x="40" y="452" width="480" height="8" rx="3" fill="#94a3b8"/>
  <g font-size="14" fill="#18202e">
    <text x="545" y="56" font-weight="700">Closes the model</text>
    <text x="545" y="76" fill="#5b6577">the last brick</text>
    <text x="545" y="420" font-weight="700">Goes first</text>
    <text x="545" y="440" fill="#5b6577">every brick above reads</text>
    <text x="545" y="458" fill="#5b6577">the instruments it declares</text>
    <text x="545" y="230" font-weight="700">8 blocks</text>
    <text x="545" y="250" fill="#5b6577">GTAP_BLOCK_ORDER</text>
    <text x="545" y="270" fill="#5b6577">in blocks/gtap/__init__.py</text>
  </g>
  <text x="280" y="486" font-size="13" fill="#5b6577" text-anchor="middle">Model.add_block checks every snap</text>
</svg>
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
<svg viewBox="0 0 900 290" role="img" aria-label="One single-period model copied into base, check and shock; only the shock copy has a changed instrument"
     style="max-width:100%;height:auto;font-family:sans-serif">
  <defs>
    <marker id="arch-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#8b96a3"/>
    </marker>
  </defs>
  <rect x="0" y="0" width="900" height="290" rx="14" fill="#f7f6f2"/>
  <g>
    <rect x="40" y="70" width="150" height="34" rx="4" fill="#1E40AF"/>
    <rect x="40" y="106" width="150" height="34" rx="4" fill="#1D4ED8"/>
    <rect x="40" y="142" width="150" height="34" rx="4" fill="#1E40AF"/>
    <rect x="40" y="178" width="150" height="34" rx="4" fill="#C2410C"/>
  </g>
  <text x="115" y="50" text-anchor="middle" font-size="16" font-weight="700" fill="#18202e">SP model</text>
  <text x="115" y="240" text-anchor="middle" font-size="13" fill="#5b6577">written once</text>
  <text x="115" y="258" text-anchor="middle" font-size="13" fill="#5b6577">no time index</text>
  <g fill="none" stroke="#8b96a3" stroke-width="1.6" marker-end="url(#arch-arr)">
    <line x1="205" y1="141" x2="300" y2="141"/>
  </g>
  <text x="252" y="128" text-anchor="middle" font-size="12" fill="#5b6577">copy per period</text>
  <g font-size="16" font-weight="700" fill="#18202e" text-anchor="middle">
    <text x="395" y="50">base</text><text x="595" y="50">check</text><text x="795" y="50">shock</text>
  </g>
  <g>
    <rect x="320" y="70" width="150" height="34" rx="4" fill="#1E40AF"/><rect x="320" y="106" width="150" height="34" rx="4" fill="#1D4ED8"/><rect x="320" y="142" width="150" height="34" rx="4" fill="#1E40AF"/><rect x="320" y="178" width="150" height="34" rx="4" fill="#C2410C"/>
    <rect x="520" y="70" width="150" height="34" rx="4" fill="#1E40AF"/><rect x="520" y="106" width="150" height="34" rx="4" fill="#1D4ED8"/><rect x="520" y="142" width="150" height="34" rx="4" fill="#1E40AF"/><rect x="520" y="178" width="150" height="34" rx="4" fill="#C2410C"/>
    <rect x="720" y="70" width="150" height="34" rx="4" fill="#1E40AF"/><rect x="720" y="106" width="150" height="34" rx="4" fill="#1D4ED8"/><rect x="720" y="142" width="150" height="34" rx="4" fill="#1E40AF"/><rect x="720" y="178" width="150" height="34" rx="4" fill="#C2410C" stroke="#18202e" stroke-width="3"/>
  </g>
  <g font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="13" fill="#ffffff" text-anchor="middle">
    <text x="395" y="200">imptx = 1.00</text><text x="595" y="200">imptx = 1.00</text><text x="795" y="200" font-weight="700">imptx = 1.10</text>
  </g>
  <g font-size="13" fill="#5b6577" text-anchor="middle">
    <text x="395" y="240">the benchmark data</text>
    <text x="595" y="240">must replicate base</text>
    <text x="795" y="240">one cell differs:</text>
    <text x="795" y="258">that is the shock</text>
  </g>
</svg>
```

Because the bricks never saw `t`, changing the time structure (for example,
a list of years for a recursive dynamic model) does not touch any block's
equations. The copying lives in `templates/gtap/gtap_model_multiperiod.py`,
and the periods in `blocks/gtap/periods.py`.

## A shock is one cell that differs

The `ShockBlock` declares every policy and technology **instrument** (`imptx`,
`lambdava`, `aft`, …) as a variable fixed in every period. A **shock** is an
instrument whose `shock` cell differs from its `check` cell, exactly as in
GAMS ([ADR-0003](#decisions)). You write it on the built model, in GEMPACK's
terms: the number is the percentage you would read in a `.EXP` file.

```python
from equilibria.templates.gtap import apply_shock

# +10 % on the tariff power (1 + tms) of one route, as in a GEMPACK .EXP
apply_shock(m, {"imptx": {("EU_28", "Mnfcs", "USA"): 10.0}})
```

Base and check do not move, so they always replicate the benchmark, whatever
the shock is. The full walk-through is in {doc}`gtap_quickstart`.

## An exercise swaps one brick, in one period

Some exercises need a different closure or an extra equation: a real-wage
target, a quantity target. Instead of copying the model, the exercise
changes **one** block, and only in the shock period, with `@overwrite`
([ADR-0004](#decisions)):

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

The **driver** (`templates/gtap/gtap_multiperiod_driver.py`) is the person
assembling the kit: it does not design bricks, it follows the instructions
([ADR-0002](#decisions)). For each period it asks for the period to be
prepared, solves it with the closure it gets back, and moves on. How a
period is prepared (freeze the other periods, seed, recalibrate, fix and
bound) is a **recipe**: an ordered list of steps per period and mode in
`templates/gtap/period_prep.py`. Equations, instruments and the shock belong
to blocks, never to the driver.

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
<svg viewBox="0 0 920 150" role="img" aria-label="equilibria pipeline"
     style="max-width:100%;height:auto;font-family:sans-serif;font-size:13px">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#8b96a3"/>
    </marker>
  </defs>
  <g fill="none" stroke="#8b96a3" stroke-width="1.5" marker-end="url(#arr)">
    <line x1="168" y1="75" x2="192" y2="75"/>
    <line x1="352" y1="75" x2="376" y2="75"/>
    <line x1="536" y1="75" x2="560" y2="75"/>
    <line x1="720" y1="75" x2="744" y2="75"/>
  </g>
  <g text-anchor="middle">
    <rect x="8" y="45" width="160" height="60" rx="10" fill="#e8f0f8" stroke="#2f6fb0"/>
    <text x="88" y="70" fill="#1a2129" font-weight="600">Data I/O</text>
    <text x="88" y="90" fill="#5a6673" font-size="11">SAM · HAR · GDX · MIP</text>
    <rect x="192" y="45" width="160" height="60" rx="10" fill="#e8f0f8" stroke="#2f6fb0"/>
    <text x="272" y="70" fill="#1a2129" font-weight="600">Calibration</text>
    <text x="272" y="90" fill="#5a6673" font-size="11">benchmark → parameters</text>
    <rect x="376" y="45" width="160" height="60" rx="10" fill="#e8f0f8" stroke="#2f6fb0"/>
    <text x="456" y="70" fill="#1a2129" font-weight="600">Blocks → model</text>
    <text x="456" y="90" fill="#5a6673" font-size="11">gtap · pep_pyomo · simple_open</text>
    <rect x="560" y="45" width="160" height="60" rx="10" fill="#e8f0f8" stroke="#2f6fb0"/>
    <text x="640" y="70" fill="#1a2129" font-weight="600">Driver + solver</text>
    <text x="640" y="90" fill="#5a6673" font-size="11">PATH (MCP) · IPOPT (NLP)</text>
    <rect x="744" y="45" width="168" height="60" rx="10" fill="#e6f3ec" stroke="#2f8f5b"/>
    <text x="828" y="70" fill="#1a2129" font-weight="600">Validation</text>
    <text x="828" y="90" fill="#5a6673" font-size="11">gates vs GAMS · GEMPACK</text>
  </g>
</svg>
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
| 0002 | The driver prepares and solves; it holds no model logic |
| 0003 | The shock lives in the ShockBlock and is written as in GEMPACK |
| 0004 | An exercise changes one block with `@overwrite` and leaves the others alone |
| 0005 | One name, one declaration; each block declares what it uses |
| 0006 | Blocks do not know about periods; the SP copy adds the time index |
| 0007 | No block registry |
| 0008 | GAMS and GEMPACK are the source of truth; fidelity outranks match % |
| 0009 | One name dictionary per model, checked for completeness against both sides |

See the {doc}`templates overview <templates>` for per-template status and
the {doc}`API reference <../api/index>` for the generated reference.
