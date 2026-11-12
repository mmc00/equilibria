# One driver for every model, fed by a recipe

## Why

`templates/gtap/gtap_multiperiod_driver.py` is 2,502 lines and about 80 % of
them are not orchestration: it rewrites block equations between periods
(`_recalibrate_*`), carries GTAP-only fixes to square the system, builds a
single-period reference model, and tunes the garbage collector and PATH options.
The maintainer's objections (2026-11-11): the driver knows GTAP details, it mixes
closure, shock and solve, and it is not clear how it fits the block architecture.
It also cannot serve PEP or any other model. ADR-0002 already says the driver
"holds no model logic"; the code is not there.

## What changes

- **Step 0, cleanup (before any redesign):** delete what turns out to be
  unnecessary, one item per PR, each with a failing test first and a measurement
  against GAMS. First candidates: the four per-period recalibrations
  (`_recalibrate_and_ava`, `_gx_ax`, `_alphaa_gov_inv`, `_alphaa_hhd`), which GAMS
  does not do (it calibrates every share once, in `loop(tsim$t0)`).
- A generic `Recipe` (abstract class) and a generic driver in the core. The driver
  only walks the recipe's periods: steps before, solve, steps after, freeze.
- One recipe per model, next to its model: `GTAPRecipe` first, then
  `SimpleOpenRecipe` as the second model that proves the driver is generic.
- The new driver is built **next to** the old one. GTAP switches over in one
  move, once the new driver gives the same numbers on every gate.
- PEP gets periods and its recipe later, by Ricardo, as stage E3 of `R-PEP`.

## Issues

Board cards: `R-DRIVER-FIT`, `R-DRIVER`, `R-DYN` (unblocked by this),
`R-PROGRESS` (its default summary hooks into the new driver), `R-PEP` (E3).
Issues to open per task when the change is accepted.

## Impact (areas, ADRs)

Areas: `blocks` (core: `Recipe`, driver, period reflection), `templates/gtap`,
`templates/simple_open`, `solver` (GC and PATH-option handling move there).
ADRs: relies on ADR-0001, 0003, 0004, 0006. Supersedes ADR-0002 (the driver
moves out of `templates/gtap/` and the recipe becomes a public, per-model
contract); a new ADR records that.
