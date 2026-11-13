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

- **First, #214** (exercises leaking `@overwrite` hooks) is fixed on the old
  driver.
- **Step 0, cleanup (before any redesign):** delete what turns out to be
  unnecessary, one item per PR, each with a failing test first and a measurement
  against GAMS. First candidates: the four per-period recalibrations
  (`_recalibrate_and_ava`, `_gx_ax`, `_alphaa_gov_inv`, `_alphaa_hhd`), which GAMS
  does not do (it calibrates every share once, in `loop(tsim$t0)`).
- Three concepts with one job each: the **Template** (which blocks, in what
  order, with its data), the **Recipe** (periods, steps, solve strategy and
  solver per kind of period) and the **Driver** (walks the recipe; knows no
  model). An **Exercise** (template + recipe + shock + `@overwrite`) is the
  public way to run: `Exercise(...).run()`.
- `simple_open` is the second user of the core, to prove the driver is generic.
  It is never solved today, so it first becomes a real model (design, step 4).
- The new driver is built **next to** the old one. GTAP switches over in one
  move, once the new driver gives the same numbers on every gate; the Burfisher
  notebooks move to `Exercise` then.
- The docs get new Lego drawings (3D bricks, a parts table, colors by functional
  form, the shock as a Technic suspension) and pages told with a rocket.
- PEP gets periods and its recipe later, by Ricardo, as stage E3 of `R-PEP`.

## Issues

#214. Board cards: `R-DRIVER-FIT`, `R-DRIVER`, `R-DOCS-LEGO`, `R-DOCS-P2`,
`R-DYN` (unblocked by this), `R-PROGRESS` (its default summary hooks into the
new driver), `R-PEP` (E3).
Issues to open per task when the change is accepted.

## Impact (areas, ADRs)

Areas: `blocks` (core: `Recipe`, `Step`, driver, `Exercise`, period
reflection), `templates` (`ModelTemplate`), `templates/gtap`,
`templates/simple_open`, `solver` (GC and PATH-option handling move there),
docs (architecture page, a new guide, the GTAP quickstart, the glossary).
ADRs: relies on ADR-0001, 0003, 0004, 0005, 0006, 0007. Supersedes ADR-0002 (the
driver moves out of `templates/gtap/`, and template, recipe and exercise become
public contracts); a new ADR records that.
