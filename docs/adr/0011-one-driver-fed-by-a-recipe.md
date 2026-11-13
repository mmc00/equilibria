---
status: accepted
date: 2026-11-12
---

# One driver for every model, fed by a recipe

Supersedes ADR-0002. A run is split into five things with one job each. **Blocks**
hold the economics, for one period (ADR-0006). A **template** says which blocks a
model has, in what order they are built, its default closure and its default
recipe, and carries the model's data. A **recipe** lists the periods, each with
a kind (`calibrate`, `replicate`, `shock`, `step`), and for each kind the steps
before and after the solve, the solve strategy and the solver. The **driver**
flies the recipe: for each period it runs the steps, solves and freezes, and it
knows no model, shock or closure. An **exercise** is one run: template, recipe,
shock and its own `@overwrite` blocks, run with `Exercise(...).run()`.

ADR-0002 already moved period preparation into a recipe, but the recipe was
private to GTAP (`period_prep.py`), its steps reached back into the driver file,
and the driver still held about 1,700 lines of GTAP logic. One driver that only
flies a recipe can run GTAP, `simple_open` and later PEP and a recursive dynamic
GTAP, and an exercise that carries its own `@overwrite` blocks cannot leak them
into the next one (#214).

## Considered Options

- **A driver with two modes** (one model indexed by `t`, or one model per period,
  for PEP). Rejected: more responsibilities for the driver, the opposite goal.
- **Change the existing driver little by little.** Rejected: a new driver is
  built next to the old one and switched in one move once every gate matches.
- **One object that builds and runs** (the recipe also listing the blocks).
  Rejected: one template runs under several recipes (static, dynamic, Johansen).

## Consequences

- Equations are never rewritten between periods. A coefficient that must change
  between periods is a block's own update rule run as a step. Step 0 of the
  migration deleted four per-period recalibrations; `_recalibrate_and_ava` stays
  until the NLP gtap7_15x10 ifSUB=0 row it holds is understood (#243).
- The migration gate is that every check and every Burfisher exercise gives the
  same result as on `main`, at every step.

## Sources

- `specs/templates/gtap/changes/driver-recipe/design.md`
- #214, #240, #242, #243
