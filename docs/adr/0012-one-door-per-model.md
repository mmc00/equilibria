---
status: accepted
date: 2026-11-26
---

# One door per model, with the same names in every model

A user who wants to load a base, run a quick shock and read the results should not
need to know where each piece lives inside the package. Before this decision a GTAP
shock needed five imports from five internal modules (`core.exercise`,
`templates.gtap.gtap_contract`, `templates.gtap.recipe`, `templates.gtap.template`,
`datasets`), nine closure fields and the results read cell by cell from the Pyomo
model. The public names have to be fixed before the stable release: after it,
changing them breaks the users' code.

So every model gets one public door, `equilibria.<model>` (today `equilibria.gtap`;
later `equilibria.simple_open` and `equilibria.pep`), and every door follows the same
contract:

| Name in the door | What it is | Rule |
|---|---|---|
| `load(source, ...)` | Loads the model's data | Accepts a bundled name, a `data/` dataset name and a folder path, in that order; a missing file is a `FileNotFoundError` naming the folder and the file |
| the template class (`GTAP`, ...) | What `Exercise` builds | Satisfies `core.exercise.Template` (`build`, `default_recipe`) |
| `ClosureConfig` | The model's closure class | Subclass of `contracts.ModelClosureConfig`; at least one named classmethod starting point that returns a `ClosureConfig` and takes `**changes` (GTAP: `gtap_default`); its field defaults never change for the door's sake |
| a recipe factory (`gtap_static`, ...) | How the periods are solved | Returns a `core.recipe.Recipe` |
| `Exercise` | The core `Exercise` | The same object as `core.exercise.Exercise` |
| `results(ex, ...)` | The view of a solved exercise | Raises `RuntimeError` before `ex.run()`; returns an object with `tables: dict[str, pandas.DataFrame]` and `summary: pandas.DataFrame` |

Rules for the whole door:

- It only re-exports or wraps: every name is the same object as its source, except
  `load` and `results`, which are new functions. No piece is hidden behind a
  do-everything call: the user still builds `Exercise(GTAP(...), recipe, shock=...)`
  and sees each piece.
- It only adds: the internal import paths keep working.
- It declares `__all__`, and every contract name is in it.

The contract is written three times and the three must agree: here (the why),
in `src/equilibria/core/door.py` (what is checked, `check_door`) and in
`tests/core/test_door_contract.py` (which runs the check on every door in `DOORS`).
The behaviour of `load` and `results` needs data and a solve, so each door's own
tests check it (GTAP: `tests/gtap/test_load.py`, `tests/gtap/test_results.py`).

No template is left without a door by accident. `find_templates` reads the source
(nothing imported) and finds every template class: one with a real
`default_recipe`, or a subclass of `ModelTemplate`. Each must be the template of a
door in `DOORS` or be listed in `PENDING` with the reason it has none yet (today
`simple_open`, which has no closure class of its own). A `PENDING` entry that names
no template, or whose template already has a door, is an error too. Every door also
has its page in the API reference (`docs/site/api/`). These checks run in CI
(`tests/core/test_door_contract.py`) and before every commit that touches
`src/equilibria/` (the `check-doors` hook, `scripts/dev/check_doors.py`, about 1 s).

## Considered Options

- **Three functions that do everything** (`load`, `shock`, `results`). Rejected: it
  hides the template, the closure and the recipe, and a user who needs to change one
  of them has to learn the internal modules anyway.
- **Change the closure defaults to the standard closure.** Rejected: every
  `GTAPClosureConfig()` in tests, scripts and the family backend would change in
  silence. A named starting point (`ClosureConfig.gtap_default()`) shows which
  closure the user starts from and breaks nothing.
- **A new, short GEMPACK name list for `results`.** Rejected: it would be a second,
  unvalidated translation. `results` uses the one in `equilibria.compare.gtap7_side`,
  the same one the GEMPACK gates use.

## Consequences

- A model without a closure class (`simple_open` today) or with an engine still in
  transition (PEP, R-PEP) gets its door later, and the contract test then asks the
  same of it.
- Advanced closures keep going through `@overwrite` (ADR-0004); the door re-exports
  `overwrite`, and there is no "swap" API.

## Sources

- MAR-249 (design `2026-11-26-mar249-gtap-door-design.md`), MAR-250, MAR-251
