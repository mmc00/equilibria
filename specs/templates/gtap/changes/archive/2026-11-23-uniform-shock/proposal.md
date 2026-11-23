# A uniform shock on an instrument

Linear: MAR-227.

## Why

The linearization study (`scripts/gtap/gen_linearization_study.py`) compares
Python with GEMPACK at five shock sizes (10, 3, 1, 0.3, 0.1 %), but Python is
solved at 10 % only: the smaller columns read "py-shock pending". The GEMPACK
records for every size already exist (`gempack_tm<shock>_s8-16-32_capflex`),
each shocked with one line, `Shock tm = uniform <x>`.

Python has no way to say "x % on every cell of `imptx`". `Exercise(shock=...)`
takes explicit cells, and which cells can take a shock (`check_shock_cell`)
is only known once the model is built, inside `ex.run()`. Building the cell
list in the caller means copying that rule from the data (a cell too many
raises, a cell too few shocks another set than the 10 % default) or building
the model twice. The default tariff already has the rule inside, but its 10 %
is a constant.

The goal is internal: see how Python behaves against GEMPACK as the shock
shrinks.

## What changes

- `shocks.uniform(pct)` (a small marker value) may stand in for the cell map of
  an instrument: `Exercise(..., shock={"imptx": uniform(3)})`.
- `apply_shock` (Pyomo, `templates/gtap/shocks.py`) and its array port
  (`family/shock.py`) expand the marker, with the model built, to every `shock`
  cell of that instrument that can take a shock (`_can_take_shock` /
  `can_take_shock`, the filter `apply_default_tariff` uses), each at `pct`, with
  the instrument's GEMPACK kind. All or nothing, as today. No cell can take it:
  `ValueError`.
- No shock still means the +10 % default; it is not rewritten in this change.
- The study solves Python at every size with `uniform(<size>)` and reports two
  match columns: within 1 pp (today's) and within a tolerance scaled with the
  shock (1 pp x size / 10), since at 0.1 % almost no cell moves 1 pp and the
  1 pp column would read ~100 % whatever Python does.

## Approaches considered

- **A uniform marker in the exercise's shock (chosen).** The shock belongs to
  the exercise (`core/exercise.py`); the cell rule runs where the model exists;
  it mirrors GEMPACK's `uniform`.
- A `tariff_pct` option on `gtap_static`: works, but puts a shock in the
  recipe, and only for `imptx`.
- Patching `DEFAULT_TARIFF_PCT` from the script: hidden, nobody would find it.
- Explicit cells built in the script: fragile (see Why) or a double build.

## Oracle (committed before the code)

- In `mode="gtap"`, `uniform(10)` on `imptx` writes exactly the cells and the
  level values that the default tariff writes (Pyomo and family), and the
  solved 3x3 shock equals the default run to 1e-12 relative.
- `uniform(3)` writes the power (1+t)*1.03-1 on the same cell set.
- A marker on an instrument with no cell that can take a shock raises.

## Out of scope

- Altertax: `uniform` uses the GEMPACK kind (power for `imptx`), not the
  altertax rate x 1.1 default; no equality with that default is promised.
- Rewriting the default tariff on top of `uniform`.
