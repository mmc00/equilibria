---
status: accepted
date: 2026-09-17
---

# Blocks are the model; a monolith is only a starting point and an oracle

In every template the model is the composition of its blocks. A monolithic
implementation (one file that writes every equation) may exist while a template
is being built, as a starting point and as a parity oracle, but it is not the
model and the goal is always to retire it. New features, closures and exercises
go into blocks only; nothing new is added to a monolith.

GTAP shows why. The block path (`blocks/gtap/`, composed in
`templates/gtap/gtap_block_model.py`) is already a functional superset of
`templates/gtap/gtap_model_equations.py`: it has `capFlex`, `capFixDp` and
`base_calibrated=True`, which the monolith never had, and only the block path
can be checked against GEMPACK. The composer no longer imports the monolith, the
package loads it lazily, and the `mcp`, `nlp` and `gempack` gates build blocks.
The monolith survives only as what the `nl` gate checks against GAMS's `.nl`
(`test_gtap7_nl_parity.py` through `scripts/gtap/nl_compare.py`). That check is
weak today: the gate currently compares 0 constraints, because the GAMS fixtures
lack their `.row`/`.col` sidecars (#23, `ROADMAP.md` "Bloqueantes conocidos").
PEP is earlier on the same road: it still
has both the monolith `templates/pep_model_equations.py` and the Pyomo rebuild
`templates/pep_pyomo/`.

## Considered Options

- **Keep both implementations as peers, each checked against an oracle.** This
  redundancy is what exposed the Fisher snapshot bug, but it doubles gate time
  and every model change has to be written twice. Rejected: the monolith stays
  readable as a reference, but it must not produce noise or be treated as live
  code.
- **Delete the monolith now.** Rejected while it is what the `nl` gate checks
  against GAMS; it goes once that gate measures blocks with unchanged numbers.

## Consequences

A template that still has a monolith carries an open change to retire it
(`ROADMAP.md`: "Sacar el monolito del test de forma" and the
`gtap_model_equations.py` row of "Bloqueantes conocidos" for GTAP; PEP has no
such entry yet and needs one). The
monolith refuses `@overwrite` hooks (`require_blocks`) instead of ignoring them
silently, so an exercise cannot run on the wrong model by accident.

## Sources

- `docs/architecture/monolito_vs_bloques.md`
- `ROADMAP.md`, "Bloqueantes conocidos" (`gtap_model_equations.py` no es retirable)
