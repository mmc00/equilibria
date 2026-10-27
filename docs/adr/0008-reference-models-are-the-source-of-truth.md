---
status: accepted
date: 2026-09-16
---

# GAMS and GEMPACK are the source of truth; fidelity outranks match %

The reference implementations of each model (GAMS and GEMPACK) define what is
correct. A change is judged by whether it is faithful to them, not by what it
does to a match percentage: a faithful change is never reverted because a figure
dropped, and the only reason to remove one is that it is unfaithful. Divergent
cells are never excluded to raise the match.

A match percentage is a ratio, and it moves for reasons that are not fidelity.
Moving the GTAP gates from the monolith to blocks "lost" 0.17 to 0.35 points in
`altertax`, yet both models failed the same 110 cells: only the denominator had
changed (exogenous shock variables that are right by construction). A later
"drop" from 99.3 % to 98.97 % compared a June note against a September
measurement. Chasing the number would have meant changing a correct model. So
the parity gates (`scripts/gtap/run_parity_gates.py`) compare against the
reference models, and a Claude Code hook
(`scripts/gtap/claude_hooks/block_push_without_gates.py`) blocks an agent's push
that touches GTAP until they have run (a plain terminal push is not blocked).

## Consequences

Before blaming the model, check the closure, the seed and the fixture: most
historical "bugs" were closure mismatches or uncalibrated seeds. Compare the
cells that fail, not the percentages.

## Sources

- `ROADMAP.md`, "Convenciones que conviene no reaprender por las malas"
- `docs/architecture/monolito_vs_bloques.md` ("Por qué se pudo migrar" and the
  2026-09-21 correction)
