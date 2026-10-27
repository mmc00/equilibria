---
status: accepted
date: 2026-10-27
---

# One name dictionary per model, checked for completeness

Every name that equilibria shares with a reference model (GAMS, GEMPACK) is
declared once, in a dictionary that ships with the package
(`src/equilibria/compare/maps/<model>_names.yaml`). A row says which equilibria
equation or variable corresponds to which GAMS (and GEMPACK) one, how their
indices line up, or why one side has no counterpart. Every tool that pairs names
reads that dictionary; none keeps its own.

Before this, each tool paired names on its own: at least six scripts in
`scripts/gtap/` carried a private GAMS-to-Python map, and the `.nl` structural
gate guessed with a normalizer. What a map did not know was skipped without a
word. Measured on 2026-10-27 in `gtap7_3x3/base`: the gate paired 398 of the 797
GAMS rows, and within those rows 1,847 of 3,080 coefficients were not compared
because their variables are named differently (`xa`/`xaa`, `xd`/`xda`,
`regY`/`regy`). The gate was green while comparing a fraction of the model.

So the dictionary is checked for completeness against both sides: every
equation and variable family in the GAMS fixtures, and every one in the
equilibria model, must be paired or declared one-sided with its reason. A new
name on either side turns the test red instead of being skipped, so nobody has
to remember to update it.

## Consequences

A name that follows the default rule (GAMS `<x>eq` is equilibria `eq_<x>`, a
variable keeps its name) needs no row; the completeness test still counts it, so
the rule cannot hide a family. A one-sided entry is not an exclusion from a
comparison: it says the counterpart does not exist (a shifter that GAMS writes
as an equation and equilibria as a fixed value, a price that GAMS substitutes
with a macro), and it carries the reason.

## Sources

- Issue #23 and commit `6938280` (the `.nl` gate compared zero rows)
- ADR-0008 (the reference models are the source of truth)
- `src/equilibria/compare/maps/gtap7.yaml` (#132), the equilibria/GEMPACK
  comparison recipe that this dictionary feeds
