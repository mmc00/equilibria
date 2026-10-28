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
to remember to update it. One GAMS family may map to more than one equilibria
family when equilibria writes each branch of a conditional equation as its own
row (`pfteq` is `eq_pfteq` for the CET branch and `eq_xft` for the mobile one).

## Consequences

Every family gets a row, including the ones whose name is the same on both
sides (`pa`/`pa`, `xw`/`xw`). There is no default rule such as "GAMS `<x>eq` is
equilibria `eq_<x>`": it guesses wrong often enough to hide a family. Measured
in the 3x3 model, it would pair `xgeq(r,t)` (`pg*xg = yg`) with `eq_xg(r,i,t)`
(`xa(r,i,'gov')`), `xieq` with `eq_xi`, `xseq` with `eq_xs` (which is GAMS
`pseq`) and `xfteq` with `eq_xft`, and the GAMS variable `xd(r,i,aa)` with the
equilibria aggregate `xd(r,i)`. A row also says how the indices line up: the
GAMS indices fixed to one element (`xaceq(r,i,'hhd',t)` is `eq_xc(r,i,t)`) and
the ones only equilibria carries (`walras` is a scalar in GAMS and is indexed by
period in equilibria). The test checks that the counts add up, so a pair of
families with different shapes cannot be declared by mistake.

The GAMS side is read from the declarations in the `.gms` files, not from a
`.nl` export. The declarations are the same for `ifSUB=0` and `ifSUB=1`, so the
dictionary covers both formulations, while a `.nl` file only contains the one it
was generated with.

A one-sided entry is not an exclusion from a comparison: it says the
counterpart does not exist (a shifter that GAMS writes as an equation and
equilibria as a fixed value, a price that GAMS substitutes with a macro), and it
carries the reason.

## Sources

- Issue #23 and commit `6938280` (the `.nl` gate compared zero rows)
- ADR-0008 (the reference models are the source of truth)
- `src/equilibria/compare/maps/gtap7.yaml` (#132), the equilibria/GEMPACK
  comparison recipe that this dictionary feeds
