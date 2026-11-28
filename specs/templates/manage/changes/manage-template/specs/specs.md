## ADDED Requirements

### Requirement: MANAGE Ghana one-year run from blocks
Source: change manage-template, ADR-0001, ADR-0011

`ManageTemplate` must build MANAGE-WB for `gha_gtap9` from blocks on Pyomo, and
`Exercise(ManageTemplate()).run()` must run the recipe `manage_one_year`: period
`2014` (`calibrate`, not solved) and period `2015` (`step`, solved with IPOPT),
with the active equation families and row instances of the GAMS 2015 solve
(9,452 rows).

### Requirement: Calibration read from the GAMS run, in solve units
Source: change manage-template, ADR-0008

The calibration must read MANAGE's calibrated parameters and 2014 levels from
the `GHA_BaU.gdx` of the reference run, converted to GAMS's solve units, and
must not read any 2015 level of a free cell. The Python model must work in
solve units.

### Requirement: The 2015 step reproduces GAMS's year update
Source: change manage-template, ADR-0008, `data/manage/gams/inc/solve/resLastYear.inc`

Before the 2015 solve, the recipe's steps must set the exogenous 2015 cells and
the 2015 start point as GAMS's `iterloop.inc` and `resLastYear.inc` do,
including the upstream quirks (`tPaf` copied only for exported goods,
`nshrGDP` only for `f`).

### Requirement: Parity with GAMS in every cell
Source: change manage-template, ADR-0008, ADR-0009

Every variable cell of 2014 and 2015 must match the reference
`reference("gha_gtap9", "gams", form="mcp", task="baseline", year=2015, mcp_solver="nlpec-ipopt")`
through `compare/maps/manage.yaml`, at the tolerance the maintainer sets after
the measurement. Every family on both sides must be paired or declared
one-sided in `compare/maps/manage_names.yaml`.
