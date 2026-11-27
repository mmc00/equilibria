# MANAGE-WB as a blocks template (Ghana, one year)

## Why

equilibria has three reference models (GTAP, PEP, simple_open); none is a
single-country recursive-dynamic model with nested CES production, a CDE
household and a full fiscal account. The World Bank's MANAGE-WB is that model,
and its GAMS reference for Ghana is already recorded in `data/manage/`
(PR #283): base year 2014 calibrated, 2015 solved, 9,452 rows, solved locally by
NLPEC/IPOPT and confirmed by PATH on NEOS to 7.9e-9 relative. Porting it to
blocks (ADR-0001) gives equilibria its first model whose production tree and
demand system come from data, and the shared blocks that this needs (a CES tree
read from a nest table, CDE demand, taxes by agent, dual-form Armington and
CET) serve the next models (dynamic GTAP, IEEM).

## What changes

- A new template `equilibria.templates.manage` (`ManageTemplate`), built from
  blocks on the Pyomo backend, for the one-year Ghana run with dynamics,
  vintages, forward-looking expectations, climate damages and model links off.
- New shared blocks under `src/equilibria/blocks/` for forms equilibria does
  not have: a CES tree read from a nest table (used for production and for
  final demand), dual-form Armington and CET, agent prices with ad valorem
  taxes by (good, agent), and CDE household demand. Thirteen model-specific
  blocks under `templates/manage/blocks/` cover the rest (factor supply tree,
  income, government, funds and debt, capital, macro, demography, emissions,
  reporting, ...).
- A data reader and calibration for `data/manage/gha_gtap9/`: first the
  calibrated parameters and the 2014 levels read from GAMS's `GHA_BaU.gdx`
  (the reference decides, ADR-0008). Porting `inc/cal/inical.inc` from the
  bridge GDX is a follow-up change, checked against these values.
- A recipe `manage_one_year` with periods `2014` (`calibrate`) and `2015`
  (`step`), including the year-update rules of `inc/solve/resLastYear.inc`
  exactly as GAMS runs them (with its tPaf/nshrGDP quirk).
- A name dictionary `compare/maps/manage_names.yaml` (ADR-0009) and a parity test
  of every cell of 2014 and 2015 against
  `reference("gha_gtap9", "gams", form="mcp", task="baseline", year=2015, mcp_solver=...)`.
- The door `equilibria.manage` (ADR-0012), an entry in
  `docs/site/guide/templates.md`, glossary words.

## Issues

To open one per task when the change is accepted (see the plan,
`specs/2026-11-27-manage-template-plan.md`). Relates to #283 (the reference).

## Impact (areas, ADRs)

Areas: `templates/manage` (new area in `specs/areas.toml`:
`src/equilibria/templates/manage/`, `src/equilibria/manage.py`; this change),
`templates/simple_open` and `blocks` (new shared blocks under
`src/equilibria/blocks/`; existing blocks are not changed), `data` (GDX reader
coverage, design risk 1), `parity` (the name dictionary and the comparison).
ADRs relied on: 0001, 0003, 0004, 0005, 0006, 0008, 0009, 0010, 0011, 0012.
Creates ADR-0013 (a block reads its nest tree from data, proposed). No ADR is
superseded.
