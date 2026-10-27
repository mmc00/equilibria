# The default tariff shock goes through the ShockBlock

## Why

The paper (section 2.3) says a shock is a new fixed value of an instrument and
the equations are not touched, but the 10% tariff of Table 1 and Table A1 is
applied by rebuilding equations (gtap) or by patching prices after the solve
(altertax). Measured on gtap7_3x3 against the GAMS fixtures: the `imptx` Var of
the `shock` period keeps its `check` value in all four cases (gtap/altertax x
ifSUB 0/1), and in altertax the tariff never enters the solve: `xw` moves 1e-7
while GAMS moves it up to 1%. Table A1 passes only because those moves are
below its 1% tolerance.

## What changes

- With no shock in the model the driver writes the tariff into the ShockBlock
  with `apply_shock`, as GAMS does (`tm.fx`): in gtap +10% on the power 1+t,
  every route including the domestic diagonal; in altertax the rate times 1.1.
- `mode="altertax"` accepts a ShockBlock shock.
- A model without the ShockBlock (the monolith `GTAPMultiPeriodModel`) is
  rejected with a clear error; its four tests move to the block model and the
  seven diagnostic scripts that solve it stop until they are migrated.
- An `imptx` cell whose `eq_pmeq` is off (ifSUB=1) is accepted when a live
  `eq_xweq` of the same cell reads the `imptx` Var (macro M_PM).
- Removed: the equation rebuilds of the tariff (`_rebuild_eq_pmeq_shock`,
  `_rebuild_eq_ytax_mt_shock`, `_rebuild_import_demand_shock_ifsub`), the
  `params` tariff (`_apply_imptx_shock`) and the post-solve tariff patches of
  `ytax[mt]` and `pm`/`pmt`/`pa`.

## Oracle (committed before the code)

- `tests/templates/gtap/test_default_tariff_shock.py`: `imptx[*,'shock']` equals
  GAMS in the four cases; in altertax every `xw` cell that GAMS moves by more
  than 0.1% moves the same way; the gtap7_3x3 Table 1 solution equals v0.6.1
  (fixture `table1_levels_v061_gtap7_3x3.json.gz`, rel 1e-8).
- `tests/templates/gtap/test_imptx_shock_cell.py`: ifSUB=1 accepts every traded
  cell; ifSUB=0 without `eq_pmeq` still rejects.
- By hand, before merging: Table 1 for all six datasets against the v0.6.1 JSON
  of the clean replication, and the full Table A1 (MCP and NLP gates).

## Issues

- #113 (PR 2 of 2)

## Impact (areas, ADRs)

`templates/gtap`. Closes the open consequence of ADR-0003 (the default tariff
still lived in the driver). Table A1 shock figures for altertax change and the
paper's section 2.3 caveat goes away.
