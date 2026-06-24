# GTAP 7 Parity Coverage Matrix

<!-- GENERATED FROM scripts/gtap/coverage_matrix.py — do not edit by hand.
     Regenerate: uv run python scripts/gtap/gen_coverage_doc.py -->

`gap_min` is the conservative floor the tests assert vs GAMS; `gempack_min` is the floor vs GEMPACK/RunGTAP; the `_note` columns are measured snapshots. `solver`: `mcp` (PATH) or `nlp` (walras/ifMCP=0). `ci_status`: `ci` runs on ubuntu without a solver, `local` needs PATH+GAMS (run by hand), `blocked` has an unsound reference.

## Progreso global

- total: 18
- done (≥99% / 0-diff): 17
- partial: 0
- blocked: 1

## Single-period (`.nl` coefficient gate, CI, no solver)

| dataset | kind | solver | ifsub | phases | gap_min | gap_note | gempack_min | gempack_note | ci_status | ref |
|---|---|---|---|---|---|---|---|---|---|---|
| nus333 | gtap | mcp | — | base,shock | 99.5 | 100% (NEOS+GAMS) | — | — | ci | nus333 NEOS |
| 9x10 | gtap | mcp | — | base,shock | 99.5 | 100% (NEOS) | — | — | ci | job 18737509 |
| gtap7_3x3 | gtap | mcp | — | base,shock | — | 0 diffs .nl | — | — | ci | gams_base/shock.nl |
| gtap7_5x5 | gtap | mcp | — | base,shock | — | 0 diffs .nl | — | — | ci | gams_base/shock.nl |
| gtap7_10x7 | gtap | mcp | — | base,shock | — | 0 diffs .nl | — | — | ci | gams_base/shock.nl |
| gtap7_15x10 | gtap | mcp | — | base,shock | — | 0 diffs .nl | — | — | ci | gams_base/shock.nl |
| gtap7_3x4 | gtap | mcp | — | base,shock | — | 0 diffs .nl | — | — | ci | gams_base/shock.nl |

## Altertax multi-period (solver gate, local-only)

| dataset | kind | solver | ifsub | phases | gap_min | gap_note | gempack_min | gempack_note | ci_status | ref |
|---|---|---|---|---|---|---|---|---|---|---|
| gtap7_3x3 | altertax | mcp | 0 | base,check,shock | 98 | ~99% | — | — | local | out_altertax_ifsub0.gdx |
| gtap7_3x3 | altertax | mcp | 1 | base,check,shock | 98 | ~99% | — | — | local | out_altertax_ifsub1.gdx |
| gtap7_5x5 | altertax | mcp | 0 | base,check,shock | 99.5 | 100% | — | — | local | out_altertax_ifsub0.gdx |
| gtap7_5x5 | altertax | mcp | 1 | base,check,shock | 99.5 | 100% | — | — | local | out_altertax_ifsub1.gdx |
| gtap7_10x7 | altertax | mcp | 0 | base,check,shock | 98 | ~99% | — | — | local | out_altertax_ifsub0.gdx |
| gtap7_10x7 | altertax | mcp | 1 | base,check,shock | 98 | ~99% | — | — | local | out_altertax_ifsub1.gdx |
| gtap7_15x10 | altertax | mcp | 0 | base,check,shock | 99 | 99.30% | — | — | local | out_altertax_ifsub0.gdx |
| gtap7_15x10 | altertax | mcp | 1 | base,check,shock | 99 | 99.30% | — | — | local | out_altertax_ifsub1.gdx |
| gtap7_3x4 | altertax | mcp | 0 | base,check,shock | 99 | 99.61% | — | — | local | out_altertax_ifsub0.gdx |
| gtap7_3x4 | altertax | mcp | 1 | base,check,shock | 99 | 99.56% | — | — | local | out_altertax_ifsub1.gdx |
| gtap7_20x41 | altertax | mcp | 0 | base | — | blocked | — | — | blocked | NEOS ref Infeasible |

## Modularización (refactor a bloques — F3)

| bloque | estado |
|---|---|
| production_ces_nest | planned |
| armington_trade | planned |
| cde_demand | planned |
| institutions_tax | planned |
| closure_fisher | planned |
