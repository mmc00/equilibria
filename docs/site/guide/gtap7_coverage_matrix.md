# GTAP 7 Parity Coverage Matrix

<!-- GENERATED FROM scripts/gtap/coverage_matrix.py — do not edit by hand.
     Regenerate: uv run python scripts/gtap/gen_coverage_doc.py -->

`gap_min` is the conservative floor the tests assert; `gap_note` is the measured snapshot. `ci_status`: `ci` runs on ubuntu without a solver, `local` needs PATH+GAMS (run by hand), `blocked` has an unsound reference.

## Single-period (`.nl` coefficient gate, CI, no solver)

| dataset | kind | ifsub | phases | gap_min | gap_note | ci_status | ref |
|---|---|---|---|---|---|---|---|
| nus333 | gtap | — | base,shock | 99.5 | 100% (NEOS+GAMS) | ci | nus333 NEOS |
| 9x10 | gtap | — | base,shock | 99.5 | 100% (NEOS) | ci | job 18737509 |
| gtap7_3x3 | gtap | — | base,shock | — | 0 diffs .nl | ci | gams_base/shock.nl |
| gtap7_5x5 | gtap | — | base,shock | — | 0 diffs .nl | ci | gams_base/shock.nl |
| gtap7_10x7 | gtap | — | base,shock | — | 0 diffs .nl | ci | gams_base/shock.nl |
| gtap7_15x10 | gtap | — | base,shock | — | 0 diffs .nl | ci | gams_base/shock.nl |
| gtap7_3x4 | gtap | — | base,shock | — | 0 diffs .nl | ci | gams_base/shock.nl |

## Altertax multi-period (solver gate, local-only)

| dataset | kind | ifsub | phases | gap_min | gap_note | ci_status | ref |
|---|---|---|---|---|---|---|---|
| gtap7_3x3 | altertax | 0 | base,check,shock | 98 | ~99% | local | out_altertax_ifsub0.gdx |
| gtap7_3x3 | altertax | 1 | base,check,shock | 98 | ~99% | local | out_altertax_ifsub1.gdx |
| gtap7_5x5 | altertax | 0 | base,check,shock | 99.5 | 100% | local | out_altertax_ifsub0.gdx |
| gtap7_5x5 | altertax | 1 | base,check,shock | 99.5 | 100% | local | out_altertax_ifsub1.gdx |
| gtap7_10x7 | altertax | 0 | base,check,shock | 98 | ~99% | local | out_altertax_ifsub0.gdx |
| gtap7_10x7 | altertax | 1 | base,check,shock | 98 | ~99% | local | out_altertax_ifsub1.gdx |
| gtap7_15x10 | altertax | 0 | base,check,shock | 99 | 99.30% | local | out_altertax_ifsub0.gdx |
| gtap7_15x10 | altertax | 1 | base,check,shock | 99 | 99.30% | local | out_altertax_ifsub1.gdx |
| gtap7_3x4 | altertax | 0 | base,check,shock | 99 | 99.61% | local | out_altertax_ifsub0.gdx |
| gtap7_3x4 | altertax | 1 | base,check,shock | 99 | 99.56% | local | out_altertax_ifsub1.gdx |
| gtap7_20x41 | altertax | 0 | base | — | blocked | blocked | NEOS ref Infeasible |

## Pure-gtap multi-period SOLVE (real-CES, solver gate, local-only)

| dataset | kind | ifsub | phases | gap_min | gap_note | ci_status | ref |
|---|---|---|---|---|---|---|---|
| gtap7_3x3 | gtap_solve | 0 | base,check,shock | 99 | 99.70% | local | out_gtap_shock_ifsub0.gdx |
| gtap7_3x3 | gtap_solve | 1 | base,check,shock | 98 | 98.95% | local | out_gtap_shock_ifsub1.gdx |
