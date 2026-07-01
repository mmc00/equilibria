# GTAP 7 Parity Coverage Matrix

<!-- GENERATED FROM scripts/gtap/coverage_matrix.py — do not edit by hand.
     Regenerate: uv run python scripts/gtap/gen_coverage_doc.py -->

`gap_min` is the conservative floor the tests assert; `gap_note` is the measured snapshot (shock match% @ tol1% for the solver gates). `ci_status`: `ci` runs on ubuntu without a solver, `local` needs PATH+GAMS (run by hand), `blocked` has an unsound reference.

## `.nl` coefficient gate (CI, no solver)

Diffs Python vs GAMS Jacobian coefficients. Phases are base+shock, plus `check` (the CD multi-period step) where a `gams_check.nl` fixture exists (3x3/5x5/10x7). Contract: 0 coefficient diffs. ifSUB does not apply.

| dataset | kind | ifsub | phases | gap_min | gap_note | ci_status | ref |
|---|---|---|---|---|---|---|---|
| nus333 | gtap | — | base,shock | 99.5 | 100% (NEOS+GAMS) | ci | nus333 NEOS |
| 9x10 | gtap | — | base,shock | 99.5 | 100% (NEOS) | ci | job 18737509 |
| gtap7_3x3 | gtap | — | base,shock | — | 0 diffs .nl | ci | gams_base/shock.nl |
| gtap7_5x5 | gtap | — | base,shock | — | 0 diffs .nl | ci | gams_base/shock.nl |
| gtap7_10x7 | gtap | — | base,shock | — | 0 diffs .nl | ci | gams_base/shock.nl |
| gtap7_15x10 | gtap | — | base,shock | — | 0 diffs .nl | ci | gams_base/shock.nl |
| gtap7_3x4 | gtap | — | base,shock | — | 0 diffs .nl | ci | gams_base/shock.nl |

## Altertax multi-period SOLVE gate (PATH, local-only)

Builds + seeds + solves base→check→shock in altertax-CD mode; asserts 3×code=1 and shock match ≥ gap_min, per ifSUB.

| dataset | kind | ifsub | phases | gap_min | gap_note | ci_status | ref |
|---|---|---|---|---|---|---|---|
| gtap7_3x3 | altertax | 0 | base,check,shock | 98 | 99.93% (98.88% @0.5%) | local | out_altertax_ifsub0.gdx |
| gtap7_3x3 | altertax | 1 | base,check,shock | 98 | 99.78% (98.51% @0.5%) | local | out_altertax_ifsub1.gdx |
| gtap7_5x5 | altertax | 0 | base,check,shock | 99.5 | 99.88% (98.53% @0.5%) | local | out_altertax_ifsub0.gdx |
| gtap7_5x5 | altertax | 1 | base,check,shock | 99.5 | 99.81% (98.38% @0.5%) | local | out_altertax_ifsub1.gdx |
| gtap7_10x7 | altertax | 0 | base,check,shock | 98 | 99.33% (96.83% @0.5%) | local | out_altertax_ifsub0.gdx |
| gtap7_10x7 | altertax | 1 | base,check,shock | 98 | 99.31% (96.81% @0.5%) | local | out_altertax_ifsub1.gdx |
| gtap7_15x10 | altertax | 0 | base,check,shock | 99 | 99.57% (98.19% @0.5%) | local | out_altertax_ifsub0.gdx |
| gtap7_15x10 | altertax | 1 | base,check,shock | 99 | 99.40% (97.92% @0.5%) | local | out_altertax_ifsub1.gdx |
| gtap7_3x4 | altertax | 0 | base,check,shock | 99 | 99.72% (96.79% @0.5%) | local | out_altertax_ifsub0.gdx |
| gtap7_3x4 | altertax | 1 | base,check,shock | 99 | 99.72% (96.46% @0.5%) | local | out_altertax_ifsub1.gdx |
| gtap7_20x41 | altertax | 0 | base | — | blocked: ref violates 37 own eqs | blocked | out_altertax_ifsub0.gdx (corrupt) |

## Pure-gtap (real-CES) multi-period SOLVE gate (PATH, local-only)

The non-altertax real-CES model solved base→check→shock in `mode="gtap"`, per ifSUB, vs the GAMS LOCAL `out_gtap_shock_ifsub{0,1}.gdx` (gtap7_3x3 local, gtap7_5x5 via NEOS). All four cases are at parity: the sluggish factor price pft was wrongly FIXED by fix_sluggish_pft (via a nonexistent xftflag Param), freezing it against the tariff shock. Freeing it (eq_xfteq.active guard) + removing the 3x3-hardcoded eq_pfyeq trim (the matcher squares automatically) closed gtap7_5x5 ifSUB=0 64.87→100% and improved every case (commit f570e32).

| dataset | kind | ifsub | phases | gap_min | gap_note | ci_status | ref |
|---|---|---|---|---|---|---|---|
| gtap7_3x3 | gtap_solve | 0 | base,check,shock | 99 | 100% (CHECK 100%) | local | out_gtap_shock_ifsub0.gdx |
| gtap7_3x3 | gtap_solve | 1 | base,check,shock | 98 | 99.4% (CHECK 100%) | local | out_gtap_shock_ifsub1.gdx |
| gtap7_5x5 | gtap_solve | 0 | base,check,shock | 99 | 100% (CHECK 100%) | local | out_gtap_shock_ifsub0.gdx (NEOS) |
| gtap7_5x5 | gtap_solve | 1 | base,check,shock | 99 | 99.59% (CHECK 100%) | local | out_gtap_shock_ifsub1.gdx (NEOS) |
