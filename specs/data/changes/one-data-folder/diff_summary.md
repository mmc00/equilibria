# Phase 2b — what the regenerated GAMS references change

Every GAMS result exists in two variants (#187): `fbep=header` (basedata.har's
FBEP/FTRV injected after getData.gms's `fbep = 0`, FBEP with the sign cal.gms
expects) and `fbep=zero` (the official getData.gms). All of them were rerun once
with the final generators (after the final review moved nus333's HAR-to-GDX
mapping into the model's tools/, 276de253): 87 of 89 jobs recorded; the two
missing are gtap7_5x5 `gams_nlp_altertax_ifsub0` (#192, below). Every result
file of that rerun is byte-identical to the two runs before it (e85f9107,
7164d9ff), NEOS `out.gdx` included; only the NEOS job metadata (`neos.log`,
`neos_job.json`) and the listings' remote work paths change.

Rewritten after the FBEP sign fix (3ba4f88). The first version of this file
compared header results built with the wrong sign, so it listed 130+ differing
symbols per result; with the sign fixed the two variants carry the same
equilibrium.

## header vs zero

Both variants must give the same equilibrium: they only book the factor subsidy
differently. `gdxdiff RelEps 1e-6` of `out.gdx`, symbols outside the tax
bookkeeping (`fbep`, `fctts`, `fcttx`, `ftrv`, `sam`, `ytax*`):

| Result | Outside tax bookkeeping |
|---|---|
| gtap7_10x7 `gams_mcp_altertax_ifsub0` | none |
| gtap7_10x7 `gams_mcp_altertax_ifsub1` | none |
| gtap7_10x7 `gams_mcp_gtap_ifsub0` | none |
| gtap7_10x7 `gams_mcp_gtap_ifsub1` | none |
| gtap7_10x7 `gams_nlp_altertax_ifsub0` | equation marginals only (pvaeq, pxeq) |
| gtap7_10x7 `gams_nlp_altertax_ifsub1` | 78 variables, 33 equations (altertax shock, #192) |
| gtap7_10x7 `gams_nlp_gtap_ifsub0` | none |
| gtap7_10x7 `gams_nlp_gtap_ifsub1` | none |
| gtap7_15x10 `gams_mcp_altertax_ifsub0` | 46 variables, 51 equations (altertax shock, #192) |
| gtap7_15x10 `gams_mcp_altertax_ifsub1` | 69 variables, 64 equations (altertax shock, #192) |
| gtap7_15x10 `gams_mcp_gtap_ifsub0` | none |
| gtap7_15x10 `gams_mcp_gtap_ifsub1` | none |
| gtap7_20x41 `gams_nlp_gtap_ifsub1` | equation marginals only (axpeq, lambdaioeq, lambdandeq, lambdavaeq, pfeq, pfteq, savfeq, uheq, xfteq, zconseq) |
| gtap7_3x3 `gams_mcp_altertax_ifsub0` | none |
| gtap7_3x3 `gams_mcp_altertax_ifsub1` | none |
| gtap7_3x3 `gams_mcp_gtap_ifsub0` | none |
| gtap7_3x3 `gams_mcp_gtap_ifsub1` | none |
| gtap7_3x3 `gams_nlp_altertax_ifsub0` | equation marginals only (pvaeq, pxeq) |
| gtap7_3x3 `gams_nlp_altertax_ifsub1` | equation marginals only (pndeq, pvaeq, pxeq, xpeq) |
| gtap7_3x3 `gams_nlp_gtap_ifsub0` | none |
| gtap7_3x3 `gams_nlp_gtap_ifsub1` | equation marginals only (chifeq) |
| gtap7_3x4 `gams_mcp_altertax_ifsub0` | none |
| gtap7_3x4 `gams_mcp_altertax_ifsub1` | none |
| gtap7_3x4 `gams_nlp_altertax_ifsub0` | equation marginals only (pvaeq, pxeq) |
| gtap7_3x4 `gams_nlp_altertax_ifsub1` | equation marginals only (pvaeq, pxeq, xpeq) |
| gtap7_5x5 `gams_mcp_altertax_ifsub0` | none |
| gtap7_5x5 `gams_mcp_altertax_ifsub1` | none |
| gtap7_5x5 `gams_mcp_gtap_ifsub0` | none |
| gtap7_5x5 `gams_mcp_gtap_ifsub1` | none |
| gtap7_5x5 `gams_nlp_altertax_ifsub1` | equation marginals only (pvaeq, pxeq) |
| gtap7_5x5 `gams_nlp_gtap_ifsub0` | equation marginals only (arenteq) |
| gtap7_5x5 `gams_nlp_gtap_ifsub1` | equation marginals only (arenteq, kapEndeq, pabseq, pfacteq, rorceq) |
| 9x10 `gams_mcp_gtap_ifsub1` | none |

- **none**: the variants agree to 1e-6 everywhere outside the bookkeeping.
- **equation marginals only** (NLP results): the equation levels agree (checked:
  differences at most 1e-15); only the NLP duals differ, which depend on the
  solver path, not on the solution.
- **altertax shock, #192**: base and check agree to 1e-7; the shock differs, up
  to 2% (`pva`, gtap7_15x10 ifSUB=1, MEX). Altertax makes every production nest
  Cobb-Douglas (sigma = 1), and GAMS writes the CES price equations in power
  form (`pvaeq`, `pxeq`, `pndeq`: price**(1-sigma) = ...), which become `1 = 1`:
  the solution is not unique and each run stops somewhere along the free
  direction. Measured: with `sigmap`, `sigmand`, `sigmav` = 0.99 header and zero
  agree to 3e-8 in every period, both for gtap7_10x7 nlp ifSUB=1 (local IPOPT;
  7e-4 at sigma = 1) and for gtap7_15x10 mcp ifSUB=1 (NEOS 24339862/24339863;
  2% at sigma = 1). The model fix (the Cobb-Douglas limit at sigma = 1) is #192.

Not recorded, as expected: gtap7_5x5 `gams_nlp_altertax_ifsub0`, both variants
(model status 6 header, 7 zero, at check). Same cause, #192.

### gtap7_20x41

Recorded for the first time, as NLP with CONOPT on NEOS, `zcons` and `zconseq`
scaled by their calibrated level (`regen_gams_reference.py --scale-zcons`; in
the record's `notes` and the model record). PATH never solves the 20x41 shock
(time limit; a 4 h run stalls at residual 1.44; a 1% shock ends in status 5),
and unscaled CONOPT stopped at a false point with `fbep=zero`: `u(CHE)` -12.8%
against -0.9% with header. Cause, in the data: `default.prm` puts SUBPAR at the
GTAP floor (1e-6) in 8 of CHE's 20 goods, CHE being the only region with any.
GAMS sets `bh = subpar`, so every `zcons(CHE)` is near 1e-10, below the solver
tolerances. The original model sets `scaleopt = 1` but no `.scale`. Scaled,
header and zero agree to 2e-7 in every region; `u(CHE)` is -0.9063% in both
(GEMPACK -0.8645%, the same order as the GAMS-GEMPACK gap elsewhere: EU_28
-0.9069% vs -0.8807%).

### .nl files

The .nl (CONVERT) of every phase differs: FBEP enters the factor-price equations.

| Result | .nl files that differ |
|---|---|
| gtap7_10x7 `gams_nl_altertax_ifsub0` | gams_altertax.nl, gams_base.nl, gams_check.nl, gams_shock.nl |
| gtap7_10x7 `gams_nl_altertax_ifsub1` | gams_altertax.nl, gams_base.nl, gams_check.nl, gams_shock.nl |
| gtap7_15x10 `gams_nl_altertax_ifsub0` | gams_altertax.nl, gams_base.nl, gams_shock.nl |
| gtap7_15x10 `gams_nl_altertax_ifsub1` | gams_altertax.nl, gams_base.nl, gams_shock.nl |
| gtap7_3x3 `gams_nl_altertax_ifsub0` | gams_altertax.nl, gams_base.nl, gams_check.nl, gams_shock.nl |
| gtap7_3x3 `gams_nl_altertax_ifsub1` | gams_altertax.nl, gams_base.nl, gams_check.nl, gams_shock.nl |
| gtap7_3x4 `gams_nl_altertax_ifsub0` | gams_altertax.nl, gams_base.nl, gams_shock.nl |
| gtap7_3x4 `gams_nl_altertax_ifsub1` | gams_altertax.nl, gams_base.nl, gams_shock.nl |
| gtap7_5x5 `gams_nl_altertax_ifsub0` | gams_altertax.nl, gams_base.nl, gams_check.nl, gams_shock.nl |
| gtap7_5x5 `gams_nl_altertax_ifsub1` | gams_altertax.nl, gams_base.nl, gams_check.nl, gams_shock.nl |

## Old fixtures vs the new header results

`tests/fixtures/gtap7/<ds>/out_gtap_shock_ifsub<k>.gdx` and
`tests/fixtures/gtap7_altertax/<ds>/out_altertax_ifsub<k>.gdx` against the MCP
header result that replaces each. Every pair differs well beyond `RelEps 1e-8`,
in every period, and for three reasons:

1. **The FBEP sign.** The old fixtures injected FBEP with the HAR's sign, so the
   subsidy was counted as a tax: `pfa*xf = evfb + ftrv - fbep`, and the base was
   no equilibrium in subsidised cells (the FBEP oracle fails all 12 subsidised
   cells of the old 3x3 header). The largest gaps sit there: `fctts` (up to 1.2,
   the flipped rate), then `pfa`, `pf`, `nd`, `xp` in subsidised activities
   (e.g. EU_28 Land Rice; GBR SkLab Livestock, FBEP/EVFB = 0.59; `nd` EU_28
   Crops 92% in 10x7).
2. **`pwmg` on routes without margins** (expected since phase 2a): it differs
   by 1.0 on no-margin routes (e.g. 3x3 USA Svces USA), where no equation pins it.
3. **Altertax's free direction (#192)**, by its pattern (not measured with
   sigma = 0.99 for this pair): `pva`/`va` of USA Chemicals, an unsubsidised
   cell, differ by 100% in the gtap7_15x10 altertax ifSUB=0 shock only.

`tests/fixtures/gtap7/gtap7_20x41/out_gtap_shock_ifsub1.gdx` has no MCP
counterpart: it was never a GAMS result (9ef28cbc made it with Python, Newton-TR
+ MUMPS, as a calibration seed). The 20x41 GAMS reference is the NLP above.

## nus333 and 9x10

They had GAMS results outside `data/`, and now they are recorded like every
other dataset (Task 6).

- **nus333** (`gams_mcp_gtap_ifsub0_fbep-zero`, local PATH, under its 1000-row limit): one
  variant, since its data have no FBEP header. Its `out.gdx` is byte-identical
  to the `output/nus333_neos/out.gdx` it replaces (md5 2e4d348a; `gdxdiff
  RelEps 1e-8`: no differences). `diff_nus333_full` gives base 100% (1304
  cells), shock 99.62% (1305 cells), identical on origin/main.
- **9x10** (`gams_mcp_gtap_ifsub1_fbep-{header,zero}`, NEOS; recorded jobs
  24345626 header / 24345650 zero, from the final rerun; first runs
  24339989/24339990; `comp.gms`, ifSUB=1). The old
  `COMP_generated.csv`/`COMP.csv` had no generator anywhere (maintainer
  2026-11-01: regenerate). Base and check of `fbep-zero`, the official run,
  match `COMP.csv` year 2 (`u(Oceania)` 0.99999997 vs 0.999999971, `pft(Oceania,
  Land)` 1.0000019415 vs 1.000001931). The shock is not comparable: the old
  year 3 multiplied prices by 1.5 (a homogeneity test), not the +10% tariff
  shock. Header vs zero: see the table above. The 9x10 consumers (the auto seed,
  `path_capi` warm start, `run_gtap --compare-gams`) read `fbep-zero`; on
  10101 cells the seed is identical to the CSV one, except `facty`, which the
  CSV loader derived as `-ytax` and the GDX carries as GAMS's own `factY`.
