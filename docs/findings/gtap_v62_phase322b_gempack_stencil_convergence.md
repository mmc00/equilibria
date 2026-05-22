# GTAP v6.2 Phase 3.22b — GEMPACK stencil convergence check

**Date:** 2026-05-22
**Branch:** `gtap/v62-rollback`
**Builds on:** Phase 3.22 (tax revenue feedback)

## TL;DR

Ran GEMPACK Gragg-multi with progressively finer Richardson stencils to
test whether the 6pp gap vs Phase 3.22's levels Newton solve is a
Richardson extrapolation residual or a missing economic channel.

**Result: GEMPACK Gragg-multi converges identically across all stencils
tested.** The 6pp gap is real — there is a missing economic channel in
our Python implementation.

## Stencil sweep

```
Stencil                   VIWS food USA→EU    VIMS food USA→EU
Johansen-1 (linearized)   +41.5363%           +31.5363%
Gragg 2-4-6               +53.5166%           +38.1650%
Gragg 4-8-12              +53.5166%           +38.1650%
Gragg 2-4-6 auto-accuracy +53.5166%           +38.1650%
Gragg 12-24-48            +53.5166%           +38.1650%
```

Identical to the 4th decimal across 2-4-6 and 12-24-48 stencils. GEMPACK
is firmly converged at +53.52%.

## What this rules out

**Hypothesis A (now ruled out):** the 6pp gap is Richardson
extrapolation overshoot in Gragg-multi vs the true levels equilibrium.
- 6× refinement (2-4-6 → 12-24-48) changes the answer by 0.0000%.
- The +53.52% is GEMPACK's converged answer, not an artefact.

## What this means

Our Phase 3.22 Python solution at +47.22% **is** a valid levels-Newton
equilibrium — but it's the equilibrium of a slightly different
economic system than what GEMPACK is solving. There's a structural
channel in GEMPACK's model that our Python doesn't have.

Candidates (in declining likelihood of being the dominant 6pp):

1. **Margin commodity self-trade refinement** (`amgm[m,i,r,r] = 0`):
   gtap.tab lines 1775-1785 comment that in aggregated SAMs the
   margin demand for intra-region trade is treated differently. We
   include it (Phase 3.16); GEMPACK may exclude it via `amgm = 0` for
   self-trade. The svces commodity (margin in BOOK3X3) has large
   diagonal flows — could account for 2-4pp.

2. **Production-side state-dependent CES shares**: GEMPACK's Gragg
   recomputes ALL coefficients per substep (not just CDE). Our top-nest
   CES for production (σ_t between VA and intermediate composite,
   σ_va within VA) uses calibrated alpha shares frozen at benchmark.
   For sectors absorbing the food shock differently across regions,
   the differential could matter. Could account for 2-3pp.

3. **A specific CES parameter or formulation mismatch**: less likely
   given how systematically we've checked, but not impossible.

## Decision point

Three options going forward:

A. **Phase 3.23 — implement `amgm[m,i,r,r] = 0` margin exclusion**:
   small focused change in `_make_square.py` and the eq_pwmg /
   eq_market handling. ~1 day. Could close 2-4pp.

B. **Phase 3.24 — production-side state-dep CES via fixed-point loop**:
   outer loop that recalibrates alpha shares after each shocked solve.
   ~3-5 days. Could close 2-3pp.

C. **Close the branch at Phase 3.22 + this finding**: accept the 6pp
   as a documented residual representing a research-grade gap between
   our levels Newton solve and GEMPACK's Gragg-multi convention. Total
   parity floor: VIWS -6.29pp vs Gragg, +5.69pp vs Johansen-1. This is
   the cleanest stopping point with full structural alignment of the
   household income side.

## Reproduce

```powershell
$env:PYTHONIOENCODING = "utf-8"
# Run GEMPACK at the highest stencil already tested:
python scripts/gtap_v62/run_gempack_exp1a_multistep.py `
    --workdir runs/gtap_v62_oracle/BOOK3X3_Exp1a_GB_12_24_48 `
    --variant GB_12_24_48
# Expected: same VIWS = +53.5166% as the lower-stencil runs.
```

## Files

`scripts/gtap_v62/run_gempack_exp1a_multistep.py`:
- Added `GB_12_24_48` and `GB_24_48_96` variants to `METHOD_BLOCKS`.

`runs/gtap_v62_oracle/BOOK3X3_Exp1a_GB_12_24_48/`:
- New GEMPACK oracle run with 12-24-48 stencil.
- Solution file: `Exp1a_GB_12_24_48_sol.har`.
- Updated SAM: `Exp1a_GB_12_24_48-upd.har`.
