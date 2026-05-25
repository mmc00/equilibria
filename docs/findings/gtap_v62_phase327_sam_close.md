# GTAP v6.2 Phase 3.27 — SAM close (reduce prebalance residuals by ~98%)

**Date:** 2026-05-24
**Branch:** `gtap/v62-rollback`
**Builds on:** Phase 3.26 (dual NLP/MCP mode)

## TL;DR

Identified and fixed three calibration inconsistencies that caused
the v6.2 SAM to require ~1.17M USD of prebalance bake-in. After
fixes, max baseline residual dropped from **1.17e+06 → 3.74e+04**
(~98% reduction). NLP regression preserved (VIWS = +53.024% vs
Phase 3.23 +53.017%; Δ = 0.007pp).

**PATH still hits term_code=2 on shocked solve.** The SAM close
was a necessary structural improvement but not the root cause of
PATH's non-convergence. PATH's failure is from Jacobian
ill-conditioning at CES σ_m=4.64 + non-convex merit function
topology, independent of baseline imbalance magnitude.

## Three SAM imperfections fixed

### Fix #1: `eq_cgds_balance` residual (~1.17M USD per region)

The savings-investment identity
```
pcgds * qo[cgds, r] == y[r] - yp[r] - yg[r] + savf[r]
```
had a residual at benchmark because:
- `save_0[r]` was read from the HAR "SAVE" header (a savings aggregate)
- `qo[cgds, r]` was initialized from `vom[cgds, r]` = sum of intermediate
  inputs to the cgds sector (cost side)
- These differed by VDEP + DPGOV + DPPRIV components that the static
  closure doesn't model

**Fix** ([gtap_v62_calibration.py](src/equilibria/templates/gtap_v62/gtap_v62_calibration.py)):
```python
out.save_0[r] = out.vom.get((cgds_label, r), 0.0)  # force cost-side
out.savf_0[r] = yp_0 + yg_0 + save_0 - y_0          # rebalance savf
```

### Fix #2: `eq_qtm` residual (~65K USD per margin)

Margin demand `eq_qtm` excludes intra-region transport
(`amgm[m,i,r,r] = 0` in our calibration) but margin supply
`qtm_0 = sum_r VST(m,r)` included the diagonal. Discrepancy =
intra-region VTWR ~65K.

**Fix**:
```python
diag_vtwr = sum(b.vtwr.get((m_lbl, i, r, r), 0.0) for i in sets.i for r in sets.r)
out.qtm_0[m_lbl] = total_vst_full - diag_vtwr
# share_st still uses total_vst_full so it sums to 1 (for eq_ptmg)
```

### Fix #3: `eq_tax_revenue` residual (~720K USD per region — biggest!)

Phase 3.22 introduced `tax_revenue_0[r] = sum (V*A - V*M)`, but it
omitted the **implicit output tax** revenue. Phase 3.7's
reconciliation calibrates `to[i,r] = vom/vop - 1` (absorbs production-
side SAM gap), producing tax revenue `vom - vop` that needs to be in
`tax_revenue_0` too.

**Fix**:
```python
# TOUT: implicit output tax revenue from Phase 3.7's vom/vop wedge
tout = sum(
    out.vom.get((i, r), 0.0) - out.vop.get((i, r), 0.0)
    for i in sets.prod_comm
)
```

Plus: initialize `gdpmp[r]` and `rgdpmp[r]` to `y_0` (= factor_income
+ tax_revenue) instead of factor_income alone, so `eq_gdpmp:
gdpmp = y` holds at benchmark.

## Combined result

| State | max_abs baked | n_baked cells | Top residual eq family |
|:------|--------------:|--------------:|:------------------------|
| Pre-3.27          | 1.17e+06 | 43 | eq_cgds_balance |
| After Fix #1       | 7.20e+05 | 41 | eq_tax_revenue |
| After Fix #1+#2    | 7.20e+05 | 41 | eq_tax_revenue |
| **After all fixes** | **3.74e+04** | **39** | eq_market |

98% reduction in maximum imbalance magnitude.

## Solver impact

### IPOPT NLP (regression check)

```
Phase 3.23: VIWS food USA→EU = +53.017%
Phase 3.27: VIWS food USA→EU = +53.024%   (Δ = +0.007pp, negligible)
```

NLP regression preserved.

### PATH MCP (the question we set out to answer)

```
Pre-3.27 (Phase 3.26 MCP mode):
  BASELINE: term_code=2 residual=1.14e-03   ✓ (= baked-to-zero)
  SHOCKED:  term_code=2 residual=5.98e-01   ✗ VIWS=+0.796%

Phase 3.27 (with 98% smaller raw residuals):
  BASELINE: term_code=2 residual=1.14e-03   ✓ (same — baking still drives to zero)
  SHOCKED:  term_code=2 residual=5.98e-01   ✗ VIWS=+0.794%
```

Effectively identical PATH behavior. **The SAM close was not the
critical factor for PATH.**

### Bonus check: PATH without bake (bake_tolerance=0)

With baked SAM closer to natural (37K instead of 1.17M), can we drop
the bake entirely?

```
PATH no-bake:
  BASELINE: term_code=2 residual=3.77        ✗ (was 4.42 pre-3.27)
  SHOCKED:  residual=3.77 (no movement)      ✗ VIWS=+0.000%
```

Even with 98% smaller residual, PATH can't drive F(x_0) to zero
without baking — it gets stuck at the natural imbalance. So baking
remains necessary in our setup.

## Why PATH still fails despite SAM-close

The 98% reduction in baseline residuals **did NOT** translate to PATH
convergence on shocked. This rules out SAM imbalance as the root cause
of PATH's term_code=2.

The remaining causes (documented in earlier phases):

1. **Jacobian ill-conditioning at CES σ_m=4.64**: derivative spread
   ~5 orders of magnitude. PATH's Newton+line-search can't find a
   step that reduces the merit function in all directions
   simultaneously.

2. **Non-convex merit function topology**: ||F(x)||² with nested
   CES (top σ_d=2.4 × bottom σ_m=4.64 × CDE) has saddle points and
   local minima where PATH gets trapped.

3. **Pure equality formulation, not MCP-native**: GTAPinGAMS uses
   explicit complementarity pairs (price ⊥ excess supply, qty ⊥
   zero-profit) with sign constraints. Our formulation is pure
   F(x) = 0. PATH can theoretically handle this but is less robust
   without the complementarity structure.

## What changed (files)

`src/equilibria/templates/gtap_v62/gtap_v62_calibration.py`:
- Fix #1: `save_0[r] = vom[cgds, r]`
- Fix #2: `qtm_0[m] -= diag_VTWR`
- Fix #3: `tout_calibration` includes `vom - vop` from implicit output tax

`src/equilibria/templates/gtap_v62/gtap_v62_model_equations.py`:
- `gdpmp / rgdpmp` initialized to `y_0` (not factor_income alone)

`scripts/gtap_v62/validate_v62_parity.py`:
- Cleaner comment on bake_tolerance choice

## Status

| Aspect | Status |
|--------|--------|
| SAM imperfections reduced 98% | ✓ |
| NLP regression preserved | ✓ (Δ 0.007pp) |
| MCP closure mismatch=0 | ✓ |
| PATH baseline residual → 0 | ✓ (with baking) |
| PATH shocked convergence | ✗ unchanged (residual ~6e-1) |
| Diagnosis: PATH issue is Jacobian/topology, not SAM | ✓ confirmed |
| Production solver (IPOPT NLP) | ✓ +53.024% sub-1% parity |

## Next steps (Phase 3.28+ if pursued)

The three remaining PATH issues require deeper refactor:

1. **Variable scaling per-row Jacobian normalization** inside PATH
   solver — manual scaling beyond the diagonal pre-scaling we already
   have. ~2-3 days.
2. **GTAPinGAMS-style explicit complementarity reformulation**:
   replace each market clearing with `price ≥ 0 ⊥ excess_supply ≤ 0`.
   1-2 weeks.
3. **MPSGE-style auxiliary-price formulation**: full rewrite. 1-2
   months.

For research-grade v6.2 parity, **IPOPT NLP remains the production
solver** (sub-1% relative gap vs GEMPACK Gragg-multi on 4 datasets).
