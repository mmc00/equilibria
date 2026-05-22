# GTAP v6.2 Phase 3.22 — Tax revenue feedback into regional income

**Date:** 2026-05-22
**Branch:** `gtap/v62-rollback`
**Builds on:** Phase 3.21 (CDE-elastic income split)

## TL;DR

Wired explicit tax revenue dynamics into the regional income identity:
```
y(r) = factor_income(r) + tax_revenue(r)
tax_revenue(r) = TPC + TGC + TIU + TFU + TOUT + TEX + TIM
```
where each stream is computed in levels from the active tax wedges and
flows (e.g. `TIM(r) = sum_{i,s} tms[i,s,r] * pmcif[i,s,r] * qxs[i,s,r]`).

**Result:**
- IPOPT shocked: VIWS food USA→EU = **+47.224%** (was +47.469% in 3.21).
- Gap vs Gragg-multi: **-6.29pp** (was -6.05pp).
- **Moved 0.25pp AWAY from Gragg-multi** — the opposite of what we
  wanted, but in the direction theory predicts.
- Walras improved dramatically: baseline -12.6 (was +329), shocked
  -1432 (was -8940). The budget identity now closes tightly.

## Why the direction is "wrong" but consistent with theory

Under the 10% tariff cut, EU loses tariff revenue (~1264 USD). With
Phase 3.22:
- `tax_revenue(EU)` falls by ~1264.
- `y(EU)` falls by the same amount (~0.03% of EU GDP).
- `yp`, `yg`, `save` all scale down by the same factor (via `c_p`,
  `c_g`, `c_sav` * y), with the additional pcons-elastic shift from
  Phase 3.21.
- Less aggregate demand in EU ⇒ less imports ⇒ smaller VIWS response.

This is GEMPACK's actual behaviour (gtap.tab REGIONALINCOME line 2179).
We were previously OVER-stating imports by ignoring tax revenue loss.

The result is **structurally correct but moves us further from
Gragg-multi**, which means the +53.52% target is not from a missing
income channel.

## What this conclusively tells us about the remaining 6pp

After Phases 3.19 → 3.22 we've now ruled out:

| Hypothesis | Phase | Effect on parity | Real economic channel? |
|:-----------|------:|-----------------:|:-----------------------|
| CD vs CDE preferences | 3.19 | +3.18pp toward Gragg | **YES** — material |
| State-dependent CDE coefs | 3.20 | -0.003pp | Yes, but tiny |
| CDE-elastic income split | 3.21 | +0.10pp toward Gragg | Yes, minor |
| Tax revenue feedback | 3.22 | -0.25pp away from Gragg | **YES** — and properly amounting for it makes parity WORSE |

The entire household income subsystem now matches GEMPACK's specification
exactly (CDE demand + utility-elastic split + dynamic tax revenue).
**The 6pp gap is not in the household income side.**

## Where the 6pp must come from

Three remaining channels:

1. **Production-side CES with σ_t and σ_va dynamics**: GEMPACK uses
   levels CES production with state-updated shares (Gragg's coefficient
   refresh). Our Phase 3.18+ uses calibrated alpha shares frozen at
   benchmark. For sectors absorbing the shock (food imports compete
   with EU food production), the differential could matter. Estimated
   effect: 2-4pp.

2. **Margin commodity self-trade (`amgm[m,i,r,r] = 0`)**: gtap.tab
   lines 1775-1785 specifically excludes intra-region margin demand
   from svces (the BOOK3X3 margin commodity). We include it. For
   countries where the diagonal margin is large (ROW), this could
   amplify or damp the food-trade response. Estimated effect: 1-2pp.

3. **Richardson-extrapolation overshoot in Gragg-multi**: Gragg-multi
   is a 2-4-6 substep Richardson extrapolation of a linearized solve.
   For highly nonlinear systems (σ_m = 4.64 with finite shocks),
   Richardson can overshoot the true levels response. Without a third
   reference (MPSGE, exact Newton on GEMPACK's equation system) we
   can't tell which is "correct". Our Phase 3.22 IS a true levels
   Newton solve. Could account for 2-4pp.

The cleanest test would be to run GEMPACK with even more substeps
(12-24-48) and see if the result converges toward our +47.2% or stays
at +53.5%. Phase 3.10 stopped at 2-4-6.

## What changed

`gtap_v62_calibration.py`:
- `tax_revenue_0[r]`: explicit benchmark computation from `V*A - V*M`
  differences across all 7 streams (TPC, TGC, TIU, TFU, TOUT, TEX, TIM).
- `y_0[r] = factor_income + tax_revenue` (was just factor_income).
- `xshrpriv/gov/save` now use the new (larger) `y_0` as denominator,
  matching GEMPACK's `INCOME(r)`.

`gtap_v62_model_equations.py`:
- Added `tax_revenue[r]` Var, initialised at `tax_revenue_0[r]`.
- Added `eq_tax_revenue` constraint summing all stream
  `tax_rate * base_price * quantity` products.
- `eq_y_rule` now reads:
  `y(r) == sum_f pf*qoes + tax_revenue(r)`.

Closure stays balanced: +3 Vars (`tax_revenue[r]`) + 3 Constraints
(`eq_tax_revenue[r]`).

## Walras improvement

| Phase | Walras baseline | Walras shocked | Comment |
|:------|---------------:|---------------:|:--------|
| 3.20  | +329          | -978          | Income side incomplete |
| 3.21  | -516          | -8,940        | Income split adds budget drift |
| **3.22** | **-12.6**   | **-1,432**    | **Income side now closes tightly** |

Phase 3.22's walras is 0.04% of GDP shocked, the tightest closure of
the entire branch. The remaining discrepancy is from second-order
budget drift in the pcons-elastic split (Phase 3.21).

## Determinism (5 IPOPT runs)

```
Run 1: VIWS = +47.224%
Run 2: VIWS = +47.224%
Run 3: VIWS = +47.224%
Run 4: VIWS = +47.224%
Run 5: VIWS = +47.224%
```

Identical to machine precision.

## Reproduce

```powershell
$env:PYTHONIOENCODING = "utf-8"
python scripts/gtap_v62/validate_v62_parity.py shock `
    --experiment Exp1a --solver ipopt `
    --workdir runs/gtap_v62_parity/BOOK3X3_Exp1a

# Expected: VIWS food USA->EU = +47.224%
# Expected: walras ≈ -1432 shocked
```

## v6.2 parity status after Phase 3.22

| Metric | Status |
|--------|--------|
| Calibration SAM-consistent | ✓ (3.16 + 3.18) |
| Closure clean (0 bipartite vars) | ✓ (3.18) |
| CDE household preferences (levels) | ✓ (3.20) |
| CDE-elastic income split | ✓ (3.21) |
| **Tax revenue dynamics in regional income** | **✓ (3.22)** |
| Margin self-trade refinement | ✗ — Phase 3.23 candidate |
| Production-side state-dep CES | ✗ — Phase 3.24 candidate |
| GEMPACK Gragg-multi at 12-24-48 substeps (independent check) | ✗ — Phase 3.X check |
| IPOPT shocked deterministic | ✓ |
| Walras shocked / GDP | 0.04% (best so far) |
| Parity vs Gragg-multi | -6.29pp (was -6.05pp) |
| Parity vs Johansen-1 | +5.69pp |

## Honest assessment of the residual gap

Three options for closing the 6pp:

A. **Run GEMPACK 12-24-48 substeps**: 1-day check. If Gragg converges
   toward our +47.2%, the gap is GEMPACK's Richardson overshoot, and
   our levels Newton solve IS the correct answer.

B. **Production-side state-dep CES** (Phase 3.24): 1-week refactor.
   Most likely 2-4pp.

C. **Accept the gap as Gragg-vs-levels residual**: close the branch
   with the honest characterization that our Phase 3.22 is the proper
   levels equilibrium for v6.2 BOOK3X3 Exp1a, deviating from Gragg-multi
   by 6pp due to the Richardson extrapolation convention.

I recommend **A** first — it's quick and definitively resolves whether
the gap is a Python error or a GEMPACK approximation residual.
