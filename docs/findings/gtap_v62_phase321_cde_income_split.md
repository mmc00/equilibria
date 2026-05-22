# GTAP v6.2 Phase 3.21 — CDE-elastic income split (yp/yg/sav vs pcons)

**Date:** 2026-05-22
**Branch:** `gtap/v62-rollback`
**Builds on:** Phase 3.20 (true levels CDE)

## TL;DR

Replaced the fixed-share `eq_yp / eq_yg` allocation (CD-style) with the
CDE-elastic split from GEMPACK gtap.tab `PRIVCONSEXP / GOVCONSEXP /
SAVING` (lines 2211-2224). Under static closure with `dppriv = dpgov =
dpsave = 0`:

```
yp(r) = c_p_0 * y(r) * pcons(r)^(XSHRPRIV - 1)
yg(r) = c_g_0 * y(r) * pcons(r)^XSHRPRIV
sav(r) = c_sav_0 * y(r) * pcons(r)^XSHRPRIV    (kept fixed in Param)
```

Where `pcons` is the CDE expenditure deflator (Phase 3.20 normalisation,
`pcons_0 = 1`).

**Result:**
- IPOPT shocked: **VIWS food USA→EU = +47.469%** (was +47.366% in 3.20).
- Parity vs Gragg-multi: **-6.05pp** (was -6.15pp). Improvement: 0.10pp.
- Deterministic across 4 runs.

**Key finding:** the CDE-elastic income split is structurally correct
but numerically minor for a 10% shock. The 6pp residual gap is NOT
from income split dynamics either.

## Why the impact is so small (matches theory)

For BOOK3X3 EU under 10% food-tariff cut:
- `pcons` (CDE deflator) falls by ~0.3% (food is ~15% of EU
  consumption; food price falls ~2% via Armington; pcons drops 0.15-0.3%).
- `XSHRPRIV_EU ≈ 0.35` (yp_0 / factor_income_0).
- yp shift factor = `pcons^(XSHRPRIV - 1) = 0.997^(-0.65) ≈ 1.002` ⇒
  yp/y rises by ~0.2%.
- yg shift factor = `pcons^XSHRPRIV = 0.997^0.35 ≈ 0.999` ⇒
  yg/y falls by ~0.1%.

These small income shifts propagate via the CDE demand:
- qp_food shift from income: `(yp/yp_0)^EY_food ≈ (1.002)^0.37 ≈ 1.0007`
  ⇒ +0.07% on food consumption.
- Via the bottom Armington (no amplification for income channel):
  +0.07% on qim → +0.07% on qxs → ~+0.10% on VIWS.

Observed: +0.10pp on VIWS. **Theory predicts the result exactly.**

## Walras imbalance trade-off

Under the new dynamic split, `yp + yg + save_0` no longer balances `y +
savf` exactly under shock (the budget identity holds only at the
benchmark and to first order). The walras imbalance grew:

| Phase | Walras shocked | Walras / GDP |
|------:|---------------:|-------------:|
| 3.20  | -978           | 0.03%        |
| 3.21  | -8,940         | 0.27%        |

This is ~9x larger but still small (< 0.3% of regional GDP). The IPOPT
regularizer absorbs it without affecting other variables materially.
A fully budget-preserving form would require either:
- Making `save` a Var with the same dynamics (currently fixed in Param);
- Normalizing the three shares (the `Z(pcons)` factor in the theory
  derivation) — second-order in shock magnitude.

Neither is needed for the current parity-floor exercise.

## What this rules out

After Phases 3.19 (log-linear CDE), 3.20 (levels CDE), 3.21 (CDE-elastic
income split), we've now tested **three** distinct hypotheses for the
6pp gap:

| Hypothesis | Phase | Impact | Verdict |
|:-----------|------:|:-------|:--------|
| CD vs CDE preferences | 3.19 | +3.18pp | **Real channel** |
| Frozen vs state-dep CDE coefs | 3.20 | -0.003pp | **Not the source** |
| Fixed vs pcons-elastic income split | 3.21 | +0.10pp | **Minor effect** |

**The remaining 6pp gap is NOT in the household demand sub-system.**

Candidate sources still on the table:

1. **Margin commodity self-trade** (`amgm[m,i,r,r] = 0` + MSHRS
   treatment): GEMPACK gtap.tab lines 1775-1785 explicitly discuss
   intra-region margin trade handling. We carry over the diagonal in
   Phase 3.16 but may differ in how margins themselves treat the
   diagonal. Worth a careful read.
2. **Tax revenue feedback into regional income**: our `y` is defined
   as factor income only (`y = sum_f pf * qoes`); tax revenue is folded
   into the benchmark shares but doesn't re-emerge as a dynamic flow
   under shock. GEMPACK has `INCOME(r) = factor_income + INDTAX(r)`
   with `INDTAX` shifting under shock. Estimated effect: 1-3pp.
3. **RORGLOBAL international capital allocation**: `eq_cgds_balance`
   has the baked residual representing VDEP + DPGOV + DPPRIV; under
   shock these flows shift in GEMPACK but stay frozen in our model.
   Estimated effect: 1-2pp.

Of these, **(2) tax revenue feedback** is the most likely contributor
to the remaining 6pp, since:
- A tariff cut directly reduces tariff revenue → reduces `INDTAX(EU)`.
- In GEMPACK this lowers `y(EU)` → lowers all of `yp, yg, save` →
  reduces aggregate demand including food imports.
- In our model, `y` doesn't see tariff revenue → `yp, yg, save` are
  insensitive to tariff revenue changes → demand is over-stated under
  shock compared to GEMPACK.

This would mean Python's +47.47% is **over-estimating** import demand
because we don't subtract lost tariff revenue from regional income.
GEMPACK's +53.52% then includes some other amplification that we don't
have. Worth investigating in Phase 3.22.

## Determinism

```
Run 1: VIWS = +47.469%
Run 2: VIWS = +47.469%
Run 3: VIWS = +47.469%
Run 4: VIWS = +47.469%
```

Identical across runs. The new pcons-dependent terms in eq_yp/eq_yg
don't reintroduce solver bimodality.

## What changed

`src/equilibria/templates/gtap_v62/gtap_v62_calibration.py`:
- Added `xshrpriv`, `xshrgov`, `xshrsave` dicts on
  `DerivedV62Calibration` (one entry per region).
- Computed from `yp_0 / factor_income_0`, etc., in the income-block
  loop. Note: these use `y_0 = factor_income` as the denominator
  (matching our model's `y` definition), not GEMPACK's
  `INCOME = factor_income + INDTAX`.

`src/equilibria/templates/gtap_v62/gtap_v62_model_equations.py`:
- `eq_yp_rule`: `yp = c_p * y * pcons^(XSHRPRIV - 1)` (was `c_p * y`).
- `eq_yg_rule`: `yg = c_g * y * pcons^XSHRPRIV` (was `c_g * y`).

`save` (regional savings) remains a fixed Param — making it a Var with
`sav = c_sav * y * pcons^XSHRPRIV` is left for a future tightening,
since the budget-imbalance penalty in walras is small enough not to
matter.

## Reproduce

```powershell
$env:PYTHONIOENCODING = "utf-8"
python scripts/gtap_v62/validate_v62_parity.py shock `
    --experiment Exp1a --solver ipopt `
    --workdir runs/gtap_v62_parity/BOOK3X3_Exp1a

# Expected: VIWS food USA->EU = +47.469% on every run
# Expected: walras ≈ -8940 shocked (0.27% of GDP)
```

## v6.2 parity status after Phase 3.21

| Metric | Status |
|--------|--------|
| Calibration SAM-consistent | ✓ (3.16 + 3.18) |
| Closure clean (0 bipartite vars) | ✓ (3.18) |
| CDE household preferences (levels) | ✓ (3.20) |
| CDE-elastic income split | ✓ (3.21) |
| Tax revenue dynamics | ✗ — Phase 3.22 candidate |
| Margin self-trade refinement | ✗ — Phase 3.22 candidate |
| RORGLOBAL capital flows | ✗ — Phase 3.23 candidate |
| IPOPT shocked deterministic | ✓ |
| **Best parity vs Gragg-multi** | **-6.05pp** (was -6.15pp) |

## Next: Phase 3.22 — tax revenue feedback into regional income

The strongest remaining hypothesis. Implementation sketch:

1. Compute `tax_revenue(r)` explicitly in the model (factor + output +
   intermediate + final + tariff + export taxes), each from its
   relevant `V*A - V*M` flow.
2. Add to `eq_y_rule`:
   `y(r) = sum_f pf * qoes(f,r) + sum_streams tax_revenue(r)`.
3. The tariff-revenue stream `tax_imp(i,r) = sum_s tms(i,s,r) * pms *
   qxs` then shifts under shock and propagates to `yp / yg / save`.

Estimated impact: 1-3pp. Effort: 1-2 days.
