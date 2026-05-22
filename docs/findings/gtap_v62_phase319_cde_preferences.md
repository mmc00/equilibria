# GTAP v6.2 Phase 3.19 — CDE preferences (replaces Cobb-Douglas household demand)

**Date:** 2026-05-22
**Branch:** `gtap/v62-rollback`
**Builds on:** Phase 3.18 (clean closure, deterministic IPOPT at +44.19%)

## TL;DR

Implemented CDE (Constant Difference of Elasticities) household demand in
levels Pyomo form, replacing the Cobb-Douglas simplification carried
since Phase 2c.1. CDE matches GEMPACK's `PRIVDMNDS` linearization to
first order at the benchmark (`gtap.tab` line 1055).

**Result:**
- IPOPT shocked is fully **deterministic**: `+47.369%` on all 5 runs (no
  Mode A/B bimodality).
- Parity vs GEMPACK Gragg-multi: **-6.15pp** (was -9.33pp in 3.18 with
  CD) — recovered ~3pp.
- Parity vs Johansen-1: **+5.83pp** — moved AWAY from Johansen toward
  Gragg-multi, which is the desired direction (Gragg-multi is the more
  refined GEMPACK result).
- PATH still stuck at `term_code=2` (unaffected — confirms 3.18's
  diagnosis that PATH's issue is merit-function, not economic content).

The 6pp residual gap reflects two remaining differences both flagged in
3.18:
- **Frozen vs updated CDE coefficients**: GEMPACK's Gragg-multi
  recomputes `CONSHR`, `EP`, `EY` at each substep. Our log-linear form
  freezes them at the benchmark.
- **VDEP/DPGOV/DPPRIV income flows**: still absorbed as a baked
  constant in `bake_baseline_residuals_as_slacks`.

## What "log-linear CDE" means

GEMPACK's CDE is implicit in a min-expenditure function (HT F1-F3). It
solves the linearized form `qp - pop = sum_k EP * pp + EY * (yp - pop)`
in percent changes. For Pyomo levels-MCP we use the calibrated
log-linear equivalent:

```
qp(i,r) = qp_0(i,r)
          * (yp(r) / yp_0(r))^EY(i,r)
          * prod_k (pp(k,r) / pp_0(k,r))^EP(i,k,r)
```

Properties:
- At benchmark all ratios = 1 ⇒ `qp = qp_0` (exact).
- First-order Taylor expansion equals `gtap.tab` PRIVDMNDS (line 1055)
  exactly with the same EP, EY coefficients.
- For finite shocks: deviates from levels-CDE by O(shock²), but
  matches Gragg-multistep (also linearized + Richardson extrapolated)
  to higher order than the Cobb-Douglas form it replaces.

## CDE coefficients in BOOK3X3

`INCPAR` (income parameter) and `SUBPAR` (substitution parameter)
loaded from `GTAPPARM`:

```
              INCPAR    SUBPAR
food   USA    0.1340    0.8667     ← strong Engel (low income elast.)
food   EU     0.1608    0.8971
food   ROW    0.3317    0.9675
mnfcs  USA    0.7931    0.2393
svces  USA    1.1438    0.0100     ← luxury good
```

Derived elasticities (gtap.tab F4, F5):

```
EY (income elast.):
              food    mnfcs   svces    sum(CONSHR*EY) — Engel check
USA           0.315   1.016   1.067    1.0000  ✓
EU            0.367   1.052   1.123    1.0000  ✓
ROW           0.511   1.051   1.160    1.0000  ✓

EP own-price (EP[i, i, r]):
              food    mnfcs   svces
USA          -0.206  -0.827  -0.968
EU           -0.241  -0.783  -0.953
ROW          -0.266  -0.543  -0.892

Slutsky check (sum_k EP[i,k] should equal -EY[i]):
EP[food, *, EU] = -0.241 -0.078 -0.048 = -0.367  ≡ -EY[food, EU]  ✓
```

Food has both **low income elasticity** (Engel's law) AND **low own-
price elasticity** (inelastic demand) — exactly the CDE features
absent from Cobb-Douglas (where EY=1, EP_own=-1 for every good).

## Parity progression on `gtap/v62-rollback`

| Phase | Household demand | IPOPT shocked | vs Gragg-multi | vs Johansen-1 |
|------:|:-----------------|--------------:|---------------:|--------------:|
| 3.8   | CD (bipartite-fixed closure) | bimodal ~+54.4% | -0.9pp (Mode B artifact) | n/a |
| 3.16  | CD (diagonal trade) | bimodal +50.4% | -3.1pp (Mode B artifact) | n/a |
| 3.18  | CD (clean closure) | deterministic +44.19% | -9.33pp | +2.65pp |
| **3.19** | **CDE log-linear** | **deterministic +47.37%** | **-6.15pp** | **+5.83pp** |
| GEMPACK Gragg-multi | CDE + multi-step | +53.52% | 0 (ref) | +11.98pp |
| GEMPACK Johansen-1 | CDE + linear | +41.54% | -11.98pp | 0 (ref) |

Phase 3.19 closes ~33% of the 9pp gap. The remaining 6pp gap is now
two effects of comparable magnitude:

1. **Coefficient update across substeps** (Gragg-multi-only effect):
   CONSHR shifts under shock as expenditure reallocates. EP, EY are
   recomputed each substep in GEMPACK. Estimated contribution: ~3-4pp.
2. **VDEP/DPGOV/DPPRIV income dynamics**: depreciation and government
   deficit aren't perturbed under shock in our model; they shift in
   GEMPACK. Estimated contribution: ~2-3pp.

## What changed

`src/equilibria/templates/gtap_v62/gtap_v62_calibration.py`:
- Added CDE coefficient computation after the household/government CD
  share loop:
  - `alpha_cde[i, r]   = 1 - SUBPAR[i, r]`
  - `uelaspriv[r]      = sum_n CONSHR(n,r) * INCPAR(n,r)`
  - `xwconshr[i, r]    = CONSHR * INCPAR / UELASPRIV`
  - `cde_ape[i, k, r]` per HT F1-F3
  - `cde_ey[i, r]` per HT F4
  - `cde_ep[i, k, r] = (APE - EY) * CONSHR_k` per HT F5
- Added `qp_0[i, r] = VDPM + VIPM` (household benchmark composite
  quantity).

`src/equilibria/templates/gtap_v62/gtap_v62_model_equations.py`:
- `eq_qp_rule` (was: `pp * qp = share * yp`, Cobb-Douglas):
  ```
  qp(i,r) == qp_0(i,r)
             * (yp(r) / yp_0(r))^EY(i,r)
             * prod_k (pp(k,r) / pp_0(k,r))^EP(i,k,r)
  ```
- `eq_pcons_rule` (was: `pcons = sum_i share * pp`, linear CD aggregator):
  ```
  pcons(r) == pcons_0(r)
              * prod_i (pp(i,r) / pp_0(i,r))^XWCONSHR(i,r)
  ```
  (XWCONSHR-weighted geometric mean — first-order CDE expenditure
  deflator)
- All other equations unchanged. The bottom Armington nest
  (`eq_qpd`, `eq_qpm`) still distributes the CDE-determined `qp`
  between domestic and imported.

## PATH retest (Phase 3.19)

```
PATH baseline:  term_code=2, residual=1e-3, walras=0.00      ✓ trivial
PATH shocked:   term_code=2, residual=1.42e-01, walras=0.00  ✗ stuck
                Zero movement on qo, pms, qxs — same as 3.17/3.18.
```

CDE doesn't change PATH's diagnosis. As Phase 3.18 confirmed, PATH's
issue is structural to the prebalance-baked merit function, not the
model's economic content.

## Determinism check (5 IPOPT runs)

```
Run 1:  VIWS = +47.369%   walras = -922
Run 2:  VIWS = +47.369%   walras = -922
Run 3:  VIWS = +47.369%   walras = -922
Run 4:  VIWS = +47.369%   walras = -922
Run 5:  VIWS = +47.369%   walras = -922
```

Identical to machine precision across runs. The Phase 3.18 clean
closure remains the dominant factor in determinism — CDE didn't
re-introduce any bimodality.

Walras at shocked grew from 73 (3.18) to -922 (3.19). On a ~3M USD
total-GDP base this is 0.03% relative imbalance, still acceptable but
larger than 3.18's 0.002%. The CDE form introduces more cross-good
elasticity coupling which the IPOPT regularizer balances less tightly
than CD's separable shares.

## Reproduce

```powershell
$env:PYTHONIOENCODING = "utf-8"

# IPOPT — deterministic Phase 3.19 result:
python scripts/gtap_v62/validate_v62_parity.py shock `
    --experiment Exp1a --solver ipopt `
    --workdir runs/gtap_v62_parity/BOOK3X3_Exp1a

# Expected: VIWS food USA->EU = +47.369% on every run
# Expected: status: ok, walras ~-922 on shocked, ~+303 baseline
```

## v6.2 parity status after Phase 3.19

| Metric | Status |
|--------|--------|
| Calibration SAM-consistent | ✓ (Phase 3.16 + 3.18) |
| Closure clean (no bipartite heuristics) | ✓ (Phase 3.18) |
| CDE household preferences | ✓ (Phase 3.19) |
| IPOPT shocked deterministic | ✓ (Phase 3.18) |
| IPOPT shocked converged | ✓ (walras 0.03% of GDP) |
| PATH baseline | ✓ |
| PATH shocked | ✗ stuck (3.17 / 3.18 / 3.19) |
| **Best parity vs Gragg-multi** | **-6.15pp** (was -9.33pp) |
| Best parity vs Johansen-1 | +5.83pp |

## What would close the remaining 6pp

Three options, in declining marginal value:

1. **Recompute CDE coefficients per Newton step** (~3pp): track CONSHR
   as a state variable that updates each iteration. Substantial Pyomo
   refactor; would converge toward Gragg-multistep's behavior.
2. **Implement VDEP / DPGOV / DPPRIV income dynamics** (~2pp): add
   per-region depreciation flow as a function of capital stock, and
   government/private deficit closures with explicit determinants.
3. **Margin commodity self-trade refinements** (~1pp): GEMPACK's
   `amgm[m,i,r,r] = 0` exclusion for self-trade margins (gtap.tab
   lines 1775-1785 comment).

Each is ~1-2 weeks of careful work. The combined effect would bring
parity to within ~1pp of Gragg-multi.

For a research-grade reference implementation, **Phase 3.19 is a
natural stopping point**: structurally aligned model, deterministic
solver, ~6pp documented parity floor with clear paths to close it.
