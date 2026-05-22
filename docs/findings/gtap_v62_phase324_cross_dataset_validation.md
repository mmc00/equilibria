# GTAP v6.2 Phase 3.24 — Cross-dataset validation of the Phase 3.23 model

**Date:** 2026-05-22
**Branch:** `gtap/v62-rollback`
**Builds on:** Phase 3.23 (MKTCLIMP, parity -0.50pp on BOOK3X3 Exp1a)

## TL;DR

Validated the Phase 3.23 model on three datasets of progressively
larger dimension. The model builds, calibrates, applies a clean
square closure (0 unmatched cells), and IPOPT-solves baseline and
shocked equilibria on every dataset. Results are economically
sensible across all sizes.

| Dataset  | Dimension | Factors | Vars  | Constraints | Closure | Result |
|:---------|:---------:|:-------:|:-----:|:-----------:|:-------:|:-------|
| BOOK3X3  | 3×3       | 3       | 728   | 626         | 0       | ✓ VIWS +53.0% (vs Gragg-multi +53.5%) |
| ACORS3X3 | 3×3       | 5       | 788   | 660         | 0       | ✓ VIWS +50.6%, qim +0.6% |
| ASA7X5   | 7×5       | 5       | 3,900 | 3,572       | 0       | ✓ VIWS +62.7%, qim +0.1% |

The implementation **scales correctly to larger dimensions**: ASA7X5
is ~5× the size of BOOK3X3 and still produces a clean closure with
mismatch=0 and a converged IPOPT solution.

## What each dataset tests

**BOOK3X3 (3 commodities × 3 regions, 3 factors)**: the canonical
GEMPACK v6.2 sample dataset. We have a known GEMPACK Gragg-multi
reference (+53.52%) and our parity is -0.50pp. Reproduce with
`scripts/gtap_v62/validate_v62_parity.py`.

**ACORS3X3 (3 commodities × 3 regions, 5 factors)**: same dimension
as BOOK3X3 but with the **modern factor set** (Land / UnskLab /
SkLab / Capital / NatRes — 5 factors vs BOOK3X3's 3). Tests that the
model handles arbitrary factor sets. Shock: tms[Food, SSA, EU] cut
10%. Result: VIWS +50.6%, qim +0.6%.

**ASA7X5 (7 commodities × 5 regions, 5 factors)**: a substantially
larger Africa-focused dataset. 3,572 free variables, 148 baked SAM
residual cells (vs 43 in BOOK3X3). Tests that the closure logic
(bipartite matching + identity equations) generalizes correctly.
Shock: tms[FOOD, SAFRICA, EUNION] cut 10%. Result: VIWS +62.7% (the
larger response reflects the higher benchmark tariff rate ~40% vs
BOOK3X3's ~37%).

## Closure scaling

The bipartite-matching closure (`apply_v62_closure_and_square`)
produces mismatch=0 on every dataset, confirming that:

- Phase 3.18's explicit identity equations (`eq_cgds_balance`,
  `eq_pwmg` trivial-fix, `eq_qfe` no-factor-use, `pva` CGDS, etc.)
  generalize across factor sets.
- Phase 3.23's `eq_qim` MKTCLIMP form is sound regardless of
  dimension — it always squares to one cell per (commodity, region).

The SAM prebalance scales naturally:

| Dataset  | Baked cells | max\|residual\| |
|:---------|:-----------:|:---------------:|
| BOOK3X3  | 43          | 1.17e+06        |
| ACORS3X3 | 57          | 2.20e+06        |
| ASA7X5   | 148         | 2.20e+06        |

These are mostly the eq_cgds_balance VDEP+DPGOV constant residual
(per region) plus eq_qtm intra-region VTWR entries (per
margin commodity).

## Numerical robustness

All three datasets produce IPOPT-deterministic results when run via
`validate_v62_parity.py` (for BOOK3X3) or `test_cross_dataset.py`
(for ACORS3X3 and ASA7X5). The Phase 3.18 closure cleanup eliminated
the Mode A/B bimodality observed in earlier phases — every run of
every dataset gives the same answer to machine precision.

Note: a minimal regularizer (used in `test_cross_dataset.py`) can
trigger IPOPT restoration on BOOK3X3 specifically because its more
imbalanced SAM (eq_cgds_balance residual = 1.17M) interacts poorly
with the tiny anchored objective near the calibration init. For
BOOK3X3 use the production validator (`validate_v62_parity.py`)
which applies the regularizer with appropriate scaling.

## What this confirms

1. **Phase 3.23 is generic**, not BOOK3X3-specific. The MKTCLIMP fix
   and all preceding phase improvements (CDE preferences, income
   split, tax revenue) apply equally to larger datasets.
2. **The 0-mismatch closure is structural**, not coincidental. It
   holds across factor-set variation (3 vs 5 factors) and dimension
   scaling (9 commodity-region cells → 35 in ASA7X5).
3. **Solve time scales gracefully**: BOOK3X3 takes ~10s, ASA7X5
   takes ~30s on IDAES IPOPT. Both deterministic.

## Cross-dataset parity vs GEMPACK Gragg-multi (2-4-6)

After this writeup we ran GEMPACK Gragg-multi on all three datasets
using the same 10% tariff cut, via the new
`scripts/gtap_v62/run_gempack_generic.py` driver.

| Dataset  | Shock cell             | Metric | GEMPACK %  | Python %   |    Δpp |
|:---------|:-----------------------|:-------|-----------:|-----------:|-------:|
| BOOK3X3  | food / USA → EU        | VIWS   |   +53.5166 |   +53.0170 | -0.500 |
|          |                        | VIMS   |   +38.1650 |   +37.7150 | -0.450 |
| ACORS3X3 | Food / SSA → EU        | VIWS   |   +50.3029 |   +50.6020 | +0.299 |
|          |                        | VIMS   |   +35.2726 |   +35.5410 | +0.268 |
| ASA7X5   | FOOD / SAFRICA → EUNION| VIWS   |   +62.4214 |   +62.7470 | +0.326 |
|          |                        | VIMS   |   +46.1793 |   +46.4730 | +0.294 |

**All three datasets are at sub-1% relative gap vs GEMPACK:**

| Dataset  | VIWS gap | % of GEMPACK magnitude |
|:---------|---------:|-----------------------:|
| BOOK3X3  | -0.500pp |                  0.93% |
| ACORS3X3 | +0.299pp |                  0.59% |
| ASA7X5   | +0.326pp |                  0.52% |

Notes:
- **Gap sign varies** (BOOK3X3 below, ACORS3X3 / ASA7X5 above
  GEMPACK). This is consistent with the residual being higher-order
  numerical (frozen-coefficient CES nonlinearity at large σ_m) rather
  than a systematic structural bias.
- **Best parity on ASA7X5 (0.52%)** — the largest dataset. The
  Phase 3.23 MKTCLIMP fix is structurally clean.

## What this does NOT establish

- **PATH support on larger datasets**: only tested IPOPT. PATH was
  already known to be stuck on BOOK3X3 shocked (Phase 3.18) and
  would likely behave similarly on the others.

## Files

`scripts/gtap_v62/test_cross_dataset.py`:
- New script that builds, calibrates, and shock-solves the v6.2 model
  on multiple datasets, reporting key metrics. Designed for
  regression-style coverage — running the script verifies the model
  still builds and solves across all configured datasets.

`scripts/gtap_v62/run_gempack_generic.py`:
- New generic GEMPACK runner that accepts `--dataset-dir`,
  `--shock-comm`, `--shock-src`, `--shock-dst`. Generalises the
  BOOK3X3-only `run_gempack_exp1a_multistep.py`. Used to generate
  the GEMPACK oracle references for ACORS3X3 and ASA7X5.

## Reproduce

```powershell
$env:PYTHONIOENCODING = "utf-8"

# Run all three Python solves:
python scripts/gtap_v62/test_cross_dataset.py
python scripts/gtap_v62/validate_v62_parity.py shock `
    --experiment Exp1a --solver ipopt `
    --workdir runs/gtap_v62_parity/BOOK3X3_Exp1a

# Generate GEMPACK Gragg-multi references for ACORS3X3 and ASA7X5:
python scripts/gtap_v62/run_gempack_generic.py `
    --workdir runs/gtap_v62_oracle/ACORS3X3_Shock1 `
    --dataset-dir c:/runGTAP375/ACORS3X3 `
    --shock-comm Food --shock-src SSA --shock-dst EU `
    --exp-name ACORS3X3_Shock1
python scripts/gtap_v62/run_gempack_generic.py `
    --workdir runs/gtap_v62_oracle/ASA7X5_Shock1 `
    --dataset-dir c:/runGTAP375/ASA7X5 `
    --shock-comm FOOD --shock-src SAFRICA --shock-dst EUNION `
    --exp-name ASA7X5_Shock1
```

## v6.2 branch status: complete

After 24 phases on `gtap/v62-rollback`, the implementation:

- ✓ Loads v6.2 HAR data (SETS.HAR, basedata.har, Default.prm)
- ✓ Calibrates SAM-consistent with diagonal trade
- ✓ Implements true CDE preferences (Hanoch-Hertel levels)
- ✓ Implements CDE-elastic regional income split
- ✓ Endogenous tax revenue feedback
- ✓ MKTCLIMP imported-composite market clearing
- ✓ Clean square closure (0 bipartite-fixed cells)
- ✓ IPOPT-deterministic solves (no bimodality)
- ✓ Parity vs GEMPACK Gragg-multi: -0.50pp on BOOK3X3 Exp1a
- ✓ Builds and solves on BOOK3X3, ACORS3X3, ASA7X5 (3,900 vars)

**The branch is ready to be merged to main as a research-grade
v6.2 implementation.**
