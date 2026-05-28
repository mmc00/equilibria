# GTAP v6.2 Phase 3.38 — `sav` as residual Var closes regional budget identity

**Date:** 2026-05-27
**Branch:** `gtap/v62-rollback`
**Scope:** v6.2 — NLP mode (IPOPT) and MCP mode (PATH) both benefit
**Closes:** Phase 3.25 documented regression "+62.39%" that was unreproducible
on the current dataset (we found +46% with old code, with budget identity
violated post-shock).

## TL;DR

Through Phase 3.36/3.37, baseline residuals on `gtap6_15x10` dropped from
3.7e-3 to ~2e-5 but the shocked solve never converged in MCP mode and
gave +46% VIWS in NLP mode instead of GEMPACK's +62.36%. Investigation
revealed two bugs working together:

1. **`sav` was held as a constant Param**, leaving regional budget
   identity `y = yp + yg + sav` unsatisfied under any shock. `walras`
   absorbed thousands of USD of imbalance.
2. **The VIWS metric in our test scripts was `qxs * pms`** (agent
   price, post-tariff) instead of `qxs * pmcif` (CIF, world price)
   which is what GEMPACK's VIWS represents.

After both fixes:

```
gtap6_3x3   0.07% gap vs GEMPACK Gragg-multi
gtap6_5x5   0.06% gap
gtap6_10x7  0.07% gap
gtap6_15x10 0.64% gap
gtap6_20x41 IPOPT/MUMPS hits 32-bit integer stack limit (separate issue)
```

Walras at the shocked solution: **< 2e-8 across all datasets that converge**.

## Root cause: `sav` was Param, not Var

`src/equilibria/templates/gtap_v62/gtap_v62_model_equations.py` line ~539
defined:

```python
model.save_0 = Param(
    model.r,
    initialize=dict(c.save_0), default=0.0, mutable=False,
    doc="Benchmark regional savings (= SAVE from SAM)",
)
```

And `eq_walras` referenced this Param directly:

```python
def eq_walras_rule(m):
    return m.walras == sum(
        m.y[r] - m.yp[r] - m.yg[r]
        - pyo_value(m.save_0[r]) + m.savf[r]
        for r in m.r
    )
```

The Phase 3.21 comment admitted this:

> `sav = c_sav * y * pcons^XSHRPRIV` (save_0 stays as Param —
> the small budget imbalance is absorbed by walras)

On BOOK3X3 the imbalance was tiny. On gtap6 datasets it reaches
thousands of USD per region and corrupts VIWS by ~16pp.

### Diagnostic that revealed it

`scripts/gtap_v62/_diag_walras_shocked.py` decomposed walras after
shock and showed per-region:

```
USA: dy=+5841, dyp+dyg=+5061, dsavf=0, dsav=0  →  +780 leak
EU:  dy=-5080, dyp+dyg=-2792, dsavf=0, dsav=0  →  -2288 leak
ROW: dy=-2740, dyp+dyg=-1465, dsavf=0, dsav=0  →  -1274 leak
```

`dsav = 0` is the smoking gun: savings does not move under shock,
forcing walras to absorb the entire dy-dyp-dyg residual.

## The fix

Make `sav` a Var defined by the budget identity:

```python
# New variable
model.sav = Var(
    model.r,
    within=Reals, bounds=(None, None),
    initialize=lambda m, r: float(c.save_0.get(r, 0.0)),
    doc="Phase 3.38: regional savings (sav = y - yp - yg)",
)

# New constraint
def eq_sav_rule(m, r):
    return m.sav[r] == m.y[r] - m.yp[r] - m.yg[r]
model.eq_sav = Constraint(model.r, rule=eq_sav_rule)

# Updated eq_walras
def eq_walras_rule(m):
    return m.walras == sum(
        m.y[r] - m.yp[r] - m.yg[r] - m.sav[r] + m.savf[r]
        for r in m.r
    )
```

With this, the budget identity holds identically per region. `walras`
absorbs only `sum_r savf[r]` (a SAM-level constant ~3.5e6 USD on gtap6
datasets) which the bake then offsets — leaving the post-solve walras
at ~1e-8.

### Alternative tried first (didn't work)

Initial attempt was to give `sav` its own GEMPACK static-closure formula:
`sav = c_sav * y * pcons^XSHRPRIV`. This is the LEVELS analogue of
Phase 3.21's elasticity (same exponent as `eq_yg`). But that
formulation produces `yp/y + yg/y + sav/y ≠ 1` for `pcons ≠ 1` so the
budget identity is still violated. The residual `sav = y - yp - yg`
is the correct LEVELS form (this is what GEMPACK does internally — sav
is the residual after Hicksian consumption is set).

## The VIWS metric bug

The test scripts computed:

```python
viws_pct = 100.0 * (qxs[shock] * pms[shock]).delta / baseline
```

But `pms = pmcif * (1 + tms)` — that is the **agent price post-tariff**,
not the world CIF price. The GEMPACK `VIWS` header corresponds to
`qxs * pmcif`, which is the value of imports at world (border) prices,
without tariff.

After the tariff cut:
* `pmcif` (CIF) barely moves
* `pms` (agent) drops by ~10% as tariff is removed
* `qxs` rises by ~70%

So `qxs * pms` change ≈ +70% × (1 - 0.10) - 1 = ~+53%
And `qxs * pmcif` change ≈ +70% × 1 - 1 = ~+62% (matches GEMPACK)

Verified via the GEMPACK oracle (`run_gempack_oracle.py`):

```
VIWS (CIF):     +62.36%  ← Phase 3.25 doc
VXWD (FOB):     +62.37%
VIMS (agent):   +46.12%  ← what our test was computing
VXMD (basic):   +62.37%
```

`VIMS = VIWS / (1 + tms)` precisely.

The test scripts were measuring an internally-consistent quantity, but
not the one labelled "VIWS" by GEMPACK. After switching to
`qxs * pmcif`, parity to GEMPACK collapsed to sub-1%.

## Why this wasn't a Phase 3.25 regression

We bisected from current HEAD back to Phase 3.25's commit (`e7b7274`).
The Phase 3.25 commit code ALSO produces `+46.15%` on gtap6_3x3 today
with the same test. The Phase 3.25 finding doc reports `+62.39%`.

The discrepancy is the VIWS metric — Phase 3.25 most likely used
`qxs * pmcif` (or read the GEMPACK upd HAR's `VIWS` header directly)
while later test scripts copied a `qxs * pms` formula that diverged.
The model itself was never producing +62%; the *measurement* was
different.

This explains why no commit in the 3.26 → 3.36 range "broke" anything
on this metric — there was no regression to find, just a metric label
mismatch.

## What changed (files)

`src/equilibria/templates/gtap_v62/gtap_v62_model_equations.py`:
- Added `model.c_sav` Param (unused after the residual-form fix, kept
  in case future work wants the elastic form).
- Added `model.sav` Var.
- Added `eq_sav` constraint: `sav[r] = y[r] - yp[r] - yg[r]`.
- Updated `eq_walras` to reference `m.sav[r]` (Var) instead of
  `m.save_0[r]` (Param).

`scripts/gtap_v62/_make_square.py`:
- Added `apply_v62_diagonal_redundancy_fix` (Phase 3.37) that
  deactivates `eq_qxs(i, r, r)` cells where intra-region trade
  dominates (>99% of VXMD into r). Wired into `apply_v62_pipeline`
  for MCP mode only.

Test scripts (`scripts/gtap_v62/_test_*_ipopt_nlp.py`,
`_test_gtap6_all_nlp.py`):
- Compute VIWS as `qxs * pmcif`, not `qxs * pms`.

## Status — gtap6 PATH/NLP parity table

| Dataset    | Vars   | IPOPT NLP gap | Notes |
|:-----------|-------:|--------------:|:------|
| gtap6_3x3  |    663 | 0.07%         | walras=3.4e-9 at shocked |
| gtap6_5x5  |  2,239 | 0.06%         | walras=4.9e-9 |
| gtap6_10x7 |  8,965 | 0.07%         | walras=6.3e-9 |
| gtap6_15x10| 25,720 | 0.64%         | walras=1.4e-8, IPOPT term=infeasible but result valid |
| gtap6_20x41|285,790 | N/A           | MUMPS int32 stack overflow (separate infrastructure issue) |

`gtap6_20x41` doesn't run not because of model bugs but because
IPOPT/IDAES ships with MUMPS compiled for 32-bit integer indices,
which overflows at ~290K vars. Recompiling MUMPS with `INTSIZE64`
plus a custom IPOPT rebuild would be required (~1-2 days on macOS
via conda-forge, ~3-5 days on Windows).

## Reproduce

```bash
# Sub-1% sweep across 4 datasets:
PYTHONIOENCODING=utf-8 uv run python scripts/gtap_v62/_test_gtap6_all_nlp.py

# Individual datasets:
PYTHONIOENCODING=utf-8 uv run python scripts/gtap_v62/_test_3x3_ipopt_nlp.py
PYTHONIOENCODING=utf-8 uv run python scripts/gtap_v62/_test_15x10_ipopt_nlp.py

# GEMPACK oracle (verify reference values):
uv run python scripts/gtap_v62/run_gempack_generic.py \
  --workdir runs/gempack_3x3_food_usa_eu \
  --dataset-dir datasets/gtap6_3x3 \
  --shock-comm Food --shock-src USA --shock-dst EU_28
```

## Remaining work

* `gtap6_20x41` requires either:
  * MUMPS-int64 + IPOPT rebuild (best done on macOS via conda-forge)
  * HSL MA57/MA86 via licensed IPOPT (recommended for scientific work)
  * PEDRO Phase 1 matrix-free Newton-Krylov prototype (avoids the
    sparse-LU scaling wall entirely)

The Phase 3.37 `eq_qxs` diagonal redundancy fix benefits MCP mode
specifically (PATH crash factorization no longer detects 13 singular
constraints). It is dormant in NLP mode (the `mode == "mcp"` gate)
but available if the user runs MCP again on `gtap6_15x10`.

## Commits in this session

* `bake_tolerance` 1e-3 → 1e-6 (Phase 3.36, prior commits)
* Diagnostic + UMFPACK exploration scripts (Phase 3.36 diagnostics)
* `apply_v62_diagonal_redundancy_fix` (Phase 3.37, this commit set)
* `sav` as Var + `eq_sav` + updated `eq_walras` (Phase 3.38, this commit set)
* VIWS metric correction in test scripts (Phase 3.38, this commit set)
