# GTAP v6.2 Phase 3.34 — Substep count scales with dataset size

**Date:** 2026-05-25
**Branch:** `gtap/v62-rollback`
**Scope:** v6.2 only
**Builds on:** Phase 3.33 (PATH SHOCKED breakthrough on small datasets)

## TL;DR

The Phase 3.33 cocktail (bake at baseline, NO rebake, chained warm-start)
needed only **5 substeps** for gtap6_3x3 and gtap6_5x5 to converge.
For gtap6_10x7 with 4× more variables, **10 substeps** are needed —
and PATH converges term_code=1 on EVERY substep.

## Result for gtap6_10x7

```
Baseline:     tc=1 r=5.15e-08  ← term_code=1, residual essentially zero
Substep  1/10: tc=1 r=5.03e-07 VIWS= +4.876%
Substep  2/10: tc=1 r=6.37e-07 VIWS=+10.039%
Substep  3/10: tc=1 r=5.70e-07 VIWS=+15.509%
Substep  4/10: tc=1 r=5.13e-07 VIWS=+21.308%
Substep  5/10: tc=1 r=4.61e-07 VIWS=+27.458%
Substep  6/10: tc=1 r=5.85e-07 VIWS=+33.986%
Substep  7/10: tc=1 r=5.96e-07 VIWS=+40.916%
Substep  8/10: tc=1 r=6.71e-07 VIWS=+48.280%
Substep  9/10: tc=1 r=6.61e-07 VIWS=+56.108%
Substep 10/10: tc=1 r=6.52e-07 VIWS=+64.434%

TOTAL: 169 seconds, FINAL VIWS = +64.434%
GEMPACK Gragg-multi 2-4-6 ref: +64.391%
Gap: +0.04 pp (0.07% relative) ⭐
```

20-substep run gives identical answer (+64.434%) in 126 s (less work
per step balances out the extra rebuilds).

## Three-way verified table

| Dataset    | Vars   | Substeps | PATH tc | PATH VIWS | GEMPACK ref | Gap |
|:-----------|-------:|---------:|--------:|----------:|------------:|----:|
| gtap6_3x3  |    663 |    5     |   1 ✓   |  +62.400% |   +62.359%  | +0.04 pp |
| gtap6_5x5  |  2,239 |    5     |   1 ✓   |  +64.754% |   +64.553%  | +0.20 pp |
| gtap6_10x7 |  8,965 |   10     |   1 ✓   |  +64.434% |   +64.391%  | +0.04 pp |

**Three different sizes, three sub-0.25 pp gaps to GEMPACK, all
term_code=1 on every PATH solve.** PATH on v6.2 is fully working for
gtap6 datasets up to ~9K variables.

## Why 5 substeps fail on 10x7 but 10 succeeds

Watching the substep-by-substep residual progression:

**5-substep run** (each step = 20% of full shock):
```
Substep 1/5 α=0.20: tc=2 r=3.17e-04 VIWS=+10.039%
Substep 2/5 α=0.40: tc=2 r=3.46e-04 VIWS=+21.308%
Substep 3/5 α=0.60: tc=2 r=7.85e-02 VIWS=+21.537%  ← residual jump!
Substep 4/5 α=0.80: tc=2 r=8.73e-02 VIWS=+21.917%  ← stuck
Substep 5/5 α=1.00: tc=2 r=9.97e-02 VIWS=+22.032%  ← stuck
```

The first two 20%-substeps converge. The third tries to traverse
from α=0.4 to α=0.6 (a +20% shock advance) and falls outside PATH's
basin of attraction.

**10-substep run** (each step = 10% of full shock):
Every substep stays inside the basin. PATH converges term_code=1 each
time. Total wall time is similar because more substeps mean more
calls but each Newton solve is faster.

**The empirical rule**: substep size ≤ 10% of full shock keeps PATH
in basin for 10K-variable models. For smaller models (3x3, 5x5),
20% substeps are fine.

## Why this matters

It establishes a **scaling rule for PATH on v6.2**:

| Vars     | Substep size | Time per substep | Notes |
|---------:|-------------:|-----------------:|:------|
|     ~700 |  20% (5 subs) |        0.05 sec | Trivial |
|   ~2000  |  20% (5 subs) |        1-3 sec | OK |
|   ~9000  |  10% (10 subs)|       16-30 sec | THIS phase |
|  ~25000  |  ~5% (20 subs)|       1-3 min  | TBD (15x10 test pending) |
| ~290000  |  ~2% (50 subs)|     ~10-30 min | likely not feasible (gtap6_20x41) |

## What changed (files)

`scripts/gtap_v62/validate_v62_parity.py`:
- Default `--homotopy-steps` argument unchanged (user controls)
- Comment in code explains scaling rule

No code changes in this phase — all the prior phases (3.27 SAM close,
3.28 conditional fixing, 3.30 auto-drop, 3.33 no-rebake-chained-warm-start)
remain as they were. **Phase 3.34 is a TUNING discovery, not a code
change**.

## Recommendation

For PATH solves on v6.2:

```python
# Pick substep count based on dataset size:
n_substeps = max(5, n_vars // 1000)
# e.g. 663 vars → 5 substeps
#      2,239 vars → 5 substeps
#      8,965 vars → ~9 substeps
#      25,720 vars → ~26 substeps
```

CLI: `--homotopy-steps 10` for medium datasets (~10K vars).

## Status — 33+ phases of v6.2 work culminate

| Aspect | Status |
|--------|--------|
| PATH baseline converges | ✓ gtap6_3x3, 5x5, 10x7 (tc=1, r ~ 1e-7) |
| PATH SHOCKED converges | ✓ all three with sub-0.25 pp parity vs GEMPACK |
| Three-way parity (PATH ≈ IPOPT ≈ GEMPACK) | ✓ verified |
| BOOK3X3 (legacy SAM) | ✗ structural SAM defects beyond Phase 3.30 |
| gtap6_15x10 (25K vars) | TBD — likely works with ~25 substeps but long |
| gtap6_20x41 (290K vars) | not feasible without MA48-class linear solver |

For production: **IPOPT NLP is fastest** (~1-20 s per solve). **PATH
is now a working alternative** with the same parity, for users who
need MCP-native solutions. Both produce the same equilibrium.
