# GTAP v6.2 Phase 3.36 — `bake_tolerance` must scale with dataset size

**Date:** 2026-05-26
**Branch:** `gtap/v62-rollback`
**Scope:** v6.2 only
**Status:** **Real improvement on baseline residual** under both lottery
outcomes. PATH on gtap6_15x10 baseline still depends on bipartite-matching
"luck" (see "Bipartite matching lottery" below).

## TL;DR

PATH on `gtap6_15x10` (25,720 variables) was stuck at
`term_code=2 r=3.7e-3` after 70+ major iterations. Lowering
`bake_tolerance` from `1e-3` to `1e-6` reduces PATH's baseline
residual by ~100× regardless of which bipartite matching is
selected — under the "good" matching to ~`5e-8` (tc=1), under
the "bad" matching to ~`1.97e-5` (tc=2).

## Empirical PATH baseline outcomes on gtap6_15x10

Ten observed runs with `bake_tolerance=1e-6`:

| Variant | nnz | tc | residual | majors | time | source |
|:--------|----:|---:|---------:|-------:|-----:|:-------|
| 1e-3 (old default) | 901 baked | 2 | 3.70e-3 | ≥70 | 979s | failed runs |
| Lottery win 1 | 97,655 | 1 | 5.33e-8 | 13 | 295s | baseline_tc1 |
| Lottery win 2 | 97,655 | 1 | 5.33e-8 | 13 | 297s | baseline_tc1 replay |
| Lottery win 3 | 97,639 | 1 | 2.21e-7 | 15 | 270s | baseline_tc1 today |
| Lottery loss 1 | 97,635 | 2 | 1.98e-5 | 3 | 310s | baseline_tc1 today (run 2) |
| Lottery loss 2 | 97,639 | 2 | 1.98e-5 | 12 | 540s | shocked baseline |
| Lottery loss 3 | 97,635 | 2 | 1.98e-5 | 11 | 543s | lower_bake at 1e-6 |
| Deterministic (eq-side) | 97,628 | 2 | 1.98e-5 | 17 | 563s | with PYTHONHASHSEED=0 |
| Deterministic (var-side) | 97,647 | 2 | 1.98e-5 | 9 | 361s | with PYTHONHASHSEED=0 |

Observed lottery success rate: 3/9 ≈ 33%. Sample is small; do not
take the rate as precise.

## Why `bake_tolerance=1e-6` matters

The prebalance bake (Phase 3.8) absorbs constraint residuals ABOVE
its tolerance, leaving smaller ones active. Larger datasets like
`gtap6_15x10` have many cells with residuals in the band
`1e-5 – 1e-3` (specifically `eq_qim`: 63 cells, `eq_pfe`: 48 cells,
`eq_pmcif`: 1500 cells). With tolerance `1e-3` all of them stayed
active; their cumulative 2-norm is `~3.7e-3` — exactly the floor
PATH was hitting.

Tolerance `1e-6` absorbs ~158 more cells as baked constants
(`baked` count goes `901 → 1059`). PATH now sees a system with
only floor-level (sub-`1e-6`) residual at the baseline, and either:
* Reaches `tc=1 r=O(1e-7)` (lottery win), OR
* Hits a second floor at `r~1.98e-5` (lottery loss).

Both outcomes are strictly better than the old `r=3.7e-3`.

## Bipartite matching lottery

The MCP closure (`scripts/gtap_v62/_make_square.py`) uses
`nx.bipartite.maximum_matching` (Hopcroft-Karp) to pair equations to
variables. The unmatched variables get FIXED at their current value.

Python sets are involved in three places:

```python
eq_nodes = {n for n, d in G.nodes(...) if d.get("bipartite") == 0}
var_nodes = {n for n, d in G.nodes(...) if d.get("bipartite") == 1}
matched_vars = {n for n in match if n in var_nodes}
unmatched_vars = var_nodes - matched_vars
```

Python hash randomization (default since 3.3 for security) makes
set iteration order vary between processes. NetworkX's Hopcroft-Karp
implementation also uses sets for BFS layers. Result: `maximum_matching`
picks a DIFFERENT equivalent matching every process invocation.

Different matchings:
* Fix DIFFERENT specific cells of the same variable families
* Produce DIFFERENT Jacobian sparsity patterns (97,628–97,655 nnz observed)
* **Lead PATH to DIFFERENT local solutions** — some `tc=1 r~5e-8`,
  others `tc=2 r~1.98e-5`

## Why we did NOT lock down the matching deterministically

A natural fix would be to set `PYTHONHASHSEED=0` and sort all set
iterations. This was implemented in
`scripts/gtap_v62/_deterministic_startup.py` and proven to produce
identical Jacobians across runs.

**But:** the deterministic matchings tested (eq-side and var-side
`top_nodes`, both with sorted node iteration) land in the LOTTERY LOSS
basin (`r=1.98e-5`). Enforcing determinism would lock the project into
a known-bad outcome.

Until a SMART matching heuristic is implemented (one that consistently
picks a "good" matching, e.g., by weighting toward fixing QUANTITY vars
over PRICE vars), we leave hash randomization enabled so that runs
have a chance of hitting the good basin.

## What's NOT bake_tolerance-related

These were investigated and ruled out:

* **Dead-row drop threshold** (`1e-6 → 1e-1`): no effect on baseline
  residual. The blocker is not weak Jacobian coupling.
* **Convergence tolerance** (`1e-4 / 1e-6`): only changes whether PATH
  declares `tc=1` vs `tc=2` at the same residual.
* **PATH internal options** (`crash_method`, `major_iteration_limit`):
  no measurable effect on the floor.
* **Both bipartite top-side orientations** (eq-side, var-side): both
  produce systems in the `r=1.98e-5` basin under deterministic startup.

## Recommendation

For `gtap6_15x10` and similar-size datasets:

1. **Use `bake_tolerance=1e-6`** (now the default in
   `validate_v62_parity.py`). Strictly better than `1e-3` under all
   matching outcomes.
2. **Leave Python hash randomization enabled** in production (default
   behavior). Running multiple times until `tc=1` is hit is currently
   the only way to ensure a clean baseline on gtap6_15x10.
3. **Use `_deterministic_startup.py` only for diagnostic work** when
   you need bit-identical reproduction of a specific failure mode.
4. **Accept that gtap6_15x10 baseline is fragile** until a smarter
   matching heuristic is implemented.

## Status — gtap6 PATH baseline scaling

| Dataset    | Vars   | bake_tol | PATH baseline | Status |
|:-----------|-------:|---------:|--------------:|:-------|
| gtap6_3x3  |    663 |    1e-6  | tc=1 r~1e-7   | ✓ reliable |
| gtap6_5x5  |  2,239 |    1e-6  | tc=1 r~1e-7   | ✓ reliable |
| gtap6_10x7 |  8,965 |    1e-6  | tc=1 r~5e-8   | ✓ reliable |
| **gtap6_15x10** | **25,720** | **1e-6** | tc=1 r~5e-8 (33% of runs) / tc=2 r~2e-5 otherwise | **lottery** |
| gtap6_20x41 | ~290K |    1e-6  | not tested    | likely needs MA48 + matching heuristic |

## Open follow-up

Implement a smart bipartite matching that consistently lands in the
"good basin" for gtap6_15x10:

* **Approach A**: Weighted bipartite matching with family-based weights
  — high cost on price vars (`psave`, `pm`, `pwmg`, etc.) discourages
  the matcher from leaving them unmatched.
* **Approach B**: Iterative refinement — after initial matching, swap
  matched/unmatched pairs along augmenting paths to remove
  high-priority unmatched vars.
* **Approach C**: Identify the structural property that distinguishes
  `97,655 nnz` (lottery wins) from `97,628/97,635/97,647 nnz`
  (lottery losses) and engineer for it.

## Files touched

* `scripts/gtap_v62/validate_v62_parity.py` — `bake_tolerance` default
  `1e-3 → 1e-6` (the actual fix).
* `scripts/gtap_v62/_deterministic_startup.py` — new diagnostic helper
  that re-execs with `PYTHONHASHSEED=0`. **Not for production use.**
* `scripts/gtap_v62/_diag_15x10_residual_breakdown.py` — diagnostic
  for analyzing baseline residual by equation family (used during
  this investigation).
* `scripts/gtap_v62/_diag_jac_nnz_stability.py` — diagnostic for
  measuring Jacobian non-determinism across runs.
* `scripts/gtap_v62/_diag_conshr_sum.py` — diagnostic for CDE share
  consistency check across datasets.
* `scripts/gtap_v62/_test_15x10_*.py` — assorted test harnesses.
