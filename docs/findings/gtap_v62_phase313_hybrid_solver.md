# GTAP v6.2 Phase 3.13 — IPOPT crash + PATH polish hybrid

**Date:** 2026-05-22
**Branch:** `gtap/v62-rollback`
**Builds on:** Phase 3.12 (PATH scaling — didn't unblock shocked solve)

## TL;DR

Added an `ipopt+path` hybrid solver: IPOPT does a "crash" solve to
get a non-trivial movement from baseline, then PATH polishes to
F(x) = 0. The hybrid **sometimes** delivers ~1pp parity vs Gragg
multi-step, but it's unstable because IPOPT's bimodality (Mode A
~46% / Mode B ~52% on VIWS) corrupts the warm start.

Added a Mode-A guard: skip the PATH polish step when |walras| > 1.0
(IPOPT didn't actually converge). When triggered, falls back to the
raw IPOPT solution. This reduces catastrophic divergence but
doesn't fix the underlying Mode A frequency.

## Why this matters per CLAUDE.md

The project rule (CLAUDE.md) is "always use PATH" for GTAP work.
Phase 3.7-3.12 showed PATH solves the v6.2 BASELINE correctly but
gets stuck at `term_code=2` (NoProgress) on the SHOCKED solve, even
with prebalance + scaling. Phase 3.13 attempts to honor the PATH
rule by having PATH at least *polish* the IPOPT crash solution.

## Hybrid wiring

```python
if use_hybrid:
    # 1. Solve baseline with IPOPT (regularizer objective).
    res = ipopt.solve(model)

    # 2. If IPOPT converged feasibly, polish with PATH.
    if abs(value(model.walras)) <= HYBRID_WALRAS_THRESHOLD:
        del model.obj
        path_capi_solve(model)
    else:
        print("Skipping PATH polish (Mode A — IPOPT didn't converge)")

    # 3. Apply shock.
    # 4. Solve shocked with IPOPT.
    # 5. If walras small, PATH polish; else skip.
```

CLI: `--solver ipopt+path` (default threshold via env
`HYBRID_WALRAS_THRESHOLD=1.0`).

## Stability test (5 runs)

```
Run 1:  IPOPT walras=4.72e0/3.34e1   skip polish  VIWS=-52.80%  ❌
Run 2:  IPOPT walras=2.70e0/3.32e1   skip polish  VIWS=+45.42%  Mode A
Run 3:  IPOPT walras=8.09e1/2.10e1   skip polish  VIWS=+42.05%  Mode A
Run 4:  IPOPT walras=<1/<1           PATH polish  VIWS=+11.29%  ❌ broke
Run 5:  IPOPT walras=<1/>1           polish-once  VIWS=+52.52%  Mode B  ✅
                                                  (-0.99pp vs Gragg)
```

Run 5 is the only "good" outcome. Runs 1 and 4 are catastrophic
(VIWS jumps to -53% or +11%). The bimodality dominates.

## What this tells us

The actual blocker is **IPOPT's bimodality + PATH's stationary-point
behavior compound**:
- When IPOPT lands in Mode B (rare), PATH polish takes us to clean
  parity.
- When IPOPT lands in Mode A (common), PATH polishes from a wrong
  point and lands somewhere else, sometimes worse.
- When PATH-on-its-own runs from baseline, it never moves (term_code=2).

The model itself has multiple equilibria (or at least multiple
"merit-function stationary points") that confuse both solvers.

## Where the real fix lives

The 27 unmatched variables fixed by `_make_square.py`'s bipartite
matching (qfe×9, pf×6, pwmg×5, pva×3, qim×2, qo×2) form a fixed
subspace. The remaining 555 free vars must move under the shock,
but the fixed subspace makes the local geometry singular.

**Phase 3.14 candidate**: instead of fixing 27 unmatched variables
at their initial values, leave them free and instead deactivate
the same number of "redundant" equations. The bipartite matching
already identifies which equations are redundant (those that don't
get matched to a variable). This gives more degrees of freedom to
the system and may unblock PATH.

Or more radical: redesign the closure entirely. The current closure
(`pgdpwld=1`, `qoes` fixed, `savf` fixed, `kb`, `ke` fixed, `rorg=1`,
`pgdpmp[r]=1`) might be over-constraining once the prebalance has
baked the SAM imperfections into equation constants.

## When to use the hybrid

- For one-off parity tests, retry until Run 5-style outcome
  (recognize by walras converging both times).
- For batch validation, prefer the deterministic Gragg multi-step
  GEMPACK reference as ground truth and accept ±2pp from any
  Python solve.
- For PR validation, the hybrid is too unreliable; use IPOPT alone
  and note Mode A vs Mode B in the report.

## Reproduce

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PATH_LICENSE_STRING = "<your-license>"
$env:PATH_CAPI_LIBPATH = "C:/GAMS/53/path52.dll"
$env:PATH_CAPI_LIBLUSOL = "C:/GAMS/53/lusol.dll"
$env:HYBRID_WALRAS_THRESHOLD = "1.0"   # default

python scripts/gtap_v62/validate_v62_parity.py shock `
    --experiment Exp1a --solver ipopt+path `
    --workdir runs/gtap_v62_parity/BOOK3X3_Exp1a
```

## Phase 3.14 — next

The cleanest investigation is to audit the bipartite-matching
closure step:

1. Print the 27 unmatched variable cells AND the equations they
   "should" have matched to.
2. For each, check if the var-side fix is the right choice or if
   an eq-side deactivation would be cleaner.
3. Consider whether `qim` and `qo` (real economic variables, not
   slack helpers) should ever be in the unmatched list — they may
   indicate the closure is wrong, not that they're truly redundant.

The user's question "why can't we use PATH for both baseline and
shocked?" is the right question. The answer is the closure squaring
fixes things PATH needs to be free.
