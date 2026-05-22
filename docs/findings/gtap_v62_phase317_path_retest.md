# GTAP v6.2 Phase 3.17 — PATH C-API retest on Phase 3.16 calibration

**Date:** 2026-05-22
**Branch:** `gtap/v62-rollback`
**Builds on:** Phase 3.16 (semantically-correct diagonal trade calibration)

## TL;DR

Retested PATH C-API on the Phase 3.16 model where:
- `qim_0` is now consistent with `sum agent imports` (no 78k discrepancy)
- diagonal trade is included throughout calibration and equations
- bipartite matching fixes FEWER variables (82 vs 120 in Phase 3.8)

**PATH still gets stuck at `term_code=2`** with zero movement on
bilateral trade variables. Calibration consistency was the right
fix for semantic correctness but does NOT unblock PATH.

## Numbers

```
Phase 3.16 PATH C-API solve:
  Closure: 620 free vars / 620 active cons (matched)
  fixed_unmatched: pwmg=13, qfe=9, pf=4, pva=3, qo=3, qp=2  -> 34 cells
  total fixed (dangling + unmatched): 82  (was 120 in Phase 3.8)
  
  BASELINE:  term_code=2, residual=9.12e-04, walras=0.00e+00     ✓ trivial
  SHOCKED:   term_code=2, residual=1.42e-01, walras=-1.03e-01    ✗ stuck
  
  Movement on bilateral trade (pms, qxs, qim, pim) after shock: ZERO
```

Same diagnostic as Phase 3.14: PATH stops at a stationary point of
its merit function near the baseline. The shock perturbs
`eq_pms[food,USA,EU]` by +0.153 but PATH cannot find a Newton step
that decreases the merit function — every candidate step fails the
line-search test.

## Why Phase 3.16 didn't help

The cleaner calibration means:
- 38 NEW free variables for diagonal trade (qxs[r,r], pms[r,r], etc.)
- 38 NEW active constraints (eq_pe[r,r], eq_pms[r,r], etc.)
- The bipartite matching now has more elements but the same
  structural issue: 34 unmatched + 48 dangling = 82 fixed at SAM.

The trade-relevant variables (pms, qxs, qim, pim, pe, pmcif) are
still FREE in the matching. So calibration consistency isn't the
proximate cause of PATH stalling.

The fundamental issue (Phase 3.14): PATH's merit-function gradient
is locally zero at the prebalance-baked baseline. This is a
property of how the constraint system + prebalance combine, not a
calibration artifact.

## What this means for PATH on v6.2

After Phases 3.7, 3.8, 3.12, 3.13, 3.14, 3.16, 3.17 the situation is
unambiguous:

**PATH cannot solve the v6.2 BOOK3X3 shocked equilibrium under the
current model + prebalance design.** No combination of:
- Diagonal scaling (Phase 3.12)
- Homotopy substepping (Phase 3.14)
- Initial-point perturbation (Phase 3.14)
- IPOPT-warm-start polish (Phase 3.13)
- Cleaner calibration with consistent diagonal handling (3.16)

unblocks PATH for the shocked solve.

What WOULD unblock PATH (each is a substantial refactor):

1. **Audit and fill the 27 missing equation cells** for the
   unmatched-variable families (pwmg, qfe, pf, pva, qo, qp).
   Once the system has a genuinely square structural form without
   the bipartite-fix workaround, the merit function should be
   well-behaved.

2. **Switch to GTAPinGAMS-style auxiliary-price MCP** with explicit
   complementarity pairs. The auxiliary formulation naturally
   produces a non-singular Jacobian.

3. **Use the Pyomo MPEC/PATH extension** with built-in complementarity
   support, instead of treating the system as pure equalities passed
   to `solve_nonlinear_mcp`.

## Current best v6.2 parity (unchanged)

```
IPOPT in Mode B (~20-40% of runs):
  VIWS USA→EU  Python +50.43%  vs Gragg-multi +53.52%  ->  -3.1pp gap
```

This is the verifiable parity floor for v6.2 BOOK3X3 with the
current model architecture. Tighter parity requires structural
refactor work, not solver tuning.

## Reproduce

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PATH_LICENSE_STRING = "<your-license>"
$env:PATH_CAPI_LIBPATH = "C:/GAMS/53/path52.dll"
$env:PATH_CAPI_LIBLUSOL = "C:/GAMS/53/lusol.dll"

python scripts/gtap_v62/validate_v62_parity.py shock `
    --experiment Exp1a --solver path-capi `
    --workdir runs/gtap_v62_parity/BOOK3X3_Exp1a
```

Expected output: term_code=2 with zero movement on pms/qxs/qim/pim.

## Final recommendation for this branch

The 16 phases of investigation on `gtap/v62-rollback` have produced:

1. ✓ A working v6.2 Python model with SAM-consistent calibration.
2. ✓ PATH C-API integration (works for baseline, stuck on shocked).
3. ✓ IPOPT shocked solve achieving ~3pp parity vs GEMPACK Gragg-multi.
4. ✓ Complete documentation of WHERE the model differs from GEMPACK.
5. ✗ Production-grade solver stability (Mode A/B bimodality + PATH stall).

For a production v6.2 implementation, three paths remain:

A. Continue with model refactor (audit the 27 missing equations).
   Estimated effort: 1-2 weeks of careful TAB-equation reading +
   Pyomo coding + iterative testing.

B. Abandon v6.2 levels-MCP, use GTAPinGAMS conventions instead.
   Estimated effort: ~1 month rewrite, but solver stability is the
   GTAPinGAMS strong suit.

C. Accept ~3pp parity in best-case Mode B as v6.2's research-grade
   result, document the limitations, and merge the branch as a
   reference implementation. Production users can fall back to
   GEMPACK's gtap.exe for shocks.
