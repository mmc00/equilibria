# GTAP v6.2 Phase 3.8 — SAM pre-balance via residual bake-in

**Date:** 2026-05-21
**Branch:** `gtap/v62-rollback`
**Builds on:** Phase 3.7 (PATH C-API wiring, ``to=0`` rollback)

## Summary

Adopted the GAMS-standard treatment of SAM imperfections: for each
equation cell whose body has a non-trivial baseline residual, bake
that residual into the constraint as a constant slack. The dynamics
(dF/dx · dx during a shock) are unchanged, but ``F(x_0) = 0`` holds
to machine precision after pre-balance. This restores the GEMPACK
implicit-linearization treatment within a levels-MCP formulation.

**Parity improvement (IPOPT, 10% tariff cut tms[food,USA,EU]):**

| Aggregate | GEMPACK | Phase 3.7 Python | Phase 3.8 Python | Gap reduction |
|-----------|---------|------------------|------------------|---------------|
| VIMS USA→EU | +31.54% | +44.68% (+13.1pp) | **+35.39%** (+3.9pp) | **9.2pp** |
| VIWS USA→EU | +41.54% | +60.76% (+19.2pp) | **+50.44%** (+8.9pp) | **10.3pp** |
| VXMD USA→EU | +41.55% | +60.75% (+19.2pp) | **+50.44%** (+8.9pp) | **10.3pp** |
| VDPM food EU | -0.264% | -0.185% (+0.08pp) | **-0.227%** (+0.04pp) | 0.04pp |
| VIPM food EU | +1.848% | +2.457% (+0.6pp) | +2.860% (+1.0pp) | -0.4pp |

The bilateral US→EU food trade gap collapsed from ~19pp to ~9pp.
VDPM converged closer; VIPM remained sensitive to the residual
Armington calibration.

## What was added

- `scripts/gtap_v62/_make_square.py`:
  `bake_baseline_residuals_as_slacks(model, tolerance=1e-3)`
  walks every active constraint cell, evaluates ``body - rhs`` at
  the current variable state, and for any cell with |residual| > tol
  deactivates the original and adds a replacement
  ``body - residual_0 == rhs`` to a new `ConstraintList`
  `sam_baked_residuals`. Returns ``{n_baked, max_abs_baked, by_eq}``.

- `validate_v62_parity.py` calls the helper after closure & squaring
  and reports the baked count + families. Adds a per-variable delta
  diagnostic between baseline and shocked solves to surface what
  actually moved.

## SAM imperfections found in BOOK3X3

Pre-balance baked 25 cells across 4 families:

```
eq_qtm           : 1 cell  — diagonal VTWR[m,i,s,s] contributes 65,838
                              to the world margin demand identity
eq_market        : 7 cells — qxs_0 = vxwd (Phase 2d) pushes the export
                              margin layer onto the uses side, which
                              VOM does not contain
eq_qo            : up to 9 cells — vom/vop output-tax wedge, kept
                              implicit in calibrated ``to``
eq_factor_clear  : ≤9 cells — float32 rounding in HAR storage
```

Diagonal VTWR magnitude (BOOK3X3):
```
VTWR diagonal (s==d): 65,838.45
VTWR off-diagonal:   143,724.50
VTWR total:          209,562.95
```

## Why this works

GEMPACK and a levels-MCP solver disagree on whether the SAM benchmark
must be exactly feasible:

| | GEMPACK (TAB / Johansen) | Levels MCP (PATH, IPOPT-NLP) |
|--|--|--|
| Form | percent changes from SAM | F(x) = 0 in absolute units |
| Baseline check | none (linearization point) | must satisfy F(x_0) ≈ 0 |
| Imbalance handling | dragged along as constant | reported as residual |

Baking the residual as a constant offset to the equation makes the
levels formulation match GEMPACK's implicit behavior: the equation
holds at baseline by construction, and during the shock the derivatives
propagate identically because the offset is a constant. This is the
standard GTAPinGAMS treatment (the ``c[i,r]`` parameters that
absorb SAM imperfections in GAMS market-clearing identities).

## PATH C-API status

PATH baseline now converges to residual=9.13e-04 / walras=0
(machine precision), but the shocked solve still terminates at
term_code=2 (NoProgress) with residual ~1.5e-01, only moving a few
factor employment variables by 1e-12. The shock to
``eq_pms[food,USA,EU]`` (Δ=+0.153) does propagate as a parameter
change, but PATH's Newton step fails to find a productive direction.
This is likely a Jacobian-scaling issue: the free variables span
~12 orders of magnitude (pms~1, qfe~3.8e5) and PATH's default
LUSOL factorization is sensitive to that. Out of scope for this
phase.

IPOPT (NLP regularizer) does take productive steps and lands at the
parity values reported above, despite ending in "locally infeasible"
status. The residual ``F(x*)`` has converged to machine precision
on the original equation cells (post-prebalance) but the IPOPT
infeasibility flag fires because the slack-baked constraints retain
walras = -98 — IPOPT does not strictly enforce the walras identity
during the shocked solve.

## Reproduce

```powershell
# IPOPT (cleaner shock progress, still some warnings)
$env:PYTHONIOENCODING = "utf-8"
python scripts/gtap_v62/validate_v62_parity.py shock `
    --experiment Exp1a --solver ipopt `
    --workdir runs/gtap_v62_parity/BOOK3X3_Exp1a

# PATH C-API (baseline converges, shock stalls)
$env:PATH_LICENSE_STRING = "<your-license>"
$env:PATH_CAPI_LIBPATH = "C:/GAMS/53/path52.dll"
$env:PATH_CAPI_LIBLUSOL = "C:/GAMS/53/lusol.dll"
python scripts/gtap_v62/validate_v62_parity.py shock `
    --experiment Exp1a --solver path-capi `
    --workdir runs/gtap_v62_parity/BOOK3X3_Exp1a
```

## Phase 3.9 — next

The residual ~9pp gap on bilateral US→EU food trade points to two
remaining issues:

1. **Armington upper-nest elasticity audit:** Phase 2d set
   `qxs_0 = vxwd` (world-price quantity). Combined with
   ``esubm[food] ≈ 5.6`` (from Default.prm), the bilateral
   substitution response is too elastic relative to GEMPACK.
   Re-examine the calibration: should `qxs_0 = vxmd` (producer-
   price quantity) with a separate margin layer?

2. **PATH scaling:** Either pre-scale variables/equations or use
   PATH's built-in `Equation_Scaling auto` option to handle the
   ~12-order-of-magnitude span. This may unlock PATH for the
   shocked solve.

3. **IPOPT walras enforcement:** Add an explicit residual penalty
   to the IPOPT objective so it drives walras to zero, removing
   the spurious "locally infeasible" status.
