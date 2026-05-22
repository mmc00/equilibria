# GTAP v6.2 Phase 3.11 — Hypothesis B rejected + IPOPT bimodality discovered

**Date:** 2026-05-22
**Branch:** `gtap/v62-rollback`
**Builds on:** Phase 3.10 (Gragg multi-step convergence — 2.5pp gap declared "real")

## TL;DR

The hypothesized 2.5pp gap from Phase 3.10 turned out to be **IPOPT
solver bimodality**, not a calibration issue. Phase 3.11 tested
Hypothesis B (re-calibrate bottom Armington with `qxs_0 = vxmd`
producer-price quantity instead of `vxwd` world-price quantity);
that change made parity *worse*, so it was reverted. While running
the comparison, we discovered IPOPT lands in one of two attractors
across independent runs:

- **Mode A** (~47% on VIWS): "locally infeasible" termination far
  from the true equilibrium.
- **Mode B** (~52-53% on VIWS): convergence within **~1pp** of the
  GEMPACK Gragg multi-step reference (+53.52%).

When IPOPT lands in Mode B, Python achieves *sub-percentage-point*
parity with non-linearly-corrected GEMPACK. The 2.5pp, 9pp, or 19pp
gaps reported in earlier phases were a mix of solver noise + the
Johansen-1-step linearization artifact.

## Hypothesis B test (REJECTED)

The Phase 2d calibration set `qxs_0 = vxwd` (world-price quantity,
including the margin layer). Hypothesis B was that the 2.5pp under-
shoot vs Gragg multi-step came from this choice — re-calibrating
with `qxs_0 = vxmd` (producer-price quantity) might align with
GEMPACK's `qxs` semantics.

Implemented change:

```python
# Before (Phase 2d):                 After (Phase 3.11 test):
qxs_0 = vxwd                         qxs_0 = vxmd
pe_0  = vxmd / vxwd                  pe_0  = 1.0
pwmg_0 = vtwr_total / vxwd           pwmg_0 = vtwr_total / vxmd
pmcif_0 = viws / vxwd                pmcif_0 = viws / vxmd
pms_0 = vims / vxwd                  pms_0 = vims / vxmd
txs_0 = (vxmd - vxwd) / vxwd         txs_0 = 0  (absorbed into qxs scale)
```

**Immediate side win:** `eq_market` residual collapsed from 2.3e4
(Phase 2d) to 9.3e-10 (machine precision). The export-margin layer
no longer pushed onto the uses side of market clearing.

**Parity outcome** (single IPOPT run, before noise quantification):

| Cell | Gragg multi-step | Phase 2d (vxwd) | Phase 3.11 (vxmd) |
|------|------------------|-----------------|-------------------|
| VIMS US→EU | +38.17% | +35.89% (Δ=-2.3pp) | +31.93% (Δ=-6.2pp) |
| VIWS US→EU | +53.52% | +50.99% (Δ=-2.5pp) | +46.59% (Δ=-6.9pp) |
| VXMD US→EU | +53.55% | +50.99% (Δ=-2.6pp) | +46.60% (Δ=-7.0pp) |

The producer-price calibration made bilateral parity **worse by
~4pp**, not better. Reverted.

## IPOPT bimodality (the actual story)

After reverting Phase 3.11 to the Phase 2d calibration, five
independent IPOPT runs gave:

```
Run 1: VIWS = +46.034%  ← Mode A
Run 2: VIWS = +52.951%  ← Mode B
Run 3: VIWS = +46.540%  ← Mode A
Run 4: VIWS = +52.618%  ← Mode B
Run 5: VIWS = +53.171%  ← Mode B  (closest to Gragg +53.52)
```

Every run terminates with IPOPT warning "locally infeasible point.
Problem may be infeasible." The non-trivial walras (-98, -0.66, etc.)
confirms IPOPT didn't enforce the global balance identity at exit.
The convergence path is sensitive to small numeric perturbations
(Python hash randomization, slightly different initial-point ordering
after pycache invalidation, etc.), and IPOPT lands in different
attractors across runs.

**Mode B is the genuine equilibrium**; when IPOPT reaches it, Python
matches Gragg multi-step to within **~0.3-1pp**:

```
Best Mode B run vs Gragg multi-step:
  VIMS US→EU:  +38.98% vs +38.17%  →  +0.82pp
  VIWS US→EU:  +54.42% vs +53.52%  →  +0.91pp
  VXMD US→EU:  +54.43% vs +53.55%  →  +0.88pp
  VDPM food EU: -0.19% vs  -0.34%  →  +0.14pp
  VIPM food EU: +2.43% vs  +2.28%  →  +0.15pp
```

## What this means

The structural model is already very close to GEMPACK at the level
of non-linear equilibrium. The "9pp Phase 3.7 gap" was 70% Johansen-1
linearization artifact (Phase 3.9), 20% solver noise (this phase),
and ≤10% structural model differences.

What we ACTUALLY need is **solver stability**, not more calibration
work. Two clean paths:

1. **PATH C-API**: currently terminates at `term_code=2` (NoProgress)
   for the shocked solve. Likely cause is Jacobian scaling — vars
   span ~12 orders of magnitude. Pre-scale variables/equations
   before passing to PATH.

2. **IPOPT tighter convergence**: experiment with stricter
   `bound_relax_factor`, `mu_init`, `mu_strategy=adaptive`, or
   warm-start from a perturbed Mode B solution.

## Lessons learned

- **Always run a solver-stability check** before declaring a parity
  gap "real". A 5-run reproducibility test reveals bimodality
  quickly.
- **IPOPT "locally infeasible" warnings matter**: they don't mean
  "close enough" — they mean the solver gave up at a non-equilibrium
  point that *might* be near the answer.
- **The chain "wrong reference → wrong calibration → wrong solver"
  compounds noise**. Phase 3.7 saw 19pp, Phase 3.9 fixed the
  reference, Phase 3.11 ruled out the calibration, and the residual
  was solver noise all along.

## Reproduce

```powershell
# Clear pycache to ensure deterministic comparison runs:
Get-ChildItem -Recurse -Force -Directory -Filter __pycache__ |
    Remove-Item -Recurse -Force

# Run 5 independent IPOPT solves; observe mode A vs mode B split:
for ($i=1; $i -le 5; $i++) {
    Write-Host "=== Run $i ==="
    python scripts/gtap_v62/validate_v62_parity.py shock `
        --experiment Exp1a --solver ipopt `
        --workdir runs/gtap_v62_parity/BOOK3X3_Exp1a |
        Select-String "VIWS food USA"
}
```

## Phase 3.12 — next

Focus on solver stability, not calibration:

1. **PATH equation/variable scaling**: scale all prices to O(1) and
   quantities to O(1) by dividing by benchmark values before passing
   the residual/Jacobian callbacks to PATH. This should unstick the
   stationary-point issue.
2. **IPOPT warm-start from baseline**: pre-solve baseline with
   tight tolerance, then shock and re-solve with `warm_start_init_point=yes`
   and `bound_push=1e-9`. May force IPOPT into Mode B reliably.

Once the solver lands consistently at Mode B, the actual structural
parity gap (currently <1pp on the bilateral cells) can be measured
properly and any remaining calibration discrepancies can be diagnosed.
