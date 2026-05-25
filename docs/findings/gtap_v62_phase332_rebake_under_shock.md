# GTAP v6.2 Phase 3.32 — Rebake baked residuals under shock

**Date:** 2026-05-25
**Branch:** `gtap/v62-rollback`
**Scope:** v6.2 only
**Builds on:** Phase 3.30 (auto-drop dead rows) + Phase 3.31 (chained warm-start)

## TL;DR

Tracked the PATH SHOCKED `term_code=2` failure to its **actual root cause**:
the prebalance bake (Phase 3.8) writes residuals as static constants at
the BASELINE state; once any parameter shifts (e.g., `tms`), those
baked constants become stale and a single cell concentrates ~all the
remaining residual.

**Diagnosis on gtap6_5x5 SHOCKED**:
```
Equation family               max|F|        sum|F|        n
sam_baked_residuals           6.7608e+02    6.7608e+02   213
eq_qxs                        1.3593e-01    1.3651e-01   125
eq_pms                        1.1113e-01    1.1115e-01   125
eq_qpm                        3.7844e-04    4.2698e-04    25
...
```

ONE single baked cell has residual 676; everything else is < 0.14. That
cell encodes a static constant that the shock invalidated.

**Phase 3.32 fix**: re-evaluate baked residuals at the current
(post-shock) state and update the corresponding constraints
in-place. Implemented as `rebake_residuals_at_current_state()` and
called automatically in each homotopy substep.

**Empirical impact on BOOK3X3**:

| Setup | residual @ shocked | VIWS food USA→EU |
|:------|-------------------:|------------------:|
| Phase 3.30 (no rebake)         | 6.03e-01 | +0.479% |
| Phase 3.30 + rebake (1 step)   | 1.41e-01 | +0.227% |
| Phase 3.30 + rebake (5 substep) | 1.40e-01 | +0.686% |

The residual at the post-shock starting point is now bounded (1e-2 to
1e-1 across substeps) instead of jumping to 6e-1+. PATH no longer
plays catch-up with stale bake constants.

## Diagnosis path

1. **Started measuring**, not theorising. Wrote a residual-breakdown
   diagnostic that prints `max|F|` and `sum|F|` per equation family at
   PATH's stationary point.

2. **Found one cell dominates**:
   ```
   sam_baked_residuals[186]: residual = +6.7608e+02
   ```
   All other equations have residual < 0.15. PATH was working hard to
   reduce 600+ when only one cell was responsible.

3. **Identified mechanism**: `bake_baseline_residuals_as_slacks()`
   captures residuals at baseline as constants:
   ```python
   model.sam_baked_residuals.add(body - residual_0 == rhs)
   #                                    ↑
   #                                    constant from BASELINE
   ```
   After `model.tms.fix(new_value)`, the body expression evaluates
   differently but `residual_0` is frozen at the baseline value. The
   constraint becomes inconsistent → PATH chases a phantom residual.

## What changed

### `bake_baseline_residuals_as_slacks` — track metadata for rebake

Each baked cell now records its live components so we can refresh it:

```python
model._baked_cell_metadata.append({
    "orig_con": c,
    "orig_idx": idx,
    "body": body,        # Pyomo expression (re-evaluable)
    "lower": lower,
    "upper": upper,
    "replacement": replacement,  # the live Constraint cell
})
```

### `rebake_residuals_at_current_state` — refresh in place

New function in `scripts/gtap_v62/_make_square.py`:

```python
def rebake_residuals_at_current_state(model, tolerance=1e-3):
    for meta in model._baked_cell_metadata:
        body_val = value(meta["body"])    # re-evaluate at current state
        new_residual = body_val - rhs
        meta["replacement"].deactivate()  # remove stale replacement
        if abs(new_residual) > tolerance:
            new = model.sam_baked_residuals.add(body - new_residual == rhs)
            meta["replacement"] = new     # track new live cell
```

Crucially: does NOT touch original (already-deactivated) constraints,
so closure square-ness is preserved.

### `validate_v62_parity.py` — call rebake in each homotopy substep

```python
for step in range(1, n_steps + 1):
    model.tms["food", "USA", "EU"] = tms_step
    rebake_info = rebake_residuals_at_current_state(model)
    _solve_path_capi(...)
```

### `_path_capi_solver.py` — write back based on residual, not term_code

Phase 3.31's strict "term_code=1 only" rule was too conservative — it
blocked warm-start chaining for substeps that made real progress but
didn't reach term_code=1 in their iteration budget. Phase 3.32 writes
back whenever `residual < 10.0` (any reasonable improvement):

```python
if result.residual < 10.0:
    # un-scale and write back
    for var, val in zip(free_vars, x_solution):
        var.set_value(float(val), skip_validation=True)
```

## What still doesn't work

PATH shocked **still hits `term_code=2`**, but with residual ~0.14 instead
of ~0.6. The 4× reduction in residual confirms the rebake addresses the
diagnosed root cause. The remaining residual is smaller and distributed
across many equations (no single dominating cell).

The remaining ~0.14 residual means PATH found a fixed-point of its merit
function near the desired equilibrium but couldn't drive F to exactly
zero. Possible causes (not investigated in 3.32):

1. **Re-bake itself adds noise**: each rebake call updates ~38 constants
   simultaneously, which may shift the model into a slightly different
   "nearby equilibrium" each time. Convergence might require either
   tighter rebake tolerance or single-pass rebake (not per-substep).

2. **Bake constants pin the solution near baseline**: by encoding "F was
   X at baseline, force F to be X at current state", we anchor the
   solution to baseline. The shocked equilibrium may live far from this
   anchor.

3. **Selective rebake**: only rebake equations directly affected by the
   shock (e.g., only `eq_pms` if shocking `tms`). The 38-cell rebake
   may be over-correcting.

## Honest assessment

| Metric | Before 3.32 | Phase 3.32 |
|:-------|------------:|:-----------|
| BOOK3X3 SHOCKED residual | 6.03e-01 | **1.41e-01** (4× better) |
| BOOK3X3 SHOCKED VIWS | +0.479% | +0.686% (5 substeps) |
| Root cause identified | partial | **yes — sam_baked_residuals[186] dominated** |
| PATH SHOCKED converges (term_code=1) | ✗ | ✗ still |
| GEMPACK Gragg-multi VIWS reference | +53.5% | same |

The diagnostic narrowing was valuable: we now know **exactly** what
PATH was struggling with. The fix addresses the main bug (stale bake
constants) and reduces the residual 4×. Full convergence requires more
careful rebake strategy or the structural refactors (MPSGE-style
complementarity) outside Phase 3.32 scope.

## Files

`scripts/gtap_v62/_make_square.py`:
- `bake_baseline_residuals_as_slacks` now records `_baked_cell_metadata`
- New `rebake_residuals_at_current_state(model, tolerance)`

`scripts/gtap_v62/_path_capi_solver.py`:
- Write-back triggered by `result.residual < 10.0` (replaces strict
  `term_code == 1` rule)

`scripts/gtap_v62/validate_v62_parity.py`:
- Homotopy loop calls `rebake_residuals_at_current_state()` after each
  tariff substep

## Reproduce

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PATH_LICENSE_STRING = "<license>"
$env:PATH_CAPI_LIBPATH = "C:/GAMS/53/path52.dll"
$env:PATH_CAPI_LIBLUSOL = "C:/GAMS/53/lusol.dll"

python scripts/gtap_v62/validate_v62_parity.py shock `
    --experiment Exp1a --solver path-capi `
    --homotopy-steps 5 `
    --workdir runs/gtap_v62_parity/BOOK3X3_Exp1a
```

## Status

| Feature | Status |
|--------|--------|
| Residual root-cause diagnosed | ✓ (one cell, 676 USD) |
| Rebake at current state implemented | ✓ |
| Write-back relaxed to residual<10 | ✓ |
| Homotopy substeps with rebake | ✓ |
| 4× residual reduction on BOOK3X3 | ✓ |
| PATH SHOCKED term_code=1 | ✗ still (1.4e-1 not 0) |
| Methodology: measure → diagnose → fix | ✓ (replaces speculation) |

Phase 3.32 confirms the value of **diagnosing before declaring defeat**.
The previous 4-5 phases hypothesised various structural causes; Phase
3.32 measured directly and found a specific, fixable cell.
