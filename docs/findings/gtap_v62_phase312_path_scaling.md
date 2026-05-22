# GTAP v6.2 Phase 3.12 — PATH variable+equation scaling (insufficient)

**Date:** 2026-05-22
**Branch:** `gtap/v62-rollback`
**Builds on:** Phase 3.11 (Hypothesis B rejected, IPOPT bimodality discovered)

## TL;DR

Implemented diagonal pre-scaling for variables and equations in
`_path_capi_solver.py` to address PATH's `term_code=2 (NoProgress)`
on the shocked solve. The scaling is structurally correct
(variables span ~12 orders of magnitude on v6.2 BOOK3X3) but **does
not unblock PATH** — the shocked solve still terminates at
`term_code=2` with residual ~1.4e-01 and zero observable movement
on the bilateral US→EU prices/quantities.

The PATH stuck behavior is more fundamental than naive variable
scaling: even with all variables and equation residuals normalized
to O(1), PATH's merit function lands at a stationary point near
the baseline and can't find a productive Newton direction. This is
likely related to the structure of the bipartite-matched constraint
system after `_make_square.py` (27 unmatched variables fixed at
their initial values may create a Jacobian rank-deficient region).

For now, IPOPT in Mode B (Phase 3.11) remains the only solver that
produces a meaningful shocked solution, and parity is ~1pp vs
GEMPACK Gragg multi-step when IPOPT actually converges (which
happens in ~60% of runs).

## What was added

`scripts/gtap_v62/_path_capi_solver.py` now wraps the PATH callbacks
with diagonal scaling:

```python
y = x / var_scale[i]       where var_scale[i] = max(|x_0[i]|, 1.0)
F_scaled = F(x) / eq_scale[i] where eq_scale[i] = max(|F_i(x_0)|, 1.0)
J_scaled[i,j] = J[i,j] * var_scale[j] / eq_scale[i]
```

The wrapper also un-scales the solution before writing back into
the Pyomo Var objects. Toggled by `--variable-scaling` /
`--equation-scaling` env vars (`PATH_VAR_SCALE`, `PATH_EQ_SCALE`),
default ON.

## Indexing bug found and fixed

While implementing the scaling, found that `JacobianStructure` in
`path_capi_python.mcp` uses **1-based** indices for `col_starts`
and `row_indices` (PATH MCP convention) but the `values` list in
the callback is 0-indexed Python. The first version of the wrapper
used 1-based indices to index a 0-based list, causing access
violations. Fixed in `jac_scaled` by converting `kk_1based - 1`
before indexing `jvals` and `out`.

## Why scaling alone doesn't unstick PATH

Even after scaling:
- Baseline: term_code=2, residual=9.12e-04, walras=-6.66e-02
- Shocked: term_code=2, residual=1.42e-01, walras=-6.66e-02
- Major iterations: 14 (shocked), with 682 minor iterations
- Variable movement: **zero** on pms/qxs/qim — the variables the
  shock should directly affect.

The shock perturbs `eq_pms[food,USA,EU]` body by +0.153. PATH
should respond by changing `pms[food,USA,EU]` (which is free).
But it doesn't move. Likely cause: the bipartite matching that
fixes 27 unmatched variables (`qfe`, `pf`, `pwmg`, `pva`, `qim`,
`qo`) at baseline values creates a system where the Newton step
in the (pms, pmcif, pe, qxs) subspace conflicts with the fixed
variables' implicit constraints.

Three more diagnostic angles for Phase 3.13:
1. **Warm-start PATH from an IPOPT Mode B solution.** If PATH
   can polish a near-equilibrium point, it confirms the issue is
   landing in the right basin, not the local geometry.
2. **Apply the shock in substeps.** Instead of `tms: 0.369 →
   0.232` in one go, do 10 substeps and re-solve PATH at each.
   This mimics GEMPACK Gragg's multi-step approach.
3. **Audit the 27 bipartite-fixed variables.** Some of them
   (`qim`, `qo`) are typically endogenous and being fixed may be
   creating the locked subspace. Reconsider which variables to
   fix.

## Current best parity (IPOPT Mode B, Phase 3.8 calibration)

When IPOPT lands in Mode B (~60% of runs), the parity is:

```
Cell             Gragg-multi    Python (best)    Gap
VIMS US→EU       +38.165%       +38.98%          +0.82pp
VIWS US→EU       +53.517%       +54.42%          +0.91pp
VXMD US→EU       +53.548%       +54.43%          +0.88pp
VDPM food EU     -0.335%        -0.19%           +0.14pp
VIPM food EU     +2.278%        +2.43%           +0.15pp
```

This is the *real* parity status — the 9pp / 2.5pp gaps from
earlier phases were a mix of GEMPACK Johansen-1 artifact (Phase
3.9) and IPOPT solver noise (Phase 3.11).

## Reproduce PATH scaling test

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PATH_LICENSE_STRING = "<your-license>"
$env:PATH_CAPI_LIBPATH = "C:/GAMS/53/path52.dll"
$env:PATH_CAPI_LIBLUSOL = "C:/GAMS/53/lusol.dll"
$env:PATH_VAR_SCALE = "1"    # default ON
$env:PATH_EQ_SCALE = "1"     # default ON

python scripts/gtap_v62/validate_v62_parity.py shock `
    --experiment Exp1a --solver path-capi `
    --workdir runs/gtap_v62_parity/BOOK3X3_Exp1a
```

Toggle scaling off by setting `PATH_VAR_SCALE=0` or `PATH_EQ_SCALE=0`.

## Phase 3.13 — next

The most likely fix at this point is IPOPT-warm-start-PATH:

1. Solve baseline with IPOPT to ~1e-6 tolerance.
2. Apply shock.
3. Solve shock with IPOPT to ~1e-6 tolerance — lands in Mode B
   ~60% of the time; retry until Mode B.
4. Use the Mode B IPOPT solution as warm start for PATH.
5. PATH polishes the last few digits.

This effectively makes IPOPT a "crash" step and PATH a "polish"
step. It's what GEMPACK GUI does internally (the .CMF file's
`subintervals` option is essentially this).
