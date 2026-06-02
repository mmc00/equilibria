# GTAP lagged-state variable map (Phase C.1)

Source: `src/equilibria/templates/reference/gtap/scripts/iterloop.gms` lines 151-182.

This is the canonical list of variables that NEOS fixes from period `tsim-1` before
solving period `tsim`. Phase C.2 (`gtap_iterloop.py`) implements `_fix_lagged_state(model, prev_model)`
which mirrors this block by copying `.value` from the previously-solved model into the
new model and pinning them with `.fix()`.

## Map

| GAMS name  | GAMS dims     | Python Var name | Python dims     | Status          | Notes |
|------------|---------------|-----------------|-----------------|-----------------|-------|
| `axp`      | `(r,a)`       | `axp`           | `(r,a)`         | ✓ direct        |       |
| `lambdand` | `(r,a)`       | `lambdand`      | `(r,a)`         | ✓ direct        |       |
| `lambdava` | `(r,a)`       | `lambdava`      | `(r,a)`         | ✓ direct        |       |
| `lambdaio` | `(r,i,a)`     | `aioall`        | `(r,i,a)`       | ✓ renamed       | Same economic role: intermediate-input efficiency |
| `lambdaf`  | `(r,fp,a)`    | `lambdaf`       | `(r,f,a)`       | ✓ direct        | `fp` ≡ `f` in Python |
| `pf`       | `(r,fp,a)`    | `pf`            | `(r,f,a)`       | ✓ direct        |       |
| `xf`       | `(r,fp,a)`    | `xf`            | `(r,f,a)`       | ✓ direct        |       |
| `pa`       | `(r,i,aa)`    | `pa`            | `(r,i,aa)`      | ✓ direct        |       |
| `xa`       | `(r,i,aa)`    | `xaa`           | `(r,i,aa)`      | ✓ renamed       | GAMS `xa` → Python `xaa` |
| `pe`       | `(r,i,rp)`    | `pe`            | `(r,i,rp)`      | ✓ direct        |       |
| `pefob`    | `(r,i,rp)`    | `pefob`         | `(r,i,rp)`      | ✓ direct        |       |
| `pmcif`    | `(r,i,rp)`    | `pmcif`         | `(rp,i,r)`      | ⚠ dim swap      | Same dataset, reversed index order; iterate `for (i,j,k): pmcif_py[k,i,j].value = pmcif_neos[i,j,k]` semantics |
| `pm`       | `(r,i,rp)`    | `pm`            | `(rp,i,r)`      | ⚠ dim swap      | Same swap as `pmcif` |
| `xw`       | `(r,i,rp)`    | `xw`            | `(r,i,rp)`      | ✓ direct        |       |
| `ptmg`     | `(m,)`        | `ptmg`          | `(m,)`          | ✓ direct        |       |
| `psave`    | `(r,)`        | `psave`         | `(r,)`          | ✓ direct        |       |
| `pi`       | `(r,)`        | `pi`            | `(r,)`          | ✓ direct        |       |
| `uh`       | `(r,h)`       | `uh`            | `(r,)`          | ⚠ dim drop      | Python has no household dim — single representative household per region. Fix `uh[r]` from prev `uh[r]` (or skip; uh is utility level, may not need fixing in single-household model) |
| `pabs`     | `(r,)`        | `pabs`          | `(r,)`          | ✓ direct        |       |
| `pmuv`     | `()`          | `pmuv`          | (Param mutable) | ⏸ skip          | Python has `pmuv` as a calibrated Param when rmuv/imuv basket empty; becomes Var only with basket. NOT fixed in C.1 unless basket is configured |
| `pfact`    | `(r,)`        | `pfact`         | `(r,)`          | ✓ direct        |       |
| `pwfact`   | `()`          | `pwfact`        | `()`            | ✓ direct        |       |
| `gdpmp`    | `(r,)`        | `gdpmp`         | `(r,)`          | ✓ direct        |       |
| `rgdpmp`   | `(r,)`        | `rgdpmp`        | `(r,)`          | ✓ direct        |       |
| `pgdpmp`   | `(r,)`        | `pgdpmp`        | `(r,)`          | ✓ direct        |       |

## Summary

- **22 Vars to fix:** `axp, lambdand, lambdava, aioall, lambdaf, pf, xf, pa, xaa, pe, pefob, pmcif, pm, xw, ptmg, psave, pi, uh, pabs, pfact, pwfact, gdpmp, rgdpmp, pgdpmp` (the rename `aioall←lambdaio` and `xaa←xa` are already resolved)
- **2 Vars need dim-swap copy:** `pmcif`, `pm` — when copying `prev.pmcif[rp,i,r].value` into `new.pmcif[rp,i,r]`, the dims are the same on both sides so the copy is direct — the GAMS label `(r,i,rp)` is just GAMS' convention. **No special handling needed; both Python sides agree on `(rp,i,r)`.**
- **1 Var with dim drop:** `uh` — Python has no `h`; copy `prev.uh[r].value → new.uh[r]`.
- **1 to skip:** `pmuv` (Param, not Var, in default closure).

## Implementation note for Phase C.2

The function signature should be:

```python
def fix_lagged_state(new_model, prev_model_or_snapshot, lagged_var_names: list[str] = LAGGED_VARS) -> int:
    """Fix new_model's lagged-state Vars from prev_model's resolved values.

    Mirrors GAMS iterloop.gms:151-182. Iterates `lagged_var_names`; for each, copies
    `prev_model.<name>[idx].value` to `new_model.<name>[idx]` and calls `.fix()`.
    Returns the number of (var, idx) tuples fixed.
    """
```

Where `LAGGED_VARS` is the list of 22 names above. The function is dim-agnostic — Pyomo's
`getattr(model, name)` returns the indexed Var; iterating it yields the same index tuples
for both sides because both are the same Pyomo model class with the same set declarations.
