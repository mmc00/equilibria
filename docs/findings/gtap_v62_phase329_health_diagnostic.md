# GTAP v6.2 Phase 3.29 — Health diagnostic helper (early singularity detection)

**Date:** 2026-05-25
**Branch:** `gtap/v62-rollback`
**Scope:** v6.2 only (Standard 7 in `templates/gtap/` unchanged)

## TL;DR

New module `scripts/gtap_v62/diagnose_health.py` with two diagnostic
functions that flag near-singular Jacobians and identify the specific
(commodity, region) cells responsible — **before** PATH wastes time on
a doomed solve.

Wired into `validate_v62_parity.py`: runs automatically when
`--solver path-capi` is used, or always when `V62_ALWAYS_DIAGNOSE=1`
env var is set.

## Two-stage diagnosis

### Stage 1 — Data-level (cheap)

`check_dataset_for_path(sets, params)` inspects the SAM data without
building a Pyomo model. Looks for:

- **% of zero-flow trade routes** (e.g., USA→Vanuatu = 0)
- **% of near-zero flows** (< 0.1% of mean)
- **Intra-region (diagonal) trade share** (`VXMD[i, r, r]`)
- **Intra-region margin share** (`VTWR[m, i, r, r]`)
- **Trade-isolated regions** (no exports or no imports)

Emits severity tags `ok` / `warn` / `error` and a list of findings.

### Stage 2 — Jacobian-level (medium cost, requires Pyomo build)

`check_jacobian_singularity(model, params, top_k=10)` builds the
sparse Jacobian at the closure-applied baseline state and computes:

- Row 2-norm `||J[i, :]||₂` for each equation
- Column 2-norm `||J[:, j]||₂` for each variable
- Estimated condition number `κ(J) ≈ σ_max / σ_min`
- `‖F(x₀)‖` — current residual magnitude

Reports the **`top_k` weakest rows/columns mapped back** to:
- `(equation_name, index)` for rows
- `(variable_name, index)` for columns

These are the precise model cells responsible for near-singularity.

## Sample output — BOOK3X3 (the legacy v6.2 dataset)

```
✗ v6.2 jacobian-level health: ERROR
  n_vars: 617  nnz: 2172  density: 0.5705%
  σ_max ≈ 9.45e+06  σ_min ≈ 2.35e-09  κ ≈ 4.02e+15
  ‖F(x₀)‖ = 1.14e-03

  Weakest equations (smallest row norm):
    • eq_qxs[('svces', 'EU', 'EU')]   row_norm=2.35e-09 ← dead row!
    • eq_qxs[('svces', 'USA', 'USA')] row_norm=2.35e-09 ← dead row!
    • eq_pcons[USA]    row_norm=9.32e-02
    • eq_pcons[EU]     row_norm=1.62e-01
    • eq_pva[('food', 'ROW')] row_norm=2.46e-01

  Weakest variables (smallest col norm):
    • pds[('CGDS', 'EU')]   col_norm=1.00e+00, val=1.00e+00
    • pds[('CGDS', 'ROW')]  col_norm=1.00e+00, val=1.00e+00
    • qds[('svces', 'ROW')] col_norm=1.00e+00, val=9.84e+06

  Findings:
    ✗ [near_singular] Smallest row norm = 2.35e-09 (≈0). Jacobian is
      effectively singular. PATH WILL FAIL with term_code=2 on shocked.
    ✗ [ill_conditioned] κ(J) ≈ 4.02e+15. Newton steps unreliable.
    ℹ [weak_eq_family] Top problem family: eq_pva (4/8 weakest rows).

  Recommendation:
    Near-singular Jacobian. Switch to IPOPT or rebuild closure.
      → most degenerate eq: eq_qxs('svces','EU','EU') row_norm=2.35e-09
      → most degenerate var: pds('CGDS','EU') col_norm=1.00, val=1.0
```

## Sample output — gtap6_3x3 (GtapAgg-generated, same dimension)

```
✓ v6.2 jacobian-level health: OK
  n_vars: 663  nnz: 2357  density: 0.5362%
  σ_max ≈ 5.70e+07  σ_min ≈ 9.02e-02  κ ≈ 6.32e+08

  Findings:
    ℹ [weak_eq_family] Top problem family: eq_pva (8/10 weakest rows).
```

Same dimension (3×3), same Pyomo equations, but **κ(J) is 10⁷ times
better** (4e+15 → 6e+8). The legacy BOOK3X3 SAM has structural
pathologies that the modern GtapAgg-generated SAMs don't.

## Key finding: the singularity isn't (just) about aggregation size

Initially I assumed PATH failed on larger datasets because aggregation
fineness brings in more near-zero trade routes. Phase 3.29 diagnostic
reveals a different story:

| Dataset    | Vars  | κ(J)     | σ_min   | PATH BASELINE | PATH SHOCKED |
|:-----------|------:|---------:|--------:|:-------------|:-------------|
| BOOK3X3    | 617   | 4.02e+15 | 2.35e-09 | term_code=2 fail/0.001 | term_code=2 |
| gtap6_3x3  | 663   | 6.32e+08 | 9.02e-02 | **term_code=1** in 1 iter | term_code=2 |
| gtap6_5x5  | 2,239 | 2.38e+08 | 1.13e-01 | **term_code=1** in 86 iter | term_code=2 |

**BOOK3X3** has a structural defect: `eq_qxs[svces, r, r]` (intra-region
trade in the margin commodity) has row norm ≈ 2.35e-09 = numerically
zero. This makes J singular regardless of size. Every legacy v6.2
PATH failure since Phase 3.7 has been driven by this.

**gtap6 datasets** built via GtapAgg from current GTAP v11.1 data are
SAM-balanced at the bilateral cell level and don't have these
pathologies. PATH baseline converges to term_code=1 on the smaller
ones.

The SHOCKED failure (still term_code=2 on all datasets) is a separate
issue: post-shock `‖F(x_0)‖` jumps from ~1e-4 to ~700 in the merit
function, and PATH's line-search rejects every Newton step from that
starting point. This is partially addressable by warm-start / homotopy
(see Phase 3.30 candidate).

## Identified problem equations / variables

The diagnostic gives **specific Pyomo identifiers** that pinpoint the
issue:

| Pattern observed | What it means | Where to look |
|:-----------------|:--------------|:--------------|
| `eq_qxs[('svces', r, r)]` row_norm = 2e-9 | Intra-region margin trade equation is degenerate (svces is BOOK3X3's margin commodity) | `gtap_v62_calibration.py` margin block; consider amgm[m,i,r,r]=0 enforcement |
| `eq_pva[(j, r)]` row_norm < 0.5 | VA price aggregator weak for sector with low share_va | `gtap_v62_model_equations.py` `_add_production_block` |
| `pds[('cgds', r)]` col_norm = 1.0 (only diag) | CGDS sector has only trivial pds coupling (no factors) | `eq_qo_rule` for cgds path |
| `pf_int[(c, 'cgds', r)]` col_norm = 0.0 | Energy intermediate price into CGDS has zero coupling | `_add_intermediates_block` |

## Recommendation tags

The diagnostic emits one of:

- **`ok`**: Jacobian well-conditioned. PATH should work for baseline,
  may struggle on large shocks.
- **`warn`**: Marginal conditioning. PATH baseline OK, shocked may
  fail. Use IPOPT for production.
- **`error`**: Near-singular Jacobian. PATH WILL FAIL term_code=2 on
  shocked. Use IPOPT or rebuild closure.

## Integration

`validate_v62_parity.py` runs the diagnostic automatically:

```python
if pyomo_mode == "mcp" or os.environ.get("V62_ALWAYS_DIAGNOSE"):
    from diagnose_health import check_dataset_for_path, ...
    data_h = check_dataset_for_path(sets, params)
    print_health_report(data_h)
    ...
    jac_h = check_jacobian_singularity(model, params, top_k=8)
    print_health_report(jac_h)
```

For other scripts (test_cross_dataset.py, benchmark_cross_dataset.py),
just import and call:

```python
from diagnose_health import check_dataset_for_path, check_jacobian_singularity
data_h = check_dataset_for_path(sets, params)  # right after loading
# ... build + closure ...
jac_h = check_jacobian_singularity(model, params)  # before any solve
```

## Reproduce

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:V62_ALWAYS_DIAGNOSE = "1"  # force diagnostic on IPOPT too

# Run validate; diagnostic prints before solve:
python scripts/gtap_v62/validate_v62_parity.py shock `
    --experiment Exp1a --solver path-capi `
    --workdir runs/gtap_v62_parity/BOOK3X3_Exp1a
```

Output will include the data-level and Jacobian-level health reports,
with weakest (eq, var) lists.

## Status

| Feature | Status |
|--------|--------|
| Data-level diagnostic | ✓ implemented |
| Jacobian-level diagnostic | ✓ implemented |
| Identifies specific (eq, var) cells | ✓ via Pyomo names/indices |
| Auto-wired into validate_v62_parity (path-capi) | ✓ |
| Manual API for other scripts | ✓ via direct imports |
| BOOK3X3 singularity correctly flagged | ✓ κ ≈ 4e+15 detected |
| gtap6_3x3 healthy verdict | ✓ κ ≈ 6e+8 (OK) |

## Next steps

Phase 3.29 only DIAGNOSES; it doesn't fix. The natural follow-ups are:

- **Phase 3.30** — Drop the dead rows automatically. When
  `eq_qxs[svces, r, r]` row norm < 1e-6, deactivate the constraint
  and fix the corresponding free var. This removes the singularity
  from the active system.
- **Phase 3.31** — Implement Gragg-style multi-step homotopy with
  chained warm-start (à la v7's `apply_solution_hint`). This addresses
  the SHOCKED convergence even on a well-conditioned baseline.

Both are 1-2 day implementations on top of Phase 3.29.
