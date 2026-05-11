# Benchmarks

Variable-by-variable parity between the Python `equilibria` GTAP
Standard 7 implementation and the reference GAMS runs (NEOS), for both the
9x10 and NUS333 datasets. Numbers come from CSVs committed under
`docs/site/_data/benchmarks/` — Read the Docs renders this page from those
files (it has no GAMS/PATH installed). Regenerate locally with:

```bash
make benchmarks
git add docs/site/_data/benchmarks/*.csv
git commit -m "benchmarks: refresh"
```

Each row reports, for one (dataset, phase, variable) triple, how many
Pyomo Var cells match GAMS within `tol_rel=1e-3 / tol_abs=1e-6` and the
worst absolute / relative error observed. The `__SUMMARY__` rows in the
underlying CSV hold per-phase totals.


## GTAP Standard 7 — 9 sectors × 10 regions

Reference: `src/equilibria/templates/reference/gtap/output/COMP.gdx` (rate-scaled 10% imptx shock, `if_sub=False`, `rorflex=10`).

*Generated `2026-05-11T01:10:43Z` from commit `74a114d`.*

### Summary

| Phase | Vars matched | Cells | Match | Diverge | Missing | Match rate | Residual |
|-------|--------------|-------|-------|---------|---------|------------|----------|
| `base` | 138/138 | 59958 | 59958 | 0 | 0 | 100.00% | 2.22e-11 |
| `shock` | 138/138 | 59978 | 59978 | 0 | 0 | 100.00% | 6.02e-13 |

## GTAP Standard 7 — NUS333 (3 sectors × 3 regions × 3 factors)

Reference: `output/nus333_neos/out.gdx` (NEOS job 18744693, power-scaled 10% imptx shock, residual region `ROW`).

*Generated `2026-05-11T01:10:48Z` from commit `74a114d`.*

### Summary

| Phase | Vars matched | Cells | Match | Diverge | Missing | Match rate | Residual |
|-------|--------------|-------|-------|---------|---------|------------|----------|
| `base` | 138/138 | 1304 | 1304 | 0 | 0 | 100.00% | 1.98e-11 |
| `shock` | 138/138 | 1310 | 1310 | 0 | 0 | 100.00% | 2.08e-07 |

