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

*Generated `2026-05-10T01:23:25Z` from commit `6adffea`.*

### Summary

| Phase | Vars matched | Cells | Match | Diverge | Missing | Match rate | Residual |
|-------|--------------|-------|-------|---------|---------|------------|----------|
| `base` | 95/138 | 59958 | 45750 | 767 | 13441 | 76.30% | 1.62e-11 |
| `shock` | 86/138 | 59978 | 42578 | 3959 | 13441 | 70.99% | 1.84e-09 |

### Top diverging variables — `base`

| Var | Py var | Cells | Diverge | Max abs err | Max rel err |
|-----|--------|-------|---------|-------------|-------------|
| `pp` | `pp_rai` | 900 | 756 | 8.764665e-02 | 8.764665e-02 |
| `rore` | `rore` | 10 | 10 | 6.087311e-02 | 1.166705e+00 |
| `rorg` | `rorg` | 1 | 1 | 2.283519e-02 | 4.894321e-01 |

### Top diverging variables — `shock`

| Var | Py var | Cells | Diverge | Max abs err | Max rel err |
|-----|--------|-------|---------|-------------|-------------|
| `pm` | `pm` | 1000 | 810 | 1.415631e-01 | 5.635153e-02 |
| `pp` | `pp_rai` | 900 | 813 | 8.764665e-02 | 8.764665e-02 |
| `rore` | `rore` | 10 | 10 | 6.064800e-02 | 1.164146e+00 |
| `pfa` | `pfa` | 450 | 162 | 2.719859e-02 | 2.440073e-02 |
| `pft` | `pft` | 40 | 9 | 2.381952e-02 | 2.440073e-02 |
| `pfy` | `pfy` | 450 | 162 | 2.381952e-02 | 2.440073e-02 |
| `rorg` | `rorg` | 1 | 1 | 2.277464e-02 | 4.886980e-01 |
| `cv` | `cv` | 10 | 5 | 8.117602e-03 | 4.306450e-03 |
| `pmcif` | `pmcif` | 1000 | 567 | 4.865874e-03 | 4.485158e-03 |
| `pefob` | `pefob` | 1000 | 570 | 4.864470e-03 | 4.552920e-03 |

## GTAP Standard 7 — NUS333 (3 sectors × 3 regions × 3 factors)

Reference: `output/nus333_neos/out.gdx` (NEOS job 18744693, power-scaled 10% imptx shock, residual region `ROW`).

*Generated `2026-05-10T01:24:17Z` from commit `6adffea`.*

### Summary

| Phase | Vars matched | Cells | Match | Diverge | Missing | Match rate | Residual |
|-------|--------------|-------|-------|---------|---------|------------|----------|
| `base` | 97/138 | 1304 | 892 | 6 | 406 | 68.40% | 8.93e-10 |
| `shock` | 93/138 | 1310 | 885 | 19 | 406 | 67.56% | 2.08e-07 |

### Top diverging variables — `base`

| Var | Py var | Cells | Diverge | Max abs err | Max rel err |
|-----|--------|-------|---------|-------------|-------------|
| `pwmg` | `pwmg` | 12 | 6 | 1.000000e+00 | 1.000000e+00 |

### Top diverging variables — `shock`

| Var | Py var | Cells | Diverge | Max abs err | Max rel err |
|-----|--------|-------|---------|-------------|-------------|
| `pwmg` | `pwmg` | 12 | 6 | 1.000000e+00 | 1.000000e+00 |
| `cv` | `cv` | 2 | 2 | 6.227103e-01 | 4.257762e-02 |
| `ev` | `ev` | 2 | 1 | 2.500340e-01 | 1.077537e-02 |
| `pmp` | `pmp` | 42 | 6 | 1.324174e-01 | 1.324174e-01 |
| `pdp` | `pdp` | 42 | 4 | 4.999795e-02 | 4.999795e-02 |

