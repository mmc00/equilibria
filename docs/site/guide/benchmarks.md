# Benchmarks

Variable-by-variable parity between the Python `equilibria` GTAP
Standard 7 implementation and reference GAMS runs, plus wall-time benchmarks
when GAMS can run locally. Numbers come from CSVs committed under
`docs/site/_data/benchmarks/` — Read the Docs renders this page from those
files (it has no GAMS/PATH installed). Regenerate locally with:

```bash
make benchmarks           # all datasets
make benchmarks-nus333    # NUS333 only (also produces local parity + timing)
```

The default number of timing runs is `BENCH_RUNS=5` (override on the
make command line).

Each parity row reports, for one (dataset, phase, variable) triple, how
many Pyomo Var cells match GAMS within `tol_rel=1e-3 / tol_abs=1e-6`
and the worst absolute / relative error observed. The `__SUMMARY__` rows
in the underlying CSV hold per-phase totals.

> **Hardware sensitivity:** wall-time numbers depend on CPU, memory and
> filesystem. Parity (cell-level matching vs GAMS) is *deterministic*
> and identical across platforms, but solve times vary. Each section
> below is labelled with the host that produced it. Only compare *ratios*
> (Python vs GAMS-local) across machines.

## Coverage matrix

The authoritative parity-coverage matrix (dataset × kind × ifSUB × phase,
with per-row gap thresholds and CI status) is generated from
`scripts/gtap/coverage_matrix.py`: see
[GTAP 7 Parity Coverage Matrix](gtap7_coverage_matrix.md).


## GTAP Standard 7 — 9 sectors × 10 regions

Reference: `src/equilibria/templates/reference/gtap/output/COMP.gdx` (rate-scaled 10% imptx shock, `if_sub=False`, `rorflex=10`).

*Generated `2026-05-11T01:38:29Z` from commit `1ba8d6d`.*

### Parity vs GAMS NEOS reference

```{raw} html
<div class="mx-card"><div class="mx-scroll"><table class="mx-table"><thead><tr><th class="mx-lbl">Phase</th><th>Vars matched</th><th>Cells</th><th>Match</th><th>Diverge</th><th>Missing</th><th>Match rate</th><th>Residual</th><th>Solve time</th></tr></thead><tbody><tr><td class="mx-lbl"><span class="mx-ds">base</span></td><td>138/138</td><td>59958</td><td>59958</td><td>0</td><td>0</td><td><div class="mx-cell"><span class="mx-num mx-good">100.00%</span><span class="mx-chip mx-good">✓ match</span></div></td><td><span class="mx-ref">2.22e-11</span></td><td><span class="mx-ref">6.57s</span></td></tr><tr><td class="mx-lbl"><span class="mx-ds">shock</span></td><td>138/138</td><td>59978</td><td>59978</td><td>0</td><td>0</td><td><div class="mx-cell"><span class="mx-num mx-good">100.00%</span><span class="mx-chip mx-good">✓ match</span></div></td><td><span class="mx-ref">6.02e-13</span></td><td><span class="mx-ref">7.59s</span></td></tr></tbody></table></div></div>
```

> ℹ️ **GAMS-local parity not available for 9x10.** The model has ~10k equations and exceeds the GAMS community-license limit of 2500 rows/cols for nonlinear models. Only the NEOS reference run is used for 9x10.

## GTAP Standard 7 — NUS333 (3 sectors × 3 regions × 3 factors)

Reference: `output/nus333_neos/out.gdx` (NEOS job 18744693, power-scaled 10% imptx shock, residual region `ROW`).

*Generated `2026-05-12T00:09:18Z` from commit `e5b9385`.*

### Parity vs GAMS NEOS reference

```{raw} html
<div class="mx-card"><div class="mx-scroll"><table class="mx-table"><thead><tr><th class="mx-lbl">Phase</th><th>Vars matched</th><th>Cells</th><th>Match</th><th>Diverge</th><th>Missing</th><th>Match rate</th><th>Residual</th><th>Solve time</th></tr></thead><tbody><tr><td class="mx-lbl"><span class="mx-ds">base</span></td><td>138/138</td><td>1304</td><td>1304</td><td>0</td><td>0</td><td><div class="mx-cell"><span class="mx-num mx-good">100.00%</span><span class="mx-chip mx-good">✓ match</span></div></td><td><span class="mx-ref">1.98e-11</span></td><td><span class="mx-ref">0.32s</span></td></tr><tr><td class="mx-lbl"><span class="mx-ds">shock</span></td><td>138/138</td><td>1310</td><td>1310</td><td>0</td><td>0</td><td><div class="mx-cell"><span class="mx-num mx-good">100.00%</span><span class="mx-chip mx-good">✓ match</span></div></td><td><span class="mx-ref">2.08e-07</span></td><td><span class="mx-ref">0.32s</span></td></tr></tbody></table></div></div>
```

### Parity vs GAMS local

```{raw} html
<div class="mx-card"><div class="mx-scroll"><table class="mx-table"><thead><tr><th class="mx-lbl">Phase</th><th>Vars matched</th><th>Cells</th><th>Match</th><th>Diverge</th><th>Missing</th><th>Match rate</th><th>Residual</th><th>Solve time</th></tr></thead><tbody><tr><td class="mx-lbl"><span class="mx-ds">base</span></td><td>138/138</td><td>1304</td><td>1304</td><td>0</td><td>0</td><td><div class="mx-cell"><span class="mx-num mx-good">100.00%</span><span class="mx-chip mx-good">✓ match</span></div></td><td><span class="mx-ref">1.98e-11</span></td><td><span class="mx-ref">0.32s</span></td></tr><tr><td class="mx-lbl"><span class="mx-ds">shock</span></td><td>138/138</td><td>1310</td><td>1310</td><td>0</td><td>0</td><td><div class="mx-cell"><span class="mx-num mx-good">100.00%</span><span class="mx-chip mx-good">✓ match</span></div></td><td><span class="mx-ref">2.08e-07</span></td><td><span class="mx-ref">0.32s</span></td></tr></tbody></table></div></div>
```

### Wall-time benchmark

Median / min / max / mean across the runs in `nus333_timing.csv`. The warm-up run is discarded — both sides solve from cold state then are re-run N times. Lower is better.

```{raw} html
<div class="mx-card"><div class="mx-scroll"><table class="mx-table"><thead><tr><th class="mx-lbl">Solver</th><th>N</th><th>Median</th><th>Min</th><th>Max</th><th>Mean</th></tr></thead><tbody><tr><td class="mx-lbl"><span class="mx-ds">Python equilibria</span><span class="mx-sub">PATH C API, nonlinear full</span></td><td>5</td><td>0.644s</td><td>0.608s</td><td>0.702s</td><td>0.643s</td></tr><tr><td class="mx-lbl"><span class="mx-ds">GAMS local</span><span class="mx-sub">comp_nus333.gms, PATH via GAMS 53</span></td><td>5</td><td>0.848s</td><td>0.769s</td><td>0.917s</td><td>0.831s</td></tr></tbody></table></div></div>
```

```{raw} html
<div class="mx-note"><span>⤷</span><span>Median ratio Python / GAMS-local: <b>0.760×</b></span></div>
```

