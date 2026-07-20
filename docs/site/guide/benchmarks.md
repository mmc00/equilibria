# Benchmarks

Variable-by-variable parity between the Python `equilibria` GTAP
Standard 7 implementation and reference GAMS runs, plus wall-time benchmarks
when GAMS can run locally. Numbers come from CSVs committed under
`docs/site/_data/benchmarks/` — Read the Docs renders this page from those
files (it has no GAMS/PATH installed). Regenerate locally with:

```bash
make benchmarks           # all datasets (parity + MCP wall-time)
make benchmarks-nus333    # NUS333 only (also produces local parity + timing)
make benchmarks-nlp       # NLP wall-time: Python IPOPT vs GAMS local IPOPT
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

## NLP wall-time (Python IPOPT vs GAMS local IPOPT)

The MCP path uses PATH, which the GAMS community license caps at 1000 rows — so anything larger than ~NUS333 must go to NEOS and cannot be timed head-to-head locally. The **NLP path uses IPOPT, an open-source solver GAMS does *not* license-cap**, so both sides run on the same host up to `gtap7_15x10`. Python solves the full base→check→shock sequence with `EQUILIBRIA_GTAP_SOLVE_NLP=1`; GAMS runs the same bundle with `ifMCP=0` + `option nlp=ipopt`. Regenerate with `make benchmarks-nlp` (from `nlp_timing.csv`).

*Generated `2026-07-20T15:32:52Z` from commit `8e55b9f`. Warm-up run discarded; N timed runs per side. Lower is better.*

```{raw} html
<div class="mx-card"><div class="mx-scroll"><table class="mx-table"><thead><tr><th class="mx-lbl">Dataset</th><th>Mode</th><th>ifSUB</th><th>N</th><th>Python median</th><th>GAMS median</th><th>Python / GAMS</th></tr></thead><tbody><tr><td class="mx-lbl"><span class="mx-ds">gtap7_3x3</span><span class="mx-sub">pure</span></td><td>pure</td><td>0</td><td>5</td><td>0.331s</td><td>0.809s</td><td><span class="mx-num mx-good">0.41×</span></td></tr><tr><td class="mx-lbl"><span class="mx-ds">gtap7_3x3</span><span class="mx-sub">altertax</span></td><td>altertax</td><td>0</td><td>5</td><td>0.320s</td><td>0.878s</td><td><span class="mx-num mx-good">0.36×</span></td></tr><tr><td class="mx-lbl"><span class="mx-ds">gtap7_3x4</span><span class="mx-sub">altertax</span></td><td>altertax</td><td>0</td><td>5</td><td>0.461s</td><td>1.015s</td><td><span class="mx-num mx-good">0.45×</span></td></tr><tr><td class="mx-lbl"><span class="mx-ds">gtap7_5x5</span><span class="mx-sub">pure</span></td><td>pure</td><td>0</td><td>5</td><td>1.078s</td><td>1.244s</td><td><span class="mx-num mx-good">0.87×</span></td></tr><tr><td class="mx-lbl"><span class="mx-ds">gtap7_5x5</span><span class="mx-sub">altertax</span></td><td>altertax</td><td>0</td><td>5</td><td>2.046s</td><td>2.027s</td><td><span class="mx-num mx-warn">1.01×</span></td></tr><tr><td class="mx-lbl"><span class="mx-ds">gtap7_10x7</span><span class="mx-sub">pure</span></td><td>pure</td><td>0</td><td>5</td><td>13.050s</td><td>7.113s</td><td><span class="mx-num mx-warn">1.83×</span></td></tr><tr><td class="mx-lbl"><span class="mx-ds">gtap7_10x7</span><span class="mx-sub">altertax</span></td><td>altertax</td><td>0</td><td>5</td><td>29.293s</td><td>21.352s</td><td><span class="mx-num mx-warn">1.37×</span></td></tr><tr><td class="mx-lbl"><span class="mx-ds">gtap7_15x10</span><span class="mx-sub">altertax</span></td><td>altertax</td><td>0</td><td>5</td><td>494.149s</td><td><span class="mx-ref">no local ref</span></td><td>—</td></tr></tbody></table></div></div>
```

```{raw} html
<div class="mx-note"><span>⤷</span><span>A ratio ≤ 1× means Python is at least as fast as GAMS-local on that row. Rows with <b>no local ref</b> are timed Python-only — they exceed what a local GAMS NLP reference was generated for, but IPOPT still solves them in-process (the historical large-model hang was PATH-specific, not IPOPT).</span></div>
```

