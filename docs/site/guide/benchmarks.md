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

---

## macOS — Apple Silicon M3, 18 GB RAM

> **Host:** MacBook (Apple M3, 8-core CPU, 18 GB RAM), macOS 15,
> Python 3.12, GAMS 53.

### GTAP Standard 7 — 9 sectors × 10 regions

Reference: `src/equilibria/templates/reference/gtap/output/COMP.gdx` (rate-scaled 10% imptx shock, `if_sub=False`, `rorflex=10`).

*Generated `2026-05-11T01:38:29Z` from commit `1ba8d6d`.*

#### Parity vs GAMS NEOS reference

| Phase | Vars matched | Cells | Match | Diverge | Missing | Match rate | Residual | Solve time |
|-------|--------------|-------|-------|---------|---------|------------|----------|------------|
| `base` | 138/138 | 59958 | 59958 | 0 | 0 | 100.00% | 2.22e-11 | 6.57s |
| `shock` | 138/138 | 59978 | 59978 | 0 | 0 | 100.00% | 6.02e-13 | 7.59s |

> ℹ️ **GAMS-local parity not available for 9x10.** The model has ~10k equations and exceeds the GAMS community-license limit of 2500 rows/cols for nonlinear models. Only the NEOS reference run is used for 9x10.

### GTAP Standard 7 — NUS333 (3 sectors × 3 regions × 3 factors)

Reference: `output/nus333_neos/out.gdx` (NEOS job 18744693, power-scaled 10% imptx shock, residual region `ROW`).

*Generated `2026-05-12T00:09:18Z` from commit `e5b9385`.*

#### Parity vs GAMS NEOS reference

| Phase | Vars matched | Cells | Match | Diverge | Missing | Match rate | Residual | Solve time |
|-------|--------------|-------|-------|---------|---------|------------|----------|------------|
| `base` | 138/138 | 1304 | 1304 | 0 | 0 | 100.00% | 1.98e-11 | 0.32s |
| `shock` | 138/138 | 1310 | 1310 | 0 | 0 | 100.00% | 2.08e-07 | 0.32s |

#### Parity vs GAMS local

| Phase | Vars matched | Cells | Match | Diverge | Missing | Match rate | Residual | Solve time |
|-------|--------------|-------|-------|---------|---------|------------|----------|------------|
| `base` | 138/138 | 1304 | 1304 | 0 | 0 | 100.00% | 1.98e-11 | 0.32s |
| `shock` | 138/138 | 1310 | 1310 | 0 | 0 | 100.00% | 2.08e-07 | 0.32s |

#### Wall-time benchmark

Median / min / max / mean across the runs in `nus333_timing.csv`. The warm-up run is discarded — both sides solve from cold state then are re-run N times. Lower is better.

| Solver | N | Median | Min | Max | Mean |
|--------|---|--------|-----|-----|------|
| Python `equilibria` (PATH C API, nonlinear full) | 5 | 0.644s | 0.608s | 0.702s | 0.643s |
| GAMS local (`comp_nus333.gms`, PATH via GAMS 53) | 5 | 0.848s | 0.769s | 0.917s | 0.831s |

*Median ratio Python / GAMS-local: **0.760×***

---

## Windows

> **Host:** *(pending — re-run `make benchmarks` on a Windows host and
> populate this section. Suggested template line: Intel/AMD CPU model,
> N cores, M GB RAM, Windows 11, Python 3.12, GAMS 53.)*

### GTAP Standard 7 — 9 sectors × 10 regions

*Pending. Run `make benchmarks` on Windows to fill this in.*

#### Parity vs GAMS NEOS reference

| Phase | Vars matched | Cells | Match | Diverge | Missing | Match rate | Residual | Solve time |
|-------|--------------|-------|-------|---------|---------|------------|----------|------------|
| `base` | — | — | — | — | — | — | — | — |
| `shock` | — | — | — | — | — | — | — | — |

### GTAP Standard 7 — NUS333

*Pending.*

#### Parity vs GAMS NEOS reference

| Phase | Vars matched | Cells | Match | Diverge | Missing | Match rate | Residual | Solve time |
|-------|--------------|-------|-------|---------|---------|------------|----------|------------|
| `base` | — | — | — | — | — | — | — | — |
| `shock` | — | — | — | — | — | — | — | — |

#### Parity vs GAMS local

*Pending — requires GAMS license on Windows host.*

#### Wall-time benchmark

| Solver | N | Median | Min | Max | Mean |
|--------|---|--------|-----|-----|------|
| Python `equilibria` (PATH C API, nonlinear full) | — | — | — | — | — |
| GAMS local (`comp_nus333.gms`, PATH via GAMS 53)  | — | — | — | — | — |

---

## Ubuntu Linux

> **Host:** *(pending — re-run `make benchmarks` on an Ubuntu host and
> populate this section. Suggested template line: Intel/AMD CPU model,
> N cores, M GB RAM, Ubuntu 24.04, Python 3.12, GAMS 53.)*

### GTAP Standard 7 — 9 sectors × 10 regions

*Pending. Run `make benchmarks` on Ubuntu to fill this in.*

#### Parity vs GAMS NEOS reference

| Phase | Vars matched | Cells | Match | Diverge | Missing | Match rate | Residual | Solve time |
|-------|--------------|-------|-------|---------|---------|------------|----------|------------|
| `base` | — | — | — | — | — | — | — | — |
| `shock` | — | — | — | — | — | — | — | — |

### GTAP Standard 7 — NUS333

*Pending.*

#### Parity vs GAMS NEOS reference

| Phase | Vars matched | Cells | Match | Diverge | Missing | Match rate | Residual | Solve time |
|-------|--------------|-------|-------|---------|---------|------------|----------|------------|
| `base` | — | — | — | — | — | — | — | — |
| `shock` | — | — | — | — | — | — | — | — |

#### Parity vs GAMS local

*Pending — requires GAMS license on Ubuntu host.*

#### Wall-time benchmark

| Solver | N | Median | Min | Max | Mean |
|--------|---|--------|-----|-----|------|
| Python `equilibria` (PATH C API, nonlinear full) | — | — | — | — | — |
| GAMS local (`comp_nus333.gms`, PATH via GAMS 53)  | — | — | — | — | — |

