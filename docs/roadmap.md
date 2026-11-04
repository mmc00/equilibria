# equilibria roadmap

The real, measured state of the project — not an aspiration. Every figure here comes from a
run recorded in `docs/archive/findings/`, from a gate of `scripts/gtap/run_parity_gates.py` or from
`scripts/paper/run_tables.py`.

Last updated: 2026-10-27.

Each topic carries an ID (`R-…`) that the author's internal roadmap uses too, so the two
documents do not drift apart.

---

## Priorities

| Priority | ID | What | Status |
|---|---|---|---|
| P1 | `R-PYPI` | Publish on PyPI | Pending: the package **is not on PyPI yet**; today it installs from the repository |
| P1 | `R-BOUND-WARN` | Warn when a solve ends with `code=1` and a price resting on its 1e-3 floor | Pending (see Technical debt) |
| P1 | `R-DIAG` | Solve diagnostics: a report that says why it did not solve (MCP count/pairing, domain error, price at a bound, stalling, seed, closure) and an `equilibria doctor` for missing solvers | Pending. Today there are parity tools against GAMS (`scripts/parity/triage.py`), not a diagnostic for the user |
| P1 | `R-NL-GATE` | Make the `.nl` gate actually compare (#23) | Pending (see Known blockers) |
| P2 | `R-MONOLITH` | Retire the monolith `gtap_model_equations.py` | Pending (see Known blockers) |
| P2 | `R-DRIVER` | A driver that only orchestrates periods, with one recipe per model, usable by PEP too | Pending |
| P2 | `R-DYN` | Periods as a list → recursive dynamic GTAP | Pending (Next 1) |
| P2 | `R-CALIB` | Each block calibrates its own part from a common data interface, as in MPSGE: the block declares which SAM rows and columns its indices map to | Pending (Next 2) |
| P2 | `R-PMUV` | Port `eq_pmuv` to blocks | Pending (see Known blockers) |
| P2 | `R-BOOK` | Compare against the printed figures of Burfisher's book | Pending (see Validation) |
| P3 | `R-PERF` | Alternative evaluation backend (`PERF-EVAL-1`) | Design, no code |
| P3 | `R-HAR-VIEW` | A HAR/`.sl4` viewer for any operating system | Pending |
| P4 | `R-ENV` | An environmental block as a modularity test | Pending (Next 3) |

---

## What is closed

### GTAP validation against GEMPACK

A uniform 10 % import tariff, compared in levels cell by cell against GEMPACK
(Gragg 8-16-32), with GEMPACK's default closure (`capFlex`, RORDELTA = 1). Measured with
`scripts/gtap/compare_gempack_levels.py` (run by `scripts/paper/run_tables.py`); every
run ends with `code=1`:

| Dataset | Solver | Cells | ≤0.1 % | ≤0.5 % | Max |
|---|---|---|---|---|---|
| 3x3 | PATH (MCP) | 776 | 100 % | 100 % | 0.099 % |
| 3x4 | PATH (MCP) | 1,058 | 100 % | 100 % | 0.099 % |
| 5x5 | PATH (MCP) | 2,472 | 100 % | 100 % | 0.097 % |
| 10x7 | PATH (MCP) | 9,130 | 97.85 % | 100 % | 0.35 % |
| 15x10 | PATH (MCP) | 24,842 | 97.96 % | 100 % | 0.42 % |
| 20x41 | Newton-TR (MUMPS) | 210,906 | 99.28 % | 100 % | 0.40 % |

All 249,184 cells fall within 0.5 %.

**Two non-negotiable conditions**, which took months to find:

1. **`base_calibrated=True`.** Without it the match drops to ~56 %. The residual that for a long
   time was blamed on linearization was the *seed*.
2. **The closure must match.** `capFlex` against a `capFix` fixture gives 70-80 % false
   mismatch. `capFlex` ↔ default and `capFix` ↔ swap `dpsave`/`del_tbalry` are not interchangeable.

The residual left in `qxs` (bilateral trade) **is not a bug**: equilibria agrees with
GEMPACK (0.076 pp) better than GAMS does in levels (0.97 pp). See
`docs/archive/findings/` for the term-by-term decomposition.

### Validation against GAMS

The 5 gates of `run_parity_gates.py` cover both paths: the monolith is the oracle against GAMS
(mcp, nlp, nl) and the blocks against GEMPACK. See `docs/architecture/monolito_vs_bloques.md`.

MCP + NLP matrix (18 + 14 rows, 5 datasets, CES and Cobb–Douglas `altertax`, ifSUB 0 and 1): every
cell of the three periods within 1 % of GAMS, `code=1` in all 96 stages (measured
2026-10-27, after #174; before, the `shock` of the `altertax` rows gave 99.08–99.91 % because
its tariff did not enter the solve).

### Burfisher book exercises (3rd ed.)

`scripts/gtap/run_burfisher.py`, dataset `nus333`:

- Against GAMS: 35 of 41 exercises with every cell within 0.1 %.
- Against GEMPACK (Gragg): the 34 that both programs solve, every cell within 0.5 %.
- ME4A/ME4B have no nonlinear solution in either program (GAMS infeasible, GEMPACK aborts).

### Infrastructure

- **Native readers** for HAR (GEMPACK) and GDX (GAMS), with writing and a semantic round-trip.
- **Free solver**: Newton-TR + MUMPS solves the 20x41 (3.43 M variables) without CONOPT or IPOPT.
- **MIP → SAM pipeline** with structural closure (see README).
- **Parity cascade**: eight Python↔GAMS diagnostic tools.

---

## Architecture plan

Decisions taken with the author (October 2026) and their status. Unlike the rest of the
document, this section carries no figures: it says what will be done and in what order.

### Done or in progress

| What | Status |
|---|---|
| One cache key for the model and the seed | Done (#101) |
| Prepare each period with a recipe (`PeriodPreparer`) | Done (#103, #108) |
| One name, one declaration: `add_block` rejects a variable, parameter or equation declared differently by two blocks, and each block lists in `Block.uses` what it reads from others | Done (#139) |
| One way to apply shocks: the shock lives in the `ShockBlock` and is written with `apply_shock` | Done (#113, #174). The default +10 % tariff is also written in the `ShockBlock` (`shocks.apply_default_tariff`) and ~900 lines of driver reconstruction left; `solve_multiperiod` rejects a model without a `ShockBlock` |
| The seed and model cache key includes Python and the numeric stack; `uv.lock` pins the versions the paper was measured with | Done (#172) |
| Written rules: ADR 0001-0008, `docs/glossary.md` and per-area specifications in `specs/` (proposal → sync → archive), checked in CI by `spec-check` | Done (#173) |
| Take the monolith out of the form test | Pending (`R-MONOLITH`). Includes generating from `Block.uses` the list of borrowed variables of `test_gtap_blocks_form.py` |
| Read GEMPACK on any operating system: the `.sl4` solution without `sltoht` | Done (#148): `read_sl4_levels`, and `equilibria compare` accepts `.sl4`. A ViewHAR-like viewer is missing (`R-HAR-VIEW`) |

### Next, in this order

1. **Periods as a list, with a transition recipe between one and the next** (`R-DYN`). Today the periods
   are fixed (`PERIODS = (base, check, shock)`) and the driver assumes that sequence. The blocks,
   on the other hand, no longer know about periods: they write their equations for a single one and the
   time index is added by the SP copy. With the list, a **recursive dynamic model** means
   solving one year at a time and, between years, updating the stocks (capital, population,
   productivity) with a dynamics block.
2. **Each block calibrates its own part from a common data interface** (SAM or HAR; `R-CALIB`). Today GTAP
   calibrates in a central file (`blocks/gtap/_derived_params.py`) from the GTAP base,
   and PEP calibrates from the SAM with its own code. `Block` already carries `CalibrationMixin`, but only
   the generic blocks use it.
3. **An environmental block (IEEM)** (`R-ENV`) as the first test that adding a module is easy:
   emissions and natural capital as a block of its own that declares what it uses from production.
   `sam_tools` already reads the IEEM SAMs, but only their economic part.

---

## Technical debt

What follows is measured and pending. No entry is speculative.

### 20x41 performance — `PERF-EVAL-1` (`R-PERF`)

Measured with `scripts/gtap/phase_times.py` (3 periods, 3 solves, `code=1`, iterations 3/3/22):

| Machine | Total |
|---|---|
| macOS, M3 Pro | 7.2 min |
| GitHub Actions, Linux (AMD EPYC 7763) | 14.5 min |
| GitHub Actions, Windows (same EPYC) | 23.7 min |
| Windows laptop, i7-1365U | 34.9 min |
| GEMPACK, same laptop | 4.3 min |

- **Solving is 11 %** (Newton-TR + MUMPS). The other 89 % is building and preparing the Pyomo
  model before each solve.
- **Windows is 1.64× slower than Linux on the same hardware**, evenly across all
  phases: it is not one phase, it is the Python work.
- **Earlier attempts.** The POI backend (`archive/perf/poi-fase0`) did not scale: IR
  generation does not scale. The JAX translator (`archive/feat/pyomo-jax-translator`) is correct (96/96
  equation families match Pyomo to <1e-7, identical Jacobian) but **was never measured
  active on the 20x41**, and compiling already costs ~8.4 min on the 3x3. It is not ruled out: it still has to be measured.

Next step: an **alternative evaluation backend** (`PERF-EVAL-1`), with Pyomo as the parity
oracle. Before going further it has to pass a prototype with two gates: exact equality with
Pyomo, and scale measured on 3x3, 10x7 and 15x10. Status: **design, no code**.

### Validation against the literature — pending (`R-BOOK`)

The current gates measure equilibria against GAMS and against GEMPACK: two
reference implementations of the same model. The third check is missing, the
one that depends on neither: **reproducing the published exercises of the
standard textbook** (Burfisher, *Introduction to Computable General
Equilibrium Models*, 2nd ed.), each one as a test over the matrix.

Criterion: each exercise ≥99 % with `code=1`, or `N/A` with the reason written down.

Status: all 41 exercises are already compared against GAMS and GEMPACK (see "What is closed").
What this section asks for is missing: comparing against the **published figures of the book**, not
against another implementation.

The `nus333` debt for gtap7 (`gdpmp[USA]` −97.93 % against +4.58 %) is resolved:
`tests/templates/gtap/test_nus333_neos_parity.py` passes on `main` (verified 2026-10-27).

### Known blockers

| Debt | Detail | First step |
|---|---|---|
| `R-NL-GATE`: the `.nl` gate passes vacuously ([#23](https://github.com/mmc00/equilibria/issues/23)) | Pairing is by name and the GAMS fixtures do not carry their `.row`/`.col` sidecars, so their rows parse anonymous. **Measured 2026-09-24: the 5 datasets of the gate compare 0 constraints**, not only `gtap7_3x3` as the issue said. No `.row`/`.col` ever existed in the repo | Regenerate the fixtures with sidecars: the `.gms` must be in NLP form (CONVERT does not write `.nl` from an MCP) and the option is `AmplNL <file>`. Tried locally with GAMS v53, without NEOS |
| `R-MONOLITH`: `gtap_model_equations.py` cannot be retired | Since #174 `solve_multiperiod` rejects a model without a `ShockBlock`, so seven scripts of `scripts/gtap/` that solved it were left unused until migrated: `measure_gate_tols`, `measure_gtap_pure_tols`, `measure_nlp_vs_nlp`, `measure_nlp_vs_nlp_altertax`, `diag_mp_3x3`, `bench_nlp_timing`, `gen_linearization_study`. Beyond that it can only not be retired because of its role as the **GAMS oracle in the `nl` gate** — measured 2026-09-24: the other four (`mcp`, `nlp`, `gempack`, `coverage_matrix`) do not read it, and `docs/architecture/monolito_vs_bloques.md:95` already said so. It is NO longer a runtime dependency of the blocks path: scaling was extracted to `gtap_benchmark_scaling.py` and the multiperiod reflection asks for its SP per method, so `gtap_block_model.py` does not import it (2026-09-16). Since 2026-09-24 it also has to be **asked for**: a model without `_sp_source` goes to blocks, and an unrecognized value raises | Migrate the `nl` gate to measure blocks, checking that the numbers do not move |
| `R-BOUND-WARN`: `code=1` with a price at its floor | A solve can end with `code=1` and a price resting on its lower bound of 1e-3 (e.g. `pft[USA,LAND]` in ME4A; also seen in 15x10 ifSUB=1). It is a complementarity point at the bound, not an economic equilibrium, and today it is reported as converged | Have the driver or the solver warn (or return another code) when prices sit at their bound at the end |
| `R-PMUV`: `eq_pmuv` not ported to blocks | Declared in `blocks/gtap/__init__.py:47-53`, not implemented in the composer. It only bites with non-empty `rmuv`/`imuv`, which no gate dataset uses | Flip `pmuv` from Param to Var when the closure asks for it |

### Tests that were broken on main

| Test | Symptom |
|---|---|
| ~~`test_gtap_blocks_form[ClosureBlock]`~~ | **Resolved 2026-09-16.** It was not a test defect: the split of the Fisher aggregates condensed `eq_pfact`/`eq_pwfact` to 4 variables, so their skeleton cannot match the monolith's. `_SPLIT_AUX_EQS` exempts them from the form comparison —with an exemption count— and equivalence is measured on the values |
| ~~`test_multiperiod_driver::test_solve_multiperiod_solves_m_not_slices`~~ | **Passes on `main`** (verified 2026-10-27, `d356595`) |
| ~~`test_cascade_layers`, `test_cascade_run`, `test_probe` (×2)~~ | **Pass on `main`** (verified 2026-10-27, `d356595`; 4 cases of `test_probe` skip without the reference GDX) |

The 2 of `test_writer` became `xfail(strict=True)`: the GDX writer declares
support for `Set` and `Parameter` only, so demanding `Variable`/`Equation` from it asks for
unimplemented functionality. When it is implemented, the test will flag that the mark is no longer needed.

### Fisher snapshot: fixed, and the POI side is still archived

`blocks/gtap/__init__.py` item 3 requires the composer to overwrite `pf0`/`xf0`/`mqfactr_bb`/
`mqfactw_bb` with the post-scaling snapshot. Nobody did it: `xf0` was off by
exactly `xscale` (10× in 26 of 45 cells of `gtap7_3x3`). In the benchmark the bias
cancels, but outside it `eq_pfact` diverged from the monolith by up to **1.15 %**.

Fixed on the Pyomo path (`apply_fisher_snapshot_overwrite`), ported from
`archive/poi/fase0-fixes`. Measured: divergence 1.15 % → **exactly 0**; gates green.

**The POI side is still only in the tag.** If that line is resumed, it has to be ported along: with only one
side fixed, a POI-vs-Pyomo comparison goes from having a symmetric bias —which cancelled—
to a real asymmetry.

### Solvers: what it takes to run the whole suite

`uv sync` does not install the solvers. The tests that solve models skip without them,
and declare it with one marker per piece (see `tests/conftest.py`):

| Marker | Needs |
|---|---|
| `needs_mumps` | `pymumps` (`conda install -c conda-forge pymumps`) |
| `needs_ipopt` | the `ipopt` executable, or `cyipopt` |
| `needs_path` | the PATH C-API library + the path-capi-python checkout |

`EQUILIBRIA_REQUIRE_SOLVERS=1` turns the guards off, so that locally a missing solver
shows up instead of hiding behind a skip.

### GTAP tests do not tolerate parallelism

They compete for the PATH library and for the model caches. Two simultaneous runs produce
dozens of spurious failures: one measurement gave 40 failures that, in isolation, came down to 1 real one.
**Do not use `pytest -n` nor launch two suites at once.**

### Latent code or code without consumers

| Item | Status |
|---|---|
| `blocks/{production,trade,demand,institutions,equilibrium}/` | 4.1 k lines; consumers only in `simple_open` and examples. Decide whether they are public API |
| `pep_cri_*.py` | Latent: 0 tests, last substantive edit 2026-04-22 |
| `pep_levels`, `pep_parity_pipeline`, `pep_scenario_parity` | Zombies: alive only through their own tests |
| `templates/gtap_loglevels/` | Marginal: 0 importers in `src`, 1 test |
| 44 symbols exported via `__init__` | No real consumer (speculative API) |

---

## Archived work

Ten branches with never-published work are kept as `archive/*` tags. To recover one:
`git checkout archive/<name>`.

| Tag | Content |
|---|---|
| `archive/perf/poi-fase0`, `archive/poi/fase0-fixes` | POI backend (PyOptInterface) with IR and LLVM. The most recent work in the repo: Phase 0 concludes the premise holds but IR generation does not scale |
| `archive/gtap6-template-impl` | GTAP 6 template: closes a 269-DOF gap with 7 price identities |
| `archive/feat/pyomo-jax-translator` | Pyomo → JAX translator |
| `archive/gtap/v62-rollback`, `archive/worktree-runMacos` | GTAP v7 datasets and reproduction on macOS |
| Other 5 | Specs, spikes and performance experiments |

---

## Conventions not worth relearning the hard way

- **Fidelity beats match percentage.** Never revert a faithful change because a figure
  drops; the only reason to remove it is that it is unfaithful.
- **GAMS and GEMPACK are the source of truth.** Do not exclude divergent cells to inflate the match.
- **Before blaming the model**, check the closure, the seed and the fixture. Most historical
  "bugs" were crossed closures or uncalibrated seeds.
- **External references resolve through environment variables** (`EQUILIBRIA_REFS_DIR`,
  `EQUILIBRIA_NUS333_DIR`, `EQUILIBRIA_PATH_CAPI_SRC`, `EQUILIBRIA_PATH_CAPI_LIB_DIR`,
  `EQUILIBRIA_CGE_BABEL_DIR`). See `src/equilibria/_local_refs.py`.
- **The parity gates are mandatory** before a push that touches GTAP; the hook
  `scripts/gtap/claude_hooks/block_push_without_gates.py` enforces it.
