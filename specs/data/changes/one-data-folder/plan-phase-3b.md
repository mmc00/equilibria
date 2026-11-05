# One data folder — Phase 3b Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every consumer of `tests/fixtures/gtap7_gempack/`, `tests/fixtures/nus333_bienestar_gempack.json.gz` and `runs/` reads the records in `data/` instead; then both folders are deleted.

**Architecture:** Consumers locate a result with `equilibria.sources.reference(dataset, tool, **variant) / "<file>"`, never a hard-coded `data/` path. Work goes one family of old files at a time: compare old vs new, shrink the allowance in `tests/data/test_no_test_fixtures.py` so the guard fails, rewire the consumers, `git rm` the old files, and the guard passes again.

**Tech Stack:** Python ≥3.10, pytest, `uv`, `equilibria.sources`, `equilibria.babel.har.read_har`, `scripts/dev/write_source.py`.

**Spec:** `specs/data/changes/one-data-folder/design.md`, `docs/adr/0010-one-data-folder.md`, and the 3b lists in `plan-phase-3a2.md` ("Replaces" per task, "Not regenerated on Windows", "Expected differences against the old files").

## Global Constraints

- No number and no test assertion changes (ADR-0008). A consumer changes where it reads, not what it checks. An old-vs-new mismatch is a finding: stop, write it in `diff_3b.md`, and ask before touching an assertion.
- Records are found with `reference()`. A test does not skip because a record is missing: `data/` is in the repository.
- Nothing under `data/gtap7/gempack/` or `data/gtap7/gams/` is edited: those files are hashed by the recorded results. Their stale comments stay (as in phase 6).
- Suite before and after: identical test by test and skip by skip (names and skip reasons), except the changes this plan lists. Baseline on `main` first (Task 0).
- Names, code, comments and commits in English; lowercase file names; `type(scope): description` with the session's `Co-Authored-By` line.
- Never two pytest sessions at once (the full suite takes ~45 min on the Mac). Before tests: `export PATH=/Library/Frameworks/GAMS.framework/Versions/53/Resources:$PATH`.

### Decisions

- **D1 (proposed 2026-11-05, to confirm).** ME4A and ME4B have no Gragg oracle. Their old `burfisher_gragg/me4a.*`, `me4b.*` were a failed run ("Accuracy too low") and are deleted with nothing in their place. The book's Johansen solutions (`gempack_burfisher/me4a`, `me4b`) are their oracle, already used by R-JOHANSEN (#203, #205). A consumer that iterated over the Gragg set drops these two with a comment naming D1; no other test changes.
- **D2.** Old files that were not GEMPACK runs move to `data/` with a `source.yaml` written by `write_source.py --no-run`, next to what they describe: `sl4levels_sample_sic.txt` (hand-written reader oracle), `updated_synthetic.har` + `make_synthetic_updated_har.py` (its generator), `table1_levels_v061_gtap7_3x3.json.gz` (equilibria v0.6.1), and `nus333_bienestar_gempack.json.gz` (computed from the book solutions; regenerated from the new records, Task 6).
- **D3.** `runs/` (Task 8): its copies of Horridge's files and its `.cmf` templates already live in `data/`; its analysis scripts move to `scripts/gtap/rungtap_compare/` and read `data/`; its written analyses (`comparison*.md`, `comparison.json`, `trajectory.json`, `RESULTADO.md`, the validation `README.md`) and `equilibria/*.json` go to `docs/archive/findings/rungtap_compare/` as they are; `gempack_updated_har_guide.md` goes to `docs/notes/guides/gempack_updated_har.md`. Scripts that wrote under `runs/` write to `notes/tmp/` (as phase 6 did for `reports/`).

### Old → new

| old (`tests/fixtures/gtap7_gempack/`) | new record | `reference(...)` |
|---|---|---|
| `{decomp,sl4dump,updated}_<ds>_<tag>.har` (66, 3a-1) | `gtap7/<ds>/gempack_<tag>/` | `(ds, "gempack", shock=, steps=, closure=)` |
| `{sl4dump,updated}_gtap7_*_tm10.har` (5+5) | copies of the `_tm10_s8-16-32` capFlex ones | same, capFlex |
| `{sl4dump,updated}_{nus333,9x10}_tm10.har` | `gtap7/<ds>/gempack_tm10_s8-16-32_dpsaveswap/` | `closure="dpsaveSwap"` |
| `sl4levels_<ds>_tm10*.har`, `sl4_gtap7_20x41_…_capfix.sl4` | `gtap7/<ds>/sltoht_<tag>/` (`sl4levels.har`, `solution.sl4`) | `(ds, "sltoht", …)` |
| `sl4levels6_<ds>_*.har`, `default_gtap7_20x41_rdlt0_windows.prm` | `gtap7/<ds>/gemsim_levels6_<tag>/` (`sl4levels.har`, `default_rdlt0.prm`) | `(ds, "gemsim", tab="gtapv7_levels", …)` |
| `burfisher/<exp>.{sl4.gz,upd,log.gz}` | `gtap7/nus333/gempack_burfisher/<exp>/` (`solution.sl4`, `updated.har`, `solve_console.txt`, `run.cmf`) | `("nus333", "gempack", experiment=, method="book")` |
| `burfisher/sl4dump_<exp>.har` (3) | `gtap7/nus333/sltoht_burfisher/<exp>/sl4dump.har` | `("nus333", "sltoht", experiment=, method="book")` |
| `burfisher_gragg/<exp>.{sl4.gz,upd,log.gz}` (41) | `gtap7/nus333/gemsim_burfisher_gragg/<exp>/` | `("nus333", "gemsim", experiment=, method="gragg")` |
| `burfisher_gragg/sl4levels_<exp>.har` (41) | `gtap7/nus333/sltoht_burfisher_gragg/<exp>/sl4levels.har` | `("nus333", "sltoht", experiment=, method="gragg")` |

Old `.sl4.gz`/`.log.gz` were gzipped; the new `solution.sl4`/`solve_console.txt` are not. A reader that opened them with `gzip` reads plain files.

## Review Focus

1. **Layout of the new `sl4levels.har`.** The exporter writes LEVA+LEVB+NOLV; the old files were LEVA-only with a 1e10 sentinel. `_oracle.py` EX16 and `test_export_sl4_levels.py` read that layout → Task 3 Step 1 compares the values cell by cell and Task 3 adapts the reader, not the assertion.
2. **`sl4levels6_gtap7_3x3_tm10.har` vs `…_tm10_s8-16-32.har`** were not byte-identical; both map to one folder → Task 3 states in `diff_3b.md` which one it matches.
3. **A consumer silently reading the wrong variant.** `reference()` with too few keys raises (more than one match), but a wrong value can still match another folder → each rewired test keeps its own assertions, and Task 2 adds `test_reference_keys_are_complete` (every `reference()` call in tests/scripts names `shock`, `steps` and `closure` for matrix records).
4. **gzip readers** given a plain file fail or, worse, read nothing → the family test runs before the old files are deleted, and must pass after.
5. **The suite losing tests to new skips** (memory: "0 changed by name" hid 18 new skips) → Task 9 compares names and skip reasons against Task 0's baseline.

---

### Task 0: Baseline

- [ ] **Step 1:** On `origin/main`, run the full suite once with `--junitxml` and keep it in the scratchpad (`suite_main_<sha>.xml`). Same command as phase 6, comparison script `suite_diff.py`.

### Task 1: The guard knows the families

**Files:** Modify `tests/data/test_no_test_fixtures.py`, `scripts/dev/check_layout.py` (unchanged until Task 9).

- [ ] **Step 1:** Replace `ALLOWED_UNTIL_PHASE_3B_PREFIX`/`ALLOWED_UNTIL_PHASE_3B` with `PENDING_3B: dict[str, str]`, one glob per family → the task that removes it (`"tests/fixtures/gtap7_gempack/decomp_*": "Task 2"`, …, `"runs/**": "Task 8"`). `test_no_files_under_tests_fixtures` allows only files matching a pending glob; add `test_pending_globs_match_something` (an empty glob is stale) and `test_no_code_reads_runs` (same scan as `test_no_code_points_into_tests_fixtures`, pattern `\bruns/|"runs" /`, exempt while `runs/**` is pending).
- [ ] **Step 2:** Run `uv run pytest tests/data/test_no_test_fixtures.py -q` → PASS (nothing removed yet). Commit `test(data): the 3b guard names each pending family`.

### Tasks 2–7: one family each

Every family task has the same five steps; the table says what changes.

- [ ] **Step 1: compare.** For each old file of the family, compare it with its new file (`read_har`, numeric, `XX*` headers skipped, as in the 3a-2 review). Write one row per group to `specs/data/changes/one-data-folder/diff_3b.md` (max abs diff, files compared). Any value above 1e-9 is a finding: stop and ask.
- [ ] **Step 2: guard red.** Remove the family's glob from `PENDING_3B` → `test_no_files_under_tests_fixtures` and `test_no_code_points_into_tests_fixtures` FAIL, naming the files and the consumers.
- [ ] **Step 3: rewire.** Each consumer in the table reads through `reference()`. Docstrings that name the old path name the record folder instead.
- [ ] **Step 4: remove and pass.** `git rm` the family's old files. Run the guard and the family's tests (table) → PASS, with the same counts of pass and skip as before Step 2.
- [ ] **Step 5: commit** `refactor(data): <family> read from data/ (phase 3b)`.

| Task | Family (old globs) | Consumers | Family tests |
|---|---|---|---|
| 2 | `decomp_*`, `sl4dump_<ds>_*`, `updated_<ds>_*` (matrix, `_tm10` copies, nus333/9x10 swap) | `tests/templates/gtap/test_gtap7_gempack_parity.py`, `test_f3_5_base_calib.py`, `test_gempack_har_reference.py`, `test_default_tariff_shock.py` (`updated`), `tests/compare/_oracle.py` (`_GEMPACK_FIXTURES`), `tests/compare/oracles/gtap7_map_3x3.py`, `tests/compare/test_gtap7_map.py`, `scripts/gtap/gen_linearization_study.py`, `scripts/paper/run_tables.py`, `scripts/gtap/compare_gempack_levels.py` (docstring) | those files + `tests/compare`; add `test_reference_keys_are_complete` (Review Focus 3) |
| 3 | `sl4levels_*`, `sl4levels6_*`, `sl4_*.sl4`, `default_gtap7_20x41_rdlt0_windows.prm` | `tests/compare/_oracle.py` (EX16, lines ~562/588), `tests/compare/oracles/gempack3x3_levels.py`, `tests/compare/test_levels_six_vars.py`, `tests/scripts/test_export_sl4_levels.py`, `tests/scripts/test_force_rdlt.py`, `tests/templates/gtap/test_default_tariff_shock.py`, `scripts/gtap/compare_gempack_levels.py` | those + `tests/compare`; Review Focus 1 and 2 |
| 4 | `burfisher/*` | `scripts/gtap/compare_burfisher_gempack.py`, `tests/compare/oracles/burfisher_levels.py`, `tests/scripts/test_mk_exp_directos.py` (reads the command file from the old `.log.gz`; the record has `run.cmf`), `tests/templates/gtap/_nus333.py` / `run_burfisher` if they read the book solutions, the `test_nus333_*` docstrings, `examples/notebooks/gtap7/burfisher/make_notebooks.py` (`GEMPACK_BOOK_DIR`, then regenerate the notebooks with it) | those + `tests/templates/gtap/test_nus333_*` + `tests/docs/test_doc_examples.py` |
| 5 | `burfisher_gragg/*` | `tests/compare/test_burfisher_gempack.py` (`GRAGG`; D1 for me4a/me4b), `make_notebooks.py` (`GEMPACK_DIR`, regenerate) | those + `tests/compare` |
| 6 | `tests/fixtures/nus333_bienestar_gempack.json.gz` | `scripts/gtap/welfare_gempack_fixture.py` reads the book records and writes `data/gtap7/nus333/gempack_burfisher_welfare/welfare.json.gz` through `write_source.py` (`--input` each book folder it reads); `tests/templates/gtap/test_nus333_ev_regional.py`, `test_nus333_psave_gtap.py`; `scripts/dev/check_layout.py:136` | Step 1 here: the regenerated JSON equals the old one after `json.load` (exact) |
| 7 | `sl4levels_sample_sic.txt`, `updated_synthetic.har`, `make_synthetic_updated_har.py`, `table1_levels_v061_gtap7_3x3.json.gz`, `burfisher*/readme.md` | their readers (`git grep` each name); D2 places; the two `readme.md` become the `README.md` of the record parents | readers' tests |

### Task 8: `runs/`

**Files:** `git mv` per D3; Modify `scripts/gtap/compare_nus333_rungtap.py`, `scripts/gtap/compare_9x10_rungtap.py` (`OUT_DIR` → `notes/tmp/`), `docs/site/guide/welfare_decomposition.md` (lines ~146–200), `tests/scripts/test_swap_job.py` (`TRACKED` reads git bytes of `runs/`: after the delete it skips as designed — keep the skip reason, it is in the expected diff), `tests/scripts/test_regenerate_paper.py` (`runs/gempack_work` default), `data/horridge_simple/README.md` (from `runs/horridge_simple/readme_equilibria.md`, not hashed).

- [ ] **Step 1:** Remove `"runs/**"` from `PENDING_3B` → `test_no_code_reads_runs` FAIL.
- [ ] **Step 2:** Move and rewire per D3; `compare_gams_vs_gempack.py` moves to `scripts/gtap/rungtap_compare/` and takes record folders via `reference("horridge_simple", ...)` defaults. `regenerate_paper.py`'s `runs/gempack_work` default is in a hashed tool: it stays, and `.gitignore` keeps `runs/` ignored.
- [ ] **Step 3:** `git rm -r runs/`; guard, `tests/scripts`, `tests/docs` (link test) → PASS; Sphinx `-W` builds. Commit `refactor(data): runs/ leaves the repository (phase 3b)`.

### Task 9: Close

**Files:** `tests/data/test_no_test_fixtures.py`, `scripts/dev/check_layout.py`, `specs/data/specs/specs.md` if it lists the pending folders.

- [ ] **Step 1:** `PENDING_3B` is empty: replace it with `test_tests_fixtures_is_gone` (`tests/fixtures/` has no tracked file) and drop `ROOT_DIRS_TEMPORARY`'s `"runs"` and the `runs/…` allowances in `check_layout.py` → layout check passes.
- [ ] **Step 2:** Full suite, compare with Task 0 by name and skip reason. Expected differences only: tests added here; `test_templates_are_the_tracked_cmf` ×3 now skip ("runs/ already removed (phase 3b)"); D1 removes me4a/me4b from any Gragg parametrization (list them). Paste the comparison in the PR.
- [ ] **Step 3:** Commit `test(data): tests/fixtures/ and runs/ are gone (phase 3b)`; PR with `diff_3b.md` and the suite comparison.
