# One data folder — Phase 3a-2 Implementation Plan (the `corridas_paper.ps1` runs and `runs/` on Windows)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every GEMPACK result that `scripts/windows/corridas_paper.ps1` (steps 1–7) and the tracked `runs/` folders produce today is regenerated on the Windows machine straight into `data/`, one folder per result, each with a `source.yaml` written by `write_source.py`. After this phase, nothing under `tests/fixtures/gtap7_gempack/` or `runs/` is the only copy of a GEMPACK result. Phase 3b (Mac) can then diff, rewire and delete.

**Architecture:** New generators live in `data/gtap7/gempack/tools/` next to the two 3a-1 scripts, which they import but never edit. One new driver, `tools/regenerate_paper.py`, lists every job of this phase in groups. It runs each job through `write_source.py` with the same resume and cleanup rules as `regenerate_gempack.py`. `write_source.py` gains two flags:
- `--input NAME=PATH` chains a result to another result under `data/`, for example a levels export to the 3a-1 `sl4dump.har` it cross-checks against.
- `--external NAME=PATH` records the sha256 (and, for an executable, the release it prints) of a file that is never put in the repository: `gtapv7.exe`, `gtapuv7.exe`, the `C:\GP` programs, and the `.gss`/`.gst`/`.gsc` that `tablo` compiles.

The model source of steps 4–7 is in the repository: RunGTAP 3.75's `gtapv7.tab` and `gtapv7.sti`, unchanged, in `data/gtap7/gempack_rungtap375/` (`kind: model`). Their patched versions (`make_levels_tab.py`) are in `data/gtap7/gempack_rungtap375_levels/` (`kind: model`, `derived_from: gtap7/gempack_rungtap375`).

Nothing old is deleted or rewired here.

**Tech Stack:** Python ≥3.10, pytest, `uv run` on Windows.
- RunGTAP 3.75 (`C:\runGTAP375`): `gtapv7.exe`, `gtapuv7.exe` — GEMPACK Release 12.1.003.
- GEMPACK in `C:\GP`: `tablo` 6.4, `gemsim` 4.3, `sltoht` 5.53 — GEMPACK Release 11.3.004, read from the consoles of the 2026-10 runs.
- No GAMS on Windows: Horridge's GAMS results run on the Mac (Task 6, decision D4).

**Spec:** `specs/data/changes/one-data-folder/design.md` (layout: `nus333/gempack_burfisher/<experiment>/`, `horridge_simple/`; "`source.yaml`"; "When a regenerated file differs from the old one"); `plan-phase-3a1.md`.

## Global Constraints

- **Same rules as 3a-1.**
  - Every result is written by `scripts/dev/write_source.py`, which runs the command, times it and writes the `source.yaml`. No hash is computed by hand and no `source.yaml` is edited.
  - A failed job leaves no folder behind.
  - The driver resumes: a folder that already has a `source.yaml` is skipped.
  - Before each commit, `check_tree()` prints `[]` for everything the commit adds.
  - `data/.gitattributes` (`* -text`) keeps the bytes as written on Windows.
  - Names, code, comments, this plan and commit messages are in English.
- **The job set reproduces what is committed today, nothing more.** Every task names the old files it replaces. An old file with no job here is listed under "Not regenerated on Windows".
- **Frozen 3a-1 tools.** `data/gtap7/gempack/tools/run_gempack_matrix.py` and `regenerate_gempack.py` are listed by sha256 in all 66 matrix records. Editing either makes all 66 stale and forces the 9-hour rerun. New code imports them and does not edit them (`prepare`, `solve`, `convert_sl4`, `config_tag`, `_force_rdlt`, `collect_job`). Task 1 adds a test that pins them.
- **Generators live with the data.** These scripts make results, so they move into `data/gtap7/gempack/tools/` (`git mv`, importers updated):
  - `scripts/gtap/make_levels_tab.py`
  - `scripts/gtap/export_sl4_levels.py`

  Every record of this phase hashes the whole `data/gtap7/gempack/` folder with `--model-file`, and the model record `data/gtap7/gempack/source.yaml` is re-recorded by `write_source.py --kind model` whenever a tool file is added. **All code tasks (0–2) land before the first Windows run.** Once a group is recorded, its tools are frozen the same way.
- **Tool and version, one release per record.** `tool`/`tool_version` name the program that made the folder's primary output, with the release that program prints. The tool script reads it from the console banner and refuses a mismatch with the driver's argument, as `--expect-release` does in 3a-1.

  | tool | executable | release | driver argument |
  |---|---|---|---|
  | `gempack` | `gtapv7.exe` / `gtapuv7.exe` (RunGTAP) | 12.1.003 | `--gempack-release` |
  | `gemsim` | `C:\GP\gemsim` (+ `tablo` of the same release, compiling a TAB) | 11.3.004 | `--gp-release` |
  | `sltoht` | `C:\GP\sltoht` (+ `export_sl4_levels.py`) | 11.3.004 | `--gp-release` |
  | `gams` | GAMS 53.5 on the Mac (Task 6 only) | read from the `.lst` | — |

  Any other executable a record uses goes in `external` with its sha256 and release. For example, an `sltoht` levels record re-solves with `gtapv7.exe`, so `external` holds `gtapv7.exe: {path, sha256, release: 12.1.003}`.
- **`reference()` stays unique.** No new record shares (`dataset`, `tool`) with a 3a-1 record while carrying a superset of its variant. `reference(ds, "gempack", shock=…, steps=…, closure=…)` must still return exactly one folder for each of the 66 (Review Focus 1). That is why the levels exports are `tool: sltoht` / `tool: gemsim` and not more `gempack_*` folders.
- **The GTAP model source (maintainer, 2026-11-03).** Steps 4–7 use RunGTAP 3.75's `gtapv7.tab` 7.0, the source of the `gtapv7.exe` that made the 66 matrix records. They do not use the official 7.1: `docs/findings/gtapv7_tab_70_vs_71_2026-11-03.md`. The 7.1 left `data/` in commit 8e9420fb (PR #186) and is tracked in issue #198.
  - **`data/gtap7/gempack_rungtap375/`** (`kind: model`, recorded like `data/gtap7/gams_jgea`): `gtapv7.tab` and `gtapv7.sti` copied byte for byte from `C:\runGTAP375`.
    - Expected sha256: tab `67cb2c6793fc40b3e14d8ce997ab1d66920b03800d0d0f3897c7a1bf0d5d2836`, sti `a3279021f72413ed3519031dd5ffbe2a893f7c770c0d8267a174415d8abc7a87`.
    - `notes`: licence "not stated by the source; to confirm".
  - **`data/gtap7/gempack_rungtap375_levels/`** (`kind: model`, `derived_from: gtap7/gempack_rungtap375`, `changes` listing each edit): `gtapv7_levels.tab`/`.sti` and `gtapuv7_levels.tab`/`.sti`, written by `tools/make_levels_tab.py` from the folder above. The uncondensed `gtapuv7_levels` needs nothing beyond `gtapv7.tab` and `gtapv7.sti`: `--uncondensed` derives it from the same two files.
  - **Never in the repository:** the `.gss`/`.gst`/`.gsc` that `tablo` compiles, and any `.exe`/`.dll`. Tasks 7 and 8 compile into a scratch work folder outside `data/`. The compiled files and every executable are recorded only by sha256 in `external` (Task 0).
  - `data/gtap7/gempack_purdue/` (the 7.1) disappears from this branch when `main` is merged in after #186 lands. Nothing here reads it.
- **Variant keys.** Each task lists them. Shared spellings:
  - `shock`: number, percent
  - `steps`: string, `"8-16-32"`
  - `closure`: `capFlex` | `capFix` | `dpsaveSwap`. `dpsaveSwap` is the retired `swap dpsave(r) = del_tbalry(r)` of the May/July `.cmf`, which is not equilibria's capFix (`docs/findings/gempack_fixture_closure_mislabeled_2026-08-02.md`).
  - `experiment`: the lower-case `.exp` stem
  - `method`: `book` (as the `.exp` says) | `gragg` (Johansen 1-step rewritten to Gragg 2-4-6)
- **Solutions are stored raw** (D3). `.sl4` files are kept uncompressed, as GEMPACK wrote them: about 42.5 MB for both Burfisher sets. `data/` is already excluded from `check-added-large-files`.
- **No `levels.txt`** (maintainer, 2026-11-04). The `sltoht -SHL -SIC` text that `export_sl4_levels.py` parses is not kept: it is 60.8 MB for each 20x41 run, about 270 MB of the first full run's 338 MB, and `sltoht` rebuilds it from `solution.sl4`. `sl4levels.har` is the record.
- **One batch of tools changes, then one full run** (maintainer, 2026-11-04). Every record lists the sha256 of every file of `data/gtap7/gempack/tools/`, so a change there makes every recorded result stale. All `tools/` changes go in one batch, reviewed before the run, and then everything runs once.

## Review Focus

1. 3a-1 references stay unique after every 3a-2 record exists → Task 2 `test_matrix_references_stay_unique` (all 66 `reference(...)` calls, on a tmp copy of `data/` with one fake record per new folder kind).
2. The two 3a-1 tools are byte-frozen → Task 1 `test_matrix_tools_are_frozen` (their sha256 equal what every `gempack_<tag>/source.yaml` lists).
3. A tool whose banner release differs from the driver's argument exits 2 before solving → Task 1 `test_release_mismatch_refuses`.
4. No compiled GEMPACK file and no executable is ever written under `data/` (`.gss`, `.gst`, `.gsc`, `.exe`, `.dll`). `.tab`/`.sti` are allowed only in `gempack_rungtap375/` and `gempack_rungtap375_levels/` → Task 2 `test_no_compiled_gempack_files_in_data` and Task 7 `test_compiled_tab_never_in_data`.
5. `gemsim` writes a `.sl4` even when it fails (ME4A/B: "Accuracy too low"). OK requires rc 0 and "completed without error" in the log → Task 7 `test_gemsim_needs_normal_end`.
6. The Johansen → Gragg rewrite leaves exactly one `Method` line and one `Steps` line → Task 8 `test_johansen_becomes_gragg_once`.
7. The 20x41 `dpsaveSwap` job is a recorded *failure*. Its tool exits 0 only when GEMPACK aborts with the known signature (assert-arith dump written, "LOGE of negative" in `E_u`). Any other outcome exits 1 and records nothing → Task 11 `test_expected_abort_only_on_signature`.
8. A failed nested job (`nus333/gempack_burfisher/<exp>/`) is removed before the next job hashes `data/gtap7/nus333`; otherwise the dataset record breaks → Task 2 `test_failed_nested_job_is_removed`.
9. `--external` never reads outside the paths it is given, and `check_tree()` does not try to verify externals off Windows → Task 0 `test_external_is_recorded_not_checked`.
10. `gempack_rungtap375/` holds RunGTAP's two files unchanged (the expected sha256 above), and the patched TABs are exactly what `make_levels_tab.py` writes from them, with every edit listed in `changes` → Task 1 `test_rungtap375_is_unchanged`, `test_levels_tabs_are_derived`.

---

### Task 0: `write_source.py` records chained inputs and external files

**Files:**
- Modify: `scripts/dev/write_source.py` (`_inputs`, new `_externals`), `src/equilibria/sources.py` (`_ALLOWED` gains `external`; shape check `{name: {path, sha256, release?}}`)
- Test: `tests/scripts/test_write_source.py`, `tests/data/test_sources_schema.py`

**Interfaces:**
- `--input NAME=PATH` (repeatable): PATH under the data root, a file or a folder. Written as `generated_from.NAME = {path, sha256}`, the same shape as `model`/`data`. Checked by `check_tree()` like every input.
- `--external NAME=PATH[@RELEASE]` (repeatable): `external.NAME = {path: "<as given>", sha256, release}`. Not checked by `check_tree()`.

- [ ] **Step 1: failing tests:**
  - `test_input_chains_a_result`: `--input sl4dump=gtap7/gtap7_3x3/gempack_tm10_s8-16-32_capflex/sl4dump.har`; then edit that file in the tmp tree, and `check_tree` names it.
  - `test_input_outside_root_refused`
  - `test_external_is_recorded_not_checked`: the record holds the sha256 of a tmp file; delete the file, and `check_tree` is still `[]`.
  - `test_schema_rejects_bad_external`
- [ ] **Step 2:** `uv run pytest tests/scripts/test_write_source.py tests/data/test_sources_schema.py -q` → FAIL. **Step 3:** implement. **Step 4:** → PASS.
- [ ] **Step 5:** Commit `feat(dev): write_source records chained inputs and external files`.

### Task 1: the tools folder, the release check and the GTAP model source

**Files:**
- Move: `scripts/gtap/make_levels_tab.py`, `scripts/gtap/export_sl4_levels.py` → `data/gtap7/gempack/tools/`. Update the importers that `git grep -lE 'make_levels_tab|export_sl4_levels'` lists among code (`scripts/gtap/gempack_reference.py`, `scripts/gtap/compare_burfisher_gempack.py`, `src/equilibria/compare/sources.py` if it imports, `tests/scripts/test_export_sl4_levels.py`, `tests/scripts/test_make_levels_tab.py`, `tests/compare/test_levels_six_vars.py`, `tests/compare/oracles/levb_six_vars.py`), plus `scripts/readme.md`. Only import paths change.
- Modify: `tools/make_levels_tab.py` gains `--changes-out FILE`. It writes the `changes` YAML (`{file, change, what, why, commit}` per edit) that `write_source.py --kind model --changes` records.
- Create: `data/gtap7/gempack/tools/release_check.py` (not `gempack_release.py`: `.gitignore` ignores `gempack_*.py`), with `release(console: Path) -> str | None` (the 3a-1 `run_gempack_matrix.gempack_release`, reused) and `require(console, expected) -> str` (raises `ReleaseMismatch`).
- Create (data, recorded by `write_source.py --kind model --no-run`):
  - `data/gtap7/gempack_rungtap375/` with `gtapv7.tab` and `gtapv7.sti`, copied byte for byte from `C:\runGTAP375`. `--source` "RunGTAP 3.75 (C:\runGTAP375), the GTAPv7.tab 7.0 its gtapv7.exe was compiled from; unchanged (CRLF kept)"; `--notes` "licence: not stated by the source; to confirm".
  - `data/gtap7/gempack_rungtap375_levels/`: `make_levels_tab.py` on that folder, run once condensed and once `--uncondensed`. `--derived-from gtap7/gempack_rungtap375`, `--changes` from `--changes-out`.
- Test: `tests/data/test_gempack_tools.py` (extend), `tests/data/test_release_check.py`, `tests/data/test_gempack_rungtap375.py`

- [ ] **Step 1: failing tests:**
  - `test_matrix_tools_are_frozen` (Review Focus 2)
  - `test_levels_tools_live_with_their_model` (the two files exist under `tools/` and not under `scripts/gtap/`)
  - `test_release_parses_both_banners` (fixtures: the first lines of a 3a-1 `solve_console.txt` and `sltoht_console.txt`)
  - `test_release_mismatch_refuses`
  - `test_rungtap375_is_unchanged` (Review Focus 10: the two expected sha256, and `kind: model`)
  - `test_levels_tabs_are_derived` (a tmp rerun of `make_levels_tab.py` on `gempack_rungtap375/` reproduces the four files byte for byte; `changes` names every edited file)
- [ ] **Step 2:** run → FAIL. **Step 3:**
  - `git mv`, fix imports, write `release_check.py` and `--changes-out`.
  - Copy the two RunGTAP files, check their sha256 against the expected ones, and stop if they differ.
  - Record both model folders, then re-record `data/gtap7/gempack/source.yaml` with `write_source.py --kind model`.
- [ ] **Step 4:** `uv run pytest tests/data tests/scripts tests/compare -q` → PASS, and `check_tree()` prints `[]`.
- [ ] **Step 5:** Commit `refactor(gempack): the levels generators live in data/gtap7/gempack/tools`, then `data(gempack): RunGTAP 3.75's gtapv7.tab 7.0 and its patched levels TABs`.

### Task 2: the driver `tools/regenerate_paper.py`

**Files:**
- Create: `data/gtap7/gempack/tools/regenerate_paper.py`; one job module per group, imported by it: `tools/burfisher_job.py`, `tools/levels_job.py`, `tools/gemsim_job.py`, `tools/swap_job.py`
- Test: `tests/scripts/test_regenerate_paper.py`

**Interfaces:**
- `jobs(groups: list[str]) -> list[Job]`, with `Job(group, dataset, folder, variant: dict, tool, command: list[str], inputs: dict, externals: dict)`.
- `main(argv) -> int`, with flags:
  - required: `--gempack-release`, `--gp-release`
  - optional: `--group G` (repeatable; one of `book`, `book-sltoht`, `levels`, `horridge`, `levels6`, `gragg`, `gragg-sltoht`, `swap`, `swap-sltoht`, `validation`), `--only NAME` (folder name), `--dry-run`, `--work DIR` (scratch for the compiled TABs; default `%TEMP%\equilibria_gempack_work`), `--gtapdir` (default `C:\runGTAP375`, only for the `.exe`), `--gp` (default `C:\GP`)
- Same resume, cleanup and exit codes as `regenerate_gempack.py`. A group whose prerequisite group is not recorded is refused by name: `book-sltoht` needs `book`; `gragg-sltoht` needs `gragg`.
- `levels6` and `gragg` compile `data/gtap7/gempack_rungtap375_levels/` with `tablo` into `--work` on first use. They refuse to compile when that folder's record does not verify.

- [ ] **Step 1: failing tests:**
  - `test_job_counts` (book 39, book-sltoht 39, levels 6, horridge 3, levels6 12, gragg 45, gragg-sltoht 45, swap 11, swap-sltoht 2, validation 1)
  - `test_matrix_references_stay_unique` (Review Focus 1)
  - `test_no_compiled_gempack_files_in_data` (Review Focus 4)
  - `test_failed_nested_job_is_removed` (Review Focus 8)
  - `test_prerequisite_group_refused`
  - `test_command_paths_are_absolute`
  - `test_resume_skips_recorded_folders`
- [ ] **Step 2:** `uv run pytest tests/scripts/test_regenerate_paper.py -q` → FAIL. **Step 3:** implement with fake runners (as in 3a-1). **Step 4:** → PASS; `--dry-run` lists 203 jobs.
- [ ] **Step 5:** Commit `feat(gempack): regenerate_paper.py lists the corridas_paper and runs/ jobs`.

> The tasks below each add their job module's real behaviour (tests with fake `gtapv7`/`gemsim`/`sltoht` scripts, as in 3a-1 Task 1), then a smoke run and the full group on Windows. Each group is committed on its own (`data(gempack): <group> on Windows`). Every push after the plan's is asked for first.

### Task 3: Burfisher book solutions (input of steps 1 and 6a)

Today the input of steps 1 and 6a is `tests/fixtures/gtap7_gempack/burfisher/`, which came from the 2026-09-24 `tools/gempack-oracle/01-run-all.bat` (branch `tools/gempack-oracle-nus333`, `gtapuv7.exe`/`gtapv7.exe` 12.1.003). This task regenerates those solutions; steps 1 and 6a then run on them (Task 4).

- **Group** `book`. **Tool** `gempack`, `--gempack-release`. **Solver** `gtapuv7` (`gtapv7` for `gtapv7.exp`).
- **Folder** `data/gtap7/nus333/gempack_burfisher/<exp>/`, for the 39 experiments committed today: `gtapuv7 gtapv7 me3a me3b me4a me4b me5 me7a me7b me8 me9a–d tbl45a–c tbl46a–c tbl54a tbl54b tbl55 tbl62a–c tbl63a tbl63b tbl64 tbl65a tbl66 tbl67 tbl75a tbl75b tbl77 tbl78 tbl79 tbl813 tbl93`. `me8` and `tbl813` run their `-dir.exp`.
- **Variant** `{experiment: <exp>, method: book}`. `--data-dir data/gtap7/nus333`.
- **Files** `run.cmf` (header + the `.exp` verbatim, as `01-run-all.bat`), `solution.sl4`, `updated.har` (the `.upd`), `summary.har`, `decomp.har`, `volume.har`, `solve_console.txt`.
- **External** `gtapuv7.exe` (or `gtapv7.exe`) with release.
- **Replaces** `burfisher/<exp>.sl4.gz`, `<exp>.upd`, `<exp>.log.gz` (117 files).

- [ ] **Step 1: failing tests** (`tests/scripts/test_burfisher_job.py`):
  - `test_cmf_header_then_exp_verbatim`
  - `test_dir_exp_for_me8_and_tbl813`
  - `test_book_method_untouched` (Johansen stays Johansen)
  - `test_needs_normal_end`
- [ ] **Step 2:** FAIL → **Step 3:** implement `burfisher_job.py` → **Step 4:** PASS. **Step 5:** commit `feat(gempack): Burfisher book solutions into data/`.
- [ ] **Step 6 (Windows, smoke):** `uv run python data/gtap7/gempack/tools/regenerate_paper.py --gempack-release 12.1.003 --gp-release 11.3.004 --group book --only me9c`. Then check that `data/gtap7/nus333/gempack_burfisher/me9c/source.yaml` says `status: verified` and lists `external.gtapuv7.exe`.
- [ ] **Step 7 (Windows, full):** the same without `--only`. Then `check_tree()` → `[]` for `gtap7/nus333`, and commit `data(gempack): Burfisher book solutions on Windows`.

### Task 4: steps 1 and 6a — `sltoht` on the book solutions

- **Group** `book-sltoht`. **Tool** `sltoht`, `--gp-release`.
- **Folder** `data/gtap7/nus333/sltoht_burfisher/<exp>/`, for the same 39.
- **Variant** `{experiment: <exp>, method: book}`. `--input solution=gtap7/nus333/gempack_burfisher/<exp>`.
- **Files**
  - All 39: `sl4dump.har` (step 6a: sltoht batch `<stem>/c/n/har/<out>/e`, as `Dump-Changes`) and `sltoht_console.txt`.
  - `tbl45a`, `me9c`, `gtapv7` also get step 1: `sl4levels.har` and `export_console.txt` (`export_sl4_levels.py --sl4dump <its own sl4dump.har>`).
- **Replaces**
  - `burfisher/sl4dump_tbl45a.har`, `sl4dump_me9c.har`, `sl4dump_gtapv7.har`.
  - Nothing for step 1's levels or the other 36 dumps: they were never committed. They appear as new oracles; step 6 already makes them today.

- [ ] **Step 1: failing tests** (`tests/scripts/test_levels_job.py`, part 1):
  - `test_dump_sti_batch_answers` (the `.sti` text equals step 6's)
  - `test_step1_levels_only_for_three` (exactly `tbl45a`, `me9c` and `gtapv7` get levels)
  - `test_levels_cross_check_names_drops` (`xcheck-mismatch=N […]` is kept in `export_console.txt`)
- [ ] **Steps 2–5:** FAIL → implement → PASS → commit `feat(gempack): sltoht dumps and levels of the Burfisher book solutions`.
- [ ] **Step 6 (smoke):** `--group book-sltoht --only me9c`. **Step 7 (full):** `--group book-sltoht`, then `check_tree()` → `[]`, then commit `data(gempack): sltoht on the Burfisher book solutions`.

### Task 5: step 2 — levels of `gtapv7.exe` matrix solutions

The 3a-1 folders do not keep the `.sl4`, and the frozen `collect_job` drops it. This tool re-solves the same job with `gtapv7.exe` in a scratch folder, through the 3a-1 `prepare`/`solve`. It keeps the `.sl4` and exports levels against the 3a-1 `sl4dump.har`. That `sl4dump.har` is the record's chained input, so the cross-check proves the re-solve is the 3a-1 solution.

- **Group** `levels`. **Tool** `sltoht`, `--gp-release`. **Solver** `gtapv7`.
- **Folder** `data/gtap7/<ds>/sltoht_tm10_s8-16-32_<capflex|capfix>/`, 6 jobs: capFlex for `gtap7_{3x3,3x4,5x5,10x7,15x10}`, and capFix for `gtap7_20x41`.
- **Variant** `{shock: 10, steps: "8-16-32", closure}`. `--data-dir data/gtap7/<ds>`; `--input sl4dump=gtap7/<ds>/gempack_tm10_s8-16-32_<closure>/sl4dump.har`.
- **External** `gtapv7.exe` with release 12.1.003.
- **Files** `run.cmf`, `solution.sl4`, `solve_console.txt`, `sl4levels.har`, `export_console.txt`.
- **Replaces**
  - `sl4levels_gtap7_{3x3,3x4,5x5,10x7,15x10}_tm10.har`
  - `sl4levels_gtap7_20x41_tm10_s8-16-32_capfix.har`
  - `sl4_gtap7_20x41_tm10_s8-16-32_capfix.sl4`
  - `sl4dump_gtap7_*_tm10.har` and `updated_gtap7_*_tm10.har` (5+5) are byte-identical copies of the tagged `_s8-16-32` files (06dd8788), so the 3a-1 capFlex folders already replace them. Nothing is regenerated for them.

- [ ] **Step 1: failing tests** (`tests/scripts/test_levels_job.py`, part 2):
  - `test_resolve_keeps_sl4`
  - `test_cross_check_against_matrix_dump` (a fake `.sl4` whose %-change differs from the chained `sl4dump.har` → exit 1, no record)
  - `test_levels_record_is_not_gempack_tool` (Review Focus 1)
- [ ] **Steps 2–5:** FAIL → implement → PASS → commit `feat(gempack): levels of the matrix solutions (step 2)`.
- [ ] **Step 6 (smoke):** `--group levels --only sltoht_tm10_s8-16-32_capflex --datasets gtap7_3x3`. **Step 7 (full):** `--group levels`, then `check_tree()` → `[]`, then commit `data(gempack): levels of the matrix solutions on Windows`.

### Task 6: step 3 — Horridge's SIMPLE

- **Model** `data/horridge_simple/` (`kind: model`). Its inputs are *copied* from the tracked `runs/horridge_simple/`; 3b deletes `runs/`:
  - Horridge & Pearson's tpmh0103 archive: `simple.tab`, `simdata.har`, `fixcap.cmf`, `mobcap.cmf`, `input.gdx`, `simpleNLP.gms`, `simpleMCP.gms`, `mpsgevh.gms`
  - written in this repo: `tariff10.cmf`, `simpleNLP_tariff.gms`, `simpleMCP_tariff.gms`
  - `source: Horridge & Pearson tpmh0103 archive`; `license: not stated by the source; to confirm`

  `simple.tab` is Horridge's, not `GTAPv7.tab`, and it is already in the repository.
- **GEMPACK side**
  - Group `horridge`. Tool `gemsim`, `--gp-release`.
  - Folders `data/horridge_simple/gemsim_<fixcap|mobcap|tariff10>/`, variant `{experiment}`.
  - Files `solution.sl4`, `updated.har`, `sl4dump.har` (sltoht, same release), `tablo_console.txt`, `gemsim_console.txt`, `sltoht_console.txt`.
  - Step 3 never ran sltoht or `tariff10`, so the comparator (`compare_gams_vs_gempack.py`) could not be fed. Both are added here: they are what `readme_equilibria.md` reports.
- **GAMS side: Mac, not Windows (D4).**
  - The 5 GAMS results (`data/horridge_simple/gams_<nlp|mcp|mpsge>_<fixcap|tariff10>/`, variant `{form, experiment}`; `mpsge` has no tariff variant) are run on the Mac with GAMS 53.5. The `tool_version` is read from the `.lst`.
  - They are not part of this Windows plan or of `regenerate_paper.py`. They are listed here so that 3b knows where they go.
- **Replaces** nothing committed: every SIMPLE result under `runs/` is git-ignored. The copies on branch `windows/corridas-paper` are stale July files.

- [ ] **Step 1: failing tests** (`tests/scripts/test_gemsim_job.py`, part 1):
  - `test_simple_model_record_lists_inputs`
  - `test_tariff10_runs` (job list)
  - `test_no_gams_job_on_windows` (the `horridge` group has exactly the 3 `gemsim` jobs)
- [ ] **Steps 2–5:** FAIL → implement → PASS → commit `feat(gempack): Horridge SIMPLE into data/horridge_simple`.
- [ ] **Step 6 (smoke):** `--group horridge --only gemsim_fixcap`. **Step 7 (full):** `--group horridge`, then `check_tree()` → `[]`, then commit `data(gempack): Horridge SIMPLE on Windows`.

### Task 7: step 4 — levels of y, qinv, rore, pinv, pva, pe (patched condensed TAB)

- **The model** is `data/gtap7/gempack_rungtap375_levels/gtapv7_levels.{tab,sti}` (Task 1), derived from RunGTAP's 7.0 in `data/gtap7/gempack_rungtap375/`. The driver copies the two files into `--work` and runs `tablo -sti gtapv7_levels.sti` there once, which writes `gtapv7_levels.gss/.gst`.
  - The records chain the model with `--input levels_model=gtap7/gempack_rungtap375_levels`.
  - `external` lists only the compiled `gtapv7_levels.gss`/`.gst` and the `C:\GP` programs. `gtapv7.exe` is not run here: the sl4dump it made is already chained through `--input`.
- **Group** `levels6`. **Tool** `gemsim`, `--gp-release`.
- **Folder** `data/gtap7/<ds>/gemsim_levels6_tm10_s8-16-32_<capflex|capfix>/`, 12 jobs (6 datasets × 2 closures).
- **Variant** `{shock: 10, steps: "8-16-32", closure, tab: gtapv7_levels}`. `--data-dir data/gtap7/<ds>`; `--input sl4dump=gtap7/<ds>/gempack_tm10_s8-16-32_<closure>/sl4dump.har` (the unpatched solution, as step 4 used).
- **Files** `run.cmf` (3a-1 `prepare` with the `Auxiliary files` line rewritten), `default_rdlt0.prm` (capFix only, from `_force_rdlt`), `solution.sl4`, `gemsim_console.txt`, `sl4levels.har`, `export_console.txt`.
- **Replaces**
  - `sl4levels6_gtap7_{3x3,3x4,5x5,10x7,15x10,20x41}_tm10_s8-16-32.har` (capFlex)
  - `…_tm10_s8-16-32_capfix.har` (capFix)
  - `sl4levels6_gtap7_3x3_tm10.har` (the same 3x3 capFlex job, saved twice)
  - `default_gtap7_20x41_rdlt0_windows.prm`, replaced by `gtap7_20x41/gemsim_levels6_tm10_s8-16-32_capfix/default_rdlt0.prm`

- [ ] **Step 1: failing tests** (`tests/scripts/test_gemsim_job.py`, part 2):
  - `test_aux_line_rewritten_once`
  - `test_capfix_prm_is_force_rdlt` (byte-equal to `_force_rdlt` on a tmp copy)
  - `test_compiled_tab_never_in_data` (Review Focus 4)
  - `test_model_read_from_data` (the TAB compiled is `gempack_rungtap375_levels/gtapv7_levels.tab`, never `C:\runGTAP375`)
  - `test_externals_listed`
  - `test_gemsim_needs_normal_end` (Review Focus 5)
- [ ] **Steps 2–5:** FAIL → implement → PASS → commit `feat(gempack): six-variable levels with the patched TAB (step 4)`.
- [ ] **Step 6 (smoke):** `--group levels6 --datasets gtap7_3x3 --only gemsim_levels6_tm10_s8-16-32_capflex`. **Step 7 (full):** `--group levels6`, then `check_tree()` → `[]`, then commit `data(gempack): six-variable levels on Windows`.

### Task 8: step 5 — Burfisher with Gragg (patched uncondensed TAB)

- **The model** is `data/gtap7/gempack_rungtap375_levels/gtapuv7_levels.{tab,sti}` (Task 1). The uncondensed model needs no file beyond RunGTAP's `gtapv7.tab` and `gtapv7.sti`. It is compiled with `tablo -sti gtapuv7_levels.sti` in `--work`.
  - The records chain the model with `--input levels_model=gtap7/gempack_rungtap375_levels`.
  - `external` lists only `gtapuv7_levels.gss`/`.gst` (and `.gsc`) and the `C:\GP` programs.
- **Group** `gragg`. **Tool** `gemsim`, `--gp-release`.
- **Folder** `data/gtap7/nus333/gemsim_burfisher_gragg/<exp>/`, for 45 experiments: the 43 committed today in `burfisher_gragg/`, plus two new ones for the paper (maintainer, 2026-11-03):
  - **`gtapv7`.** Step 5 skipped it because `gtapv7.exp` declares the closure of the CONDENSED model. It runs with Gragg 2-4-6 on step 4's condensed `gtapv7_levels` instead of `gtapuv7_levels`. Expected: nominal ×1.10, real ×1.00.
  - **`tbl53`.** Its shocks are `target% X from file tfd.shk`, which only the RunGTAP GUI solves. They are translated to plain shocks as in the proposal below, **pending the maintainer's approval; nothing runs before it**. If the translation is rejected, `tbl53` is run by hand in the GUI with Gragg 2-4-6, and the exact steps go in its `source.yaml` `notes`.
  - `me8` and `tbl813` run their `-dir.exp`.
  - The Johansen 1-step `.exp` files (TBL45A/B/C, TBL64, ME4A/B) become Gragg 2-4-6.
- **Variant** `{experiment, method: gragg}`. `--data-dir data/gtap7/nus333`.
- **Files** `run.cmf`, `solution.sl4`, `updated.har`, `summary.har`, `decomp.har`, `volume.har`, `gemsim_console.txt`.
- **ME4A and ME4B fail** ("Accuracy too low"), so no folder is written for them. Their old files came from a failed run and are not replaced (see "Not regenerated").
- **Replaces** `burfisher_gragg/<exp>.sl4.gz`, `<exp>.upd`, `<exp>.log.gz` for 41 experiments. `gtapv7` and `tbl53` are new.
- **TBL53 proposal (target%).**
  - **What the `.shk` is.** `tfd.shk` is `TFD_HAT` of `C:\runGTAP375\shocksv7.tab`: `TFD_HAT = 100*[1 - TFD_L]/TFD_L` with `TFD_L = VDFP/VDFB`, the shock that removes the tax. Checked against `data/gtap7/nus333/basedata.har`: `1/(1 + TFD_HAT/100)` equals `VDFP/VDFB` in all 18 cells (max difference 0.0).
  - **The reading.** `target% X` sets the ad valorem rate to X%, so the power goes to `1 + X/100`. It is the same family as `rate% N`, which `mk_exp_directos.py` already translates from the same file. The shock to the power is then known before solving: `100*[(1 + X/100)*(1 + TFD_HAT/100) - 1]` per cell. For example, `tfd(AGR,AGR,USA)` = 9.321930 for a 5 % target from a −3.95 % rate.
  - **No closure swap is needed:** `tfd` is exogenous, and its target level is known from the data.
  - **What has to be resolved.** `mk_exp_directos.py`'s docstring reads `target%` otherwise ("move the instrument until the target moves N%"). The proposal does not adopt that reading, because `shocksv7.tab` holds only the removal shock and Table 5.3 is titled "Changes in domestic sales taxes".
  - **Check of the translated run:** in `updated.har`, `VDFP/VDFB` of the 9 shocked cells must equal 1.05 / 1.10 / 1.02. Optionally, one GUI run of TBL53 compares the `.sl4`.

- [ ] **Step 1: failing tests** (`tests/scripts/test_gemsim_job.py`, part 3):
  - `test_johansen_becomes_gragg_once` (Review Focus 6)
  - `test_skip_list`
  - `test_runs_from_the_package_folder` (the `.exp` names its `.prm`/`.shk` without a path; cwd is a scratch copy of `data/gtap7/nus333`, never `data/` itself)
- [ ] **Steps 2–5:** FAIL → implement → PASS → commit `feat(gempack): Burfisher with Gragg on the patched TAB (step 5)`.
- [ ] **Step 6 (smoke):** `--group gragg --only me9c`. **Step 7 (full):** `--group gragg` (`tbl53` only after the proposal is approved). Expect 43 recorded and ME4A/B refused. Then `check_tree()` → `[]`, then commit `data(gempack): Burfisher with Gragg on Windows`.

### Task 9: steps 6b and 7 — `sltoht` on the Gragg solutions

- **Group** `gragg-sltoht`. **Tool** `sltoht`, `--gp-release`.
- **Folder** `data/gtap7/nus333/sltoht_burfisher_gragg/<exp>/`, for the 45 of Task 8 (ME4A/B refused as prerequisites).
- **Variant** `{experiment, method: gragg}`. `--input solution=gtap7/nus333/gemsim_burfisher_gragg/<exp>`.
- **Files**
  - `sl4dump.har` (step 6b)
  - `sl4levels.har` and `export_console.txt` (step 7, cross-checked against this folder's own `sl4dump.har`)
  - `sltoht_console.txt`
- **Replaces** `burfisher_gragg/sl4levels_<exp>.har` (41 of 43). The step-6b dumps were never committed; `gtapv7` and `tbl53` are new.
- This task also retires the `-gt 6` bug of `corridas_paper.ps1` (step 7 rejected without the local patch): the driver has no step numbers.

- [ ] **Steps 1–5:** `test_step7_reads_step6_dump_of_same_experiment` → FAIL → implement → PASS → commit `feat(gempack): sltoht dumps and levels of the Gragg solutions (steps 6b, 7)`.
- [ ] **Step 6 (smoke):** `--group gragg-sltoht --only me9c`. **Step 7 (full):** `--group gragg-sltoht`, then `check_tree()` → `[]`, then commit.

### Task 10: `runs/nus333_compare`, `runs/9x10_compare` — RunGTAP with the swap closure

- **Group** `swap`. **Tool** `gempack`, `--gempack-release`. **Solver** `gtapv7`.
- **Folders**
  - `data/gtap7/nus333/gempack_tm<k>_s8-16-32_dpsaveswap/` for k = 1…10. These are the tm10 run and the 1–10 % trajectory of `rungtap/trajectory_runner.py`.
  - `data/gtap7/9x10/gempack_tm10_s8-16-32_dpsaveswap/`.
  - Each `.cmf` is `runs/<ds>_compare/rungtap/tm10.cmf` with only the shock and the file names changed (tests pin the swap lines). The template is copied to `data/gtap7/gempack/tools/cmf/<ds>_tm10_dpsaveswap.cmf`, so it is hashed with the tools and survives 3b deleting `runs/`.
- **Variant** `{shock: k, steps: "8-16-32", closure: dpsaveSwap}`. No 3a-1 record exists for these datasets, and the closure value differs anyway.
- **Files** as 3a-1: `updated.har`, `decomp.har`, `sl4dump.har`, `run.cmf`, `solve_console.txt`, `sltoht_console.txt`.
- **Group** `swap-sltoht` (tool `sltoht`, Task 5's tool): levels for nus333 tm10 and 9x10 tm10, in `…/sltoht_tm10_s8-16-32_dpsaveswap/`.
- **Replaces**
  - `sl4dump_nus333_tm10.har`, `updated_nus333_tm10.har`, `sl4levels_nus333_tm10.har`
  - `sl4dump_9x10_tm10.har`, `updated_9x10_tm10.har`, `sl4levels_9x10_tm10.har`
  - the ignored local outputs under `runs/*/rungtap/`
- **Not here:**
  - `trajectory.json` and `comparison.{md,json}`, `comparison_9x10.md`: analyses joining these folders with equilibria's JSON. Rebuilt on the Mac in 3b.
  - `equilibria/*.json`: equilibria results, not GEMPACK; phase 4.
  - The `har2xls` `.xls` files: post-processing for those analyses.

- [ ] **Step 1: failing tests** (`tests/scripts/test_swap_job.py`):
  - `test_swap_lines_kept` (the nus333 `.cmf` keeps `swap dpsave("USA") = del_tbalry("USA")`; 9x10 keeps its 9 swaps)
  - `test_only_shock_changes` (diff of the generated `.cmf` against the tracked one)
  - `test_trajectory_shocks`
- [ ] **Steps 2–5:** FAIL → implement → PASS → commit `feat(gempack): nus333 and 9x10 RunGTAP swap-closure runs into data/`.
- [ ] **Step 6 (smoke):** `--group swap --only gempack_tm10_s8-16-32_dpsaveswap --datasets nus333`. **Step 7 (full):** `--group swap` then `--group swap-sltoht`, then `check_tree()` → `[]`, then commit.

### Task 11: `runs/gempack_20x41_validation` — the recorded abort

- **Group** `validation`. **Tool** `gempack`, `--gempack-release`.
- **Folder** `data/gtap7/gtap7_20x41/gempack_tm10_s8-16-32_dpsaveswap/`.
- **Variant** `{shock: 10, steps: "8-16-32", closure: dpsaveSwap, outcome: abort}`.
- **Files** `run.cmf` (the tracked `tm10.cmf`, 40 swaps), `solve_console.txt`, `assert_arith_fail.har`.
- The control run without swaps (`noswap.cmf`) is already `gtap7_20x41/gempack_tm10_s8-16-32_capflex/`: its `updated.har` equals the local `updated_noswap.har` (36 headers, max abs diff 0). Nothing is rerun for it.
- **Replaces** `runs/gempack_20x41_validation/{solve_console.txt, tm10-assert-arith-fail.har}`. `noswap_console.txt` is replaced by the 3a-1 capFlex `solve_console.txt`.

- [ ] **Steps 1–5:** `test_expected_abort_only_on_signature` (Review Focus 7) → FAIL → implement in `swap_job.py` → PASS → commit `feat(gempack): the 20x41 swap-closure abort is recorded as a result`.
- [ ] **Step 6 (smoke = full, 1 job):** `--group validation`, then `check_tree()` → `[]`, then commit.

### Task 12: Windows instructions

**Files:**
- Modify: `scripts/windows/README.md`. New section "Phase 3a-2: corridas_paper runs into data/". It covers:
  - pull the branch; `uv sync --frozen`
  - read the two releases: the `gtapv7.exe` banner, and the `C:\GP` banner in any `sltoht` console
  - `--dry-run` (203 jobs)
  - the smoke and full command of each group, in dependency order: `book` → `book-sltoht` → `levels` → `horridge` → `levels6` → `gragg` → `gragg-sltoht` → `swap` → `swap-sltoht` → `validation`
  - `check_tree()` → `[]` before each commit
  - `git add` limited to the group's folders (as in the 3a-1 repair)

  It says `corridas_paper.ps1` is superseded but kept until 3b.
- Modify: `data/README.md`, the D1 note on the 66 matrix records.

- [ ] **Step 1:** write it. **Step 2:** `uv run pytest tests/scripts tests/data -q` → PASS; `python scripts/dev/check_layout.py` → 0; `prek run --all-files` → Passed. **Step 3:** commit `docs(windows): how to regenerate the corridas_paper runs into data/`. Push only with the maintainer's permission.

---

## The GTAPv7.tab licence

- **Status:** `data/gtap7/gempack_rungtap375/` keeps RunGTAP 3.75's `gtapv7.tab` and `gtapv7.sti` unchanged, recorded like `data/gtap7/gams_jgea`, with the licence in `notes`: "not stated by the source; to confirm".
- **What uses it:** Tasks 7, 8 and 9 (steps 4, 5, 6b and 7), through the derived `data/gtap7/gempack_rungtap375_levels/`.
- **What stays outside the repository:** the files `tablo` compiles from those TABs, and every `.exe`. They appear only as sha256 in `external`.
- **If the licence turns out not to allow redistribution:** the two model folders leave `data/`. The records of Tasks 7–9 then keep their sha256 in `generated_from`, but no longer verify off a machine that has RunGTAP. That would mean rerecording, not rerunning.
- **Not dependent:**
  - Tasks 3, 4, 5, 10 and 11 use only the compiled `gtapv7.exe`/`gtapuv7.exe`, like 3a-1.
  - Task 6 uses Horridge's own `simple.tab`, already tracked.

## Not regenerated on Windows (listed for 3b)

| old file | why |
|---|---|
| `burfisher_gragg/me4a.*`, `me4b.*` (sl4, upd, log, `sl4levels_`) | Gragg 2-4-6 fails ("Accuracy too low"); the committed files are a failed run. 3b decides what the oracle is for these two. |
| `burfisher/` has no `tbl53` | The book set keeps it out (`target%`). It enters only the Gragg set (Task 8), pending the translation's approval. |
| `sl4levels_sample_sic.txt` | Hand-written oracle text, not a run. Phase 4 (reader fixtures). Its values follow `sl4dump_gtap7_3x3_tm10.har`. |
| `table1_levels_v061_gtap7_3x3.json.gz` | equilibria v0.6.1 on the Mac, not GEMPACK. Phase 4. |
| `make_synthetic_updated_har.py`, `updated_synthetic.har` | Pure Python. Nothing reads the committed HAR. Phase 4. |
| `runs/*/equilibria/*.json` | equilibria results (PATH C-API), not GEMPACK. Phase 4. |
| `runs/nus333_compare/{trajectory.json, comparison.*}`, `runs/9x10_compare/comparison_9x10.md` | Analyses over the Task 10 folders plus equilibria's JSON. 3b, on the Mac. |
| `runs/horridge_simple` GAMS results | Run on the Mac with GAMS 53.5, not here (D4). |
| The LEVA-only *layout* of `sl4levels_*_tm10.har` (5 gtap7, nus333, 9x10) | The current exporter writes LEVA+LEVB+NOLV. The values are regenerated (Tasks 5, 10), but `_oracle.py` EX16 and `test_export_sl4_levels.py`, which read the 1e10-sentinel layout, are a 3b finding. |

## Expected differences against the old files (for 3b's diff)

- **HAR timestamps.** `updated_*`/`sl4dump_*` differ in 12–14 header bytes between reruns. The diff compares headers numerically, as 3a-1 did.
- **Steps 4–7 were GEMSIM/SLTOHT 11.3.004 before and are again.** Values should match. A change in the dropped-variable set (`CNTendwr`, `CNTendw`) is a finding to explain.
- **`sl4levels6_gtap7_3x3_tm10.har` and `…_tm10_s8-16-32.har`** were not byte-identical before. Both are replaced by one folder, and 3b states which one it matched.
- **`table1_levels_v061`'s test** (`test_default_tariff_shock.py`) compares against the 3x3 capFlex `sl4levels6`/`updated`. A red there is a finding about the old files (ADR-0008).

## Windows-only test failures (separate issue, not fixed here)

They fail on the Windows clone before and after this phase; the maintainer asked to leave them for their own issue (2026-11-03):

- `tests/data/test_sources_checks.py::test_file_without_source_is_reported`: it expects `loose/x.gdx`, and the message has `loose\x.gdx` (path separator).
- `tests/scripts/test_make_levels_tab.py::test_cli_escribe_tab_y_sti` and `::test_cli_uncondensed_escribe_gtapuv7_levels`: `write_text` writes CRLF on Windows, and the test expects LF bytes.
- `tests/scripts/test_fbep_switch.py::test_default_bundle_unchanged`: the GAMS bundle builder prints `→` to a cp1252 console (`UnicodeEncodeError`).

## Decisions (maintainer, 2026-11-03)

- **D1.** The 66 matrix records are **not** re-recorded. `data/README.md` gets a note: their `sl4dump.har` was made by `sltoht` 11.3.004, while `tool_version` (12.1.003) is the release of the program that made the primary output, `gtapv7.exe`. Task 12 adds the note.
- **D2.** `external` records the sha256 of `gtapv7.exe` and `gtapuv7.exe` (and of every other executable a record uses) from 3a-2 on.
- **D3.** `.sl4` files are stored raw, not `.gz`.
- **D4.** The Windows GAMS licence is not renewed. Horridge's 5 GAMS results run on the Mac with GAMS 53.5 (Task 6, "Mac").
- **D5.** The nus333 trajectory (Task 10, shocks 1–9) is kept.
- **Model version.** RunGTAP 3.75's `gtapv7.tab` 7.0, not the official 7.1 (`docs/findings/gtapv7_tab_70_vs_71_2026-11-03.md`; the 7.1 is issue #198).
- **Branch.** `main` is merged into `data/phase-3a2-gempack` once #186 lands, which drops `data/gtap7/gempack_purdue/`.
