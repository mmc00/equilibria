# Phase 3b — old files against their `data/` records

Each old file under `tests/fixtures/gtap7_gempack/` (and `tests/fixtures/nus333_bienestar_gempack.json.gz`)
was compared with the record that replaces it before it was deleted
(`plan-phase-3b.md`, Step 1 of Tasks 2–7). HAR files are read with
`equilibria.babel.har.read_har`; every header except `XX*` (sltoht/GEMPACK
history and timestamps) is compared. Numeric headers: max absolute difference
over all cells in float64, set elements compared too; text headers: exact.
A group is clean when the header sets, set elements, shapes and text headers
are equal and the max abs difference is ≤ 1e-9. The old file is located by
name, the new one through `equilibria.sources.reference(...)`.

| Task | group | files compared | max abs diff | result |
|---|---|---|---|---|
| 2 | `decomp_<ds>_<tag>.har` → `gtap7/<ds>/gempack_<tag>_<closure>/decomp.har` | 66 | 0 | clean |
| 2 | `sl4dump_<ds>_<tag>.har` → `…/sl4dump.har` | 66 | 0 | clean |
| 2 | `updated_<ds>_<tag>.har` → `…/updated.har` | 66 | 0 | clean |
| 2 | `{sl4dump,updated}_gtap7_<ds>_tm10.har` → the `tm10_s8-16-32_capflex` record | 10 | 0 | clean |
| 2 | `{sl4dump,updated}_{nus333,9x10}_tm10.har` → `gempack_tm10_s8-16-32_dpsaveswap/` | 4 | 0 | clean |
| 3 | `sl4levels_<ds>_tm10.har` (LEVA-only, 1e10 sentinel) → `gtap7/<ds>/sltoht_tm10_s8-16-32_<capflex\|dpsaveswap>/sl4levels.har` (5 gtap7 capFlex, nus333, 9x10) | 7 | 0 | clean (layout, below) |
| 3 | `sl4levels_gtap7_20x41_tm10_s8-16-32_capfix.har` → `gtap7_20x41/sltoht_tm10_s8-16-32_capfix/sl4levels.har` | 1 | 0 | clean |
| 3 | `sl4_gtap7_20x41_tm10_s8-16-32_capfix.sl4` → `…/sltoht_tm10_s8-16-32_capfix/solution.sl4` | 1 | 0 (34 numeric headers; `read_sl4_levels` pre/post of 252 variables equal) | clean; text differs only in `WDIR` and `XX*` (below) |
| 3 | `sl4levels6_<ds>_tm10_s8-16-32[_capfix].har` → `gtap7/<ds>/gemsim_levels6_tm10_s8-16-32_<closure>/sl4levels.har` | 12 | 0 | clean |
| 3 | `sl4levels6_gtap7_3x3_tm10.har` → the 3x3 capFlex `gemsim_levels6` record | 1 | 0 | clean (Review Focus 2, below) |
| 3 | `default_gtap7_20x41_rdlt0_windows.prm` → `gtap7_20x41/gemsim_levels6_tm10_s8-16-32_capfix/default_rdlt0.prm` | 1 | — | byte-identical |

### Task 3 notes

- **Layout (Review Focus 1).** The seven old `sl4levels_*_tm10.har` hold all 252
  variables (251 for nus333) as LEVA, with every cell 1e10 for a variable without
  levels. The new `sl4levels.har` holds only the 44 variables with levels (LEVA
  under the same `NNNN` header, LEVB under `BNNN`) and lists the others by name in
  `NOLV`. Compared cell by cell: the 44 level headers are equal (max abs 0, same
  names and set elements); each of the 208 (207) all-sentinel variables is absent
  from the new file and named in `NOLV`, and `NOLV` has exactly that many names;
  no variable is partly sentinel; the new file has no level header the old one
  lacks. The variable name is taken from the long name: the 12-character
  coefficient name is cut (`CNTtech_at..` for `CNTtech_atmfsd`).
- **`sl4levels6_gtap7_3x3_tm10.har` vs `…_tm10_s8-16-32.har` (Review Focus 2).**
  The two old files are numerically identical (100 headers, max abs 0). The record
  `gtap7_3x3/gemsim_levels6_tm10_s8-16-32_capflex/sl4levels.har` is
  byte-identical to `…_tm10_s8-16-32.har`; `…_tm10.har` differs from it only in
  bytes outside the numbers (max abs 0).
- **The 20x41 `.sl4`.** 97 bytes differ (same size). `WDIR` names the Windows
  working folder (`runs\gempack_matrix\…` before, `runs\gempack_work\resolve\…`
  now), and `XXCD`/`XXHS`/`XXTD` are the run's date and history.
| 4 | `burfisher/<exp>.sl4.gz` (gunzipped) → `nus333/gempack_burfisher/<exp>/solution.sl4` | 39 | 0 (31 numeric headers; `read_sl4_levels` pre/post equal) | clean; text headers name other paths (below) |
| 4 | `burfisher/<exp>.upd` → `…/updated.har` | 39 | 0 | clean |
| 4 | `burfisher/sl4dump_<exp>.har` → `nus333/sltoht_burfisher/<exp>/sl4dump.har` (tbl45a, me9c, gtapv7) | 3 | 0 | clean |
| 4 | `burfisher/<exp>.log.gz` → `…/solve_console.txt` | 39 | text, not compared numerically | differs (paths, times, banner) as expected |
| 5 | `burfisher_gragg/<exp>.sl4.gz` (gunzipped) → `nus333/gemsim_burfisher_gragg/<exp>/solution.sl4` | 41 | 0 | clean; same text headers as Task 4 |
| 5 | `burfisher_gragg/<exp>.upd` → `…/updated.har` | 41 | 0 | clean |
| 5 | `burfisher_gragg/sl4levels_<exp>.har` → `nus333/sltoht_burfisher_gragg/<exp>/sl4levels.har` | 41 | 0 | clean |
| 5 | `burfisher_gragg/<exp>.log.gz` → `…/gemsim_console.txt` | 41 | text, not compared numerically | differs, as expected |
| 5 | `burfisher_gragg/me4a.*`, `me4b.*` | — | — | not compared: deleted with no replacement (D1) |

### Tasks 4–5 notes

- **`.sl4` text headers.** `CMDF` (the command file), `CMDL`, `CMFN`, `VDSC` and
  `WDIR` differ in every experiment. Checked line by line: in `CMDF` only `file`,
  `updated file`, `solution file` and `auxiliary files` statements differ (old
  absolute paths under `tools\gempack-oracle\`, new names relative to the run
  folder); the others name the `.cmf` and the working folder. **Exception:**
  `me8` and `tbl813` (the `-dir.exp` runs) also drop `aux files =
  C:\runGTAP375\GTAPUV7;` and add `CPU = yes;` and `NDS = yes;` (CPU timing, no
  displays: output options). Their 31 numeric headers are still equal (max abs 0).
- **Gragg `.sl4` have no `PLEV`** in the old files or the new ones, so
  `read_sl4_levels` does not apply; all numeric headers are compared instead
  (max abs 0).
- **Consoles.** The old `.log.gz` and the new console text differ in paths, times
  and release banners (100–4474 changed lines for the book, ~12.5k–47k for Gragg,
  which logs every step). They hold no oracle number; the readers that parse them
  are handled in Task 4.
- **Readers of the old layout (Review Focus 1).** Two oracles read the 1e10
  sentinel to find the variables without level; both now read `NOLV` instead,
  and no assertion changed: `tests/compare/test_sl4.py::_oracle_pre` (EX16) and
  `tests/scripts/test_export_sl4_levels.py` (the synthetic text gives the `NOLV`
  variables the sentinel back; the parametrize ids stay `sl4levels_<ds>_tm10`).
  `tests/compare/test_gempack3x3.py` (EX14) reads the new file through `from_har`
  unchanged: its mapping compares post only, and `periods_compared == ["post"]`
  and `unmatched == []` still hold.
- **Method and shocks from `run.cmf` (Task 4).** The old readers took the method
  (`Method = …`, last one wins) and ME8/TBL813's shocks from the gzipped GEMPACK
  logs; the records keep the command file that ran (`run.cmf`). For the 38 book
  experiments the method read from `run.cmf` equals the one read from the old log
  in every case; ME8's 63 and TBL813's 1 `Shock` lines are the same in both.
  `test_mk_exp_directos` pins them to the old logs by sha256
  (`test_the_cmf_shocks_are_the_2026_09_30_logs`), because `run.cmf` now also
  equals the `-dir.exp` the script under test writes.
- **Task 5: two maintainer decisions (2026-11-05).**
  `test_las_corridas_del_paso_5_son_todas_gragg` used to read the method of every
  old Gragg file; ME4A/B have no Gragg record (D1), so its assertion is now
  `set(EXERCISES) - con == {"ME4A", "ME4B"}`, with a comment naming D1. The
  Gragg records also hold TBL53 and GTAPV7, which the old folder lacked: their
  notebooks (`tbl53.ipynb`, `gtapv7.ipynb`) were generated and added to
  `MATCHED` in `tests/examples/test_burfisher_notebooks.py` (44 → 46 tests).

### Task 6 notes

- **Source: the Gragg records, not the book.** The plan says the welfare JSON is
  computed from the book solutions; the old script read the step-6 dumps of the
  Gragg runs. Recomputed from `sltoht_burfisher_gragg/<exp>/sl4dump.har` (runs
  kept only if `gemsim_burfisher_gragg/<exp>/gemsim_console.txt` says "completed
  without error"), all 41 old entries are equal after `json.load` (exact); from
  the book, none of 36 is. The record
  `data/gtap7/nus333/sltoht_burfisher_gragg_welfare/` (`tool: sltoht`,
  `method: gragg`, `output: welfare`; renamed from `gempack_burfisher_welfare/`
  at the Task 7 checkpoint, maintainer 2026-11-05) hashes the 86 Gragg folders it reads as inputs.
- **Two new entries.** The Gragg records also hold GTAPV7 and TBL53, so the JSON
  has 43 exercises. Consumers index it by exercise name; no test changes.
- `test_reference_keys_are_complete` also accepts a call that names `method=`
  (a Burfisher record), as it did `experiment=`.

### Task 7 notes

Moved byte for byte, each with a `write_source.py --no-run` record (`kind: data`,
found with `dataset_dir()`, as the phase 4 snapshots next to their dataset):

| old (`tests/fixtures/gtap7_gempack/`) | new | `dataset_dir(...)` |
|---|---|---|
| `sl4levels_sample_sic.txt` | `data/gtap7/gtap7_3x3/sl4levels_sample_sic/` | `gtap7_3x3_sl4levels_sample_sic` |
| `table1_levels_v061_gtap7_3x3.json.gz` | `data/gtap7/gtap7_3x3/equilibria_table1_v061/` | `gtap7_3x3_equilibria_table1_v061` |
| `updated_synthetic.har`, `make_synthetic_updated_har.py` | `data/gtap7/synthetic_updated/` | `gtap7_synthetic_updated` (the test puts the folder on `sys.path`) |
| `burfisher/readme.md`, `burfisher_gragg/readme.md` | `data/gtap7/nus333/{gempack_burfisher,gemsim_burfisher_gragg}/readme.md` | records `nus333_gempack_burfisher`, `nus333_gemsim_burfisher_gragg` |

- **`updated_synthetic.har`.** Regenerated with its generator: same values (max
  abs 0), one header byte differs (offset 315; the HAR writer changed since
  2026-07). The committed bytes are kept; no test reads them (the tests build
  their own copy in `tmp_path`). The generator's only change is its default
  output, now its own folder.
- **The two `readme.md`.** A `README.md` is exempt from hashing only at the
  root of `data/`; inside `nus333/` it would become a file of the `nus333`
  record and make every result that hashes that dataset stale. So each parent
  folder gets its own `--no-run` record, and the name stays lowercase (layout
  rule; `nus333/readme.md` is the precedent). Both were rewritten in English
  for the record layout (`solution.sl4`, `run.cmf`, ...); the facts are the old
  ones, and the old provenance lines (2026-09-24, `01-run-all.bat`) are dropped
  because each `<exp>/source.yaml` now says how its run was made.
- `sl4levels_sample_sic.txt` keeps its header, which names the pre-3b paths
  (`sl4dump_gtap7_3x3_tm10.har`, `runs/...guide.md`); its `source.yaml` says so.

### Task 8 notes

Every tracked file of `runs/` (39) either has a byte-identical copy in `data/`
(sha256) or moves per D3:

| old | new |
|---|---|
| `horridge_simple/` copies of Horridge's files (13) | already `data/horridge_simple/` (identical) |
| `{nus333,9x10}_compare/rungtap/tm10.cmf`, `gempack_20x41_validation/tm10.cmf` | already `data/gtap7/gempack/tools/cmf/*_tm10_dpsaveswap.cmf` (identical) |
| `gempack_20x41_validation/tm10-assert-arith-fail.har` | already `data/gtap7/gtap7_20x41/gempack_tm10_s8-16-32_dpsaveswap/assert_arith_fail.har` (identical) |
| the 8 analysis scripts | `scripts/gtap/rungtap_compare/` (`compare_report.py` → `compare_nus333_report.py`) |
| `comparison*.md`, `comparison.json`, `trajectory.json`, `equilibria/*.json` | `docs/archive/findings/rungtap_compare/{nus333,9x10}/` |
| `gempack_20x41_validation/{README,RESULTADO}.md`, `noswap.cmf`, `noswap_console.txt`, `solve_console.txt` | `docs/archive/findings/rungtap_compare/gempack_20x41_validation/` (`readme.md`, `results.md`) |
| `horridge_simple/readme_equilibria.md` | `docs/archive/findings/rungtap_compare/horridge_simple.md` |
| `gempack_updated_har_guide.md` | `docs/archive/findings/gempack_updated_har_guide_2026-07-22.md` |

- **Where the scripts read and write.** The RunGTAP `.xls` exports they read
  were never tracked (git-ignored under `runs/`); the scripts now read them, and
  write their reports, under the git-ignored `notes/tmp/rungtap_compare/<case>/`,
  where `compare_{nus333,9x10}_rungtap.py` also write their JSON. The 2026
  outputs are the archived copies.
- **`compare_gams_vs_gempack.py`** takes the Horridge records as defaults
  (`--experiment fixcap`): run on the Mac it gives GAMS/MCP vs NLP median
  4.26e-8, 7.97e-8, 9.64e-9, 3.83e-8 pp and GAMS vs GEMPACK 100 % within 1 pp,
  the numbers the archived `horridge_simple.md` reports.
- **Guard.** `test_no_code_reads_runs` used `\bruns/`, which `git grep -E` on
  macOS does not match: it saw only `"runs" /`. With a portable pattern it found
  the text mentions too. It exempts the git-ignored work folders of two hashed
  tools (`runs/gempack_work`, `runs/gempack_matrix`) and `test_swap_job.py`
  (git bytes of the removed `.cmf`, skips after 3b, as the plan expects).
- `scripts/windows/corridas_paper.ps1` is removed (maintainer, 2026-11-05);
  `scripts/windows/README.md` now opens with the `regenerate_paper.py` group of
  each of its steps.
