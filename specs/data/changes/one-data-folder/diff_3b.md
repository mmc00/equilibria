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
