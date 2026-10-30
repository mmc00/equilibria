# One data folder, phase 2b — GAMS MCP references on NEOS, with and without the FBEP injection

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** every GAMS GTAP result still in `tests/fixtures/gtap7*`, `src/equilibria/templates/reference/gtap/comp*` and `output/nus333_neos` is regenerated into `data/`, and every GAMS result exists in two variants: `fbep=header` (the FBEP/FTRV headers injected after `getData.gms`'s `fbep=0`, what the MCP references have used since #24) and `fbep=zero` (the official `getData.gms`: `fbep=0`, `ftrv=evfp-evfb`).

**Architecture:** one switch, `--fbep header|zero`, reaches the bundle builders and `nl_compare`; `regen_gams_reference.py` solves either locally or on NEOS (`--where neos`) through a new `scripts/gtap/neos_job.py`; `write_source.py --remote neos` records a NEOS run from the job file the runner leaves. `regenerate_gams.py` lists every job (local and NEOS, both variants) and keeps its resume rule.

**Tech Stack:** Python 3.13, GAMS 53.5 local (CONVERT, IPOPT, PATH ≤ 1000 rows), NEOS XML-RPC (`https://neos-server.org:3333`, category `cp`, solver `PATH`, input `GAMS`), pytest.

**Spec:** `specs/data/changes/one-data-folder/design.md` (phase 2; "When a regenerated file differs"; a NEOS run records `os: "NEOS server"`), `docs/adr/0010-one-data-folder.md`, phase 2a plan and its ledger. Issue #187 (why two variants).

## Global Constraints

- No move changes a number (ADR-0008): the `header` variant of every result must reproduce the file it replaces; any difference stops the phase for that file until explained in the PR.
- The `zero` variant is new: it has no old file; it is checked against the data identity instead (Task 4 oracle), committed BEFORE the runs.
- Maintainer rules: ask before every push/merge; never `git stash`; one final reviewer only; repo text and commit messages in English.
- A NEOS `source.yaml` carries the NEOS run date and enough configuration to replicate it: job number, submit/finish time (UTC), category/solver/input method, the options sent, the GAMS version NEOS reports, the sha256 of the exact `.gms` sent. It never carries the job password or an e-mail address.
- `run.machine` of a NEOS result is `os: "NEOS server"`, the other fields empty (design).
- Folder name: `gams_<form>_<mode>_ifsub<k>_fbep-<header|zero>`; variant key `fbep` with values `header` / `zero`. Datasets whose FBEP is zero everywhere (`nus333`) get one folder without the suffix and no `fbep` key.
- `scripts/gtap/*` is gitignored except a whitelist: a new script there needs its `!` line in `.gitignore`.

## Jobs

| Form | Where | Datasets × ifSUB | Variants | Folders |
|---|---|---|---|---|
| `nl` (CONVERT) | local | 3x3, 3x4, 5x5, 10x7, 15x10 × {0,1} | header, zero | 20 |
| `nlp` (IPOPT) | local | altertax: 3x3, 3x4, 5x5, 10x7; gtap: 3x3, 5x5, 10x7 × {0,1} | header, zero | 28 |
| `mcp` (PATH) | local | 3x3 ifSUB=1, gtap and altertax | header, zero | 4 |
| `mcp` (PATH) | NEOS | gtap: 3x3 {0}, 5x5/10x7/15x10 {0,1}, 20x41 {1}; altertax: 3x3 {0}, 3x4/5x5/10x7/15x10 {0,1} | header, zero | 34 |
| nus333 MCP | local (651 rows) | — | one | 1 |
| 9x10 `COMP` | Task 6 decides | — | Task 6 decides | ≤ 4 |

The 26 phase-2a folders are regenerated under the new names (their variant: `nl` = zero, `nlp`/`mcp` = header) and must hash-identical to the 2a outputs.

## Review Focus

1. A NEOS job that fails, is killed or never finishes in the middle of the 34: no folder for it, the next run resumes from it (Task 2 test `test_neos_failure_leaves_no_folder`).
2. NEOS says `Done` but PATH did not solve (model status not 1/2 in the returned `.lst`): not recorded (Task 2 test `test_neos_done_but_unsolved_is_refused`).
3. NEOS runs another GAMS than the Mac: `tool_version` is the one in the returned `.lst`, never the local one (Task 2 test `test_neos_tool_version_from_lst`).
4. Two variants of a result: `reference(...)` without `fbep` must refuse, not pick one (Task 3 test `test_reference_needs_fbep_when_both_exist`).
5. The NEOS password: never written to disk in the result folder (Task 2 test `test_neos_record_has_no_password`).

---

### Task 1: the `--fbep header|zero` switch

**Files:**
- Modify: `scripts/gtap/build_gtap7_pure_local_bundle.py`, `scripts/gtap/build_gtap7_altertax_neos_bundle.py` (the FBEP/FTRV injection blocks), `scripts/gtap/nl_compare.py` (`_build_standalone_gms` and the getData replacements), `scripts/gtap/regen_gams_reference.py` (`_build`, `_verify_subsidy_aware`)
- Test: `tests/scripts/test_fbep_switch.py`

**Interfaces:**
- Produces: builders and `nl_compare.py` accept `--fbep {header,zero}`; defaults keep today's behaviour (builders `header`, `nl_compare` `zero`). `regen_gams_reference.main` accepts `--fbep` (required). `_verify_subsidy_aware(gdxdump, gdx, fbep)`: `header` → `ytax(ft)` and `ytax(fs)` nonzero at check (today's check); `zero` → `ytax(fs)` zero in every period.
- `nl_compare._build_standalone_gms(..., fbep: str = "zero")`; `header` injects the FBEP/FTRV assignments after the `ftrv = evfp - evfb` line, reusing `_fbep_ftrv_har_as_assignments` from `build_gtap7_pure_local_bundle.py`.

- [ ] **Step 1: failing tests**, on `gtap7_3x3`:
  - `test_default_bundle_unchanged[gtap|altertax]`: the builder's output without `--fbep` equals its output with `--fbep header`, and its sha256 equals the `model.gms` recorded in `data/gtap7/gtap7_3x3/gams_mcp_<mode>_ifsub1/` (2a).
  - `test_zero_bundle_has_no_injection[gtap|altertax]`: `--fbep zero` output contains `fbep(fp,a0,r) = 0 ;` and no `FBEP(` assignment after it.
  - `test_nl_standalone_default_unchanged`: `_build_standalone_gms("base", ..., ifsub=0)` equals `data/gtap7/gtap7_3x3/gams_nl_altertax_ifsub0/gams_base.gms`.
  - `test_nl_standalone_header_injects`: with `fbep="header"` the text has `FBEP("Land","Food","EU_28") = -24050.8515625` (or the dataset's own key spelling) after the `ftrv(fp,a0,r) = evfp` line.
  - `test_verify_zero_rejects_subsidy` / `test_verify_header_rejects_blind`: `_verify_subsidy_aware` on the 2a 3x3 altertax MCP GDX passes with `header` and raises with `zero`.
- [ ] **Step 2:** run, all red (`needs_gdxdump` on the two `_verify` tests).
- [ ] **Step 3:** implement the switch.
- [ ] **Step 4:** green; `tests/scripts/test_nl_compare_local.py`, `test_regen_gams_reference.py` still green.
- [ ] **Step 5:** commit `feat(gtap): GAMS bundles and nl_compare take --fbep header|zero (#187)`.

### Task 2: a NEOS job, recorded

**Files:**
- Create: `scripts/gtap/neos_job.py` (+ `.gitignore` whitelist line)
- Modify: `scripts/gtap/regen_gams_reference.py` (`--where local|neos`), `scripts/dev/write_source.py` (`--remote neos`)
- Test: `tests/scripts/test_neos_job.py`, `tests/scripts/test_write_source.py`

**Interfaces:**
- `neos_job.run(gms: Path, out: Path, *, options: str = "", proxy=None, poll_s: float = 15, timeout_s: float = 8 * 3600) -> NeosRun`; `proxy` defaults to `xmlrpc.client.ServerProxy(NEOS_URL)` and is the seam the tests fake. It submits `gms` (category `cp`, solver `PATH`, input `GAMS`, `wantgdx/lst/log=yes`, `options` = the bundle's `path.opt` text), polls, writes `out.gdx`, `model.lst`, `neos.log` into `out` and `neos_job.json` = `{job, submitted_utc, finished_utc, category, solver, input_method, options, gams_version, sent_sha256}`. Raises `NeosError` on `Failed`/`Killed`/`Bad job`/timeout, on a missing `out.gdx`, or on a model status outside {1, 2} in `model.lst`. The e-mail NEOS requires comes from `--email` / `NEOS_EMAIL` and is not written.
- `regen_gams_reference --where neos` builds as today, calls `neos_job.run` instead of local GAMS; same output files plus `neos_job.json`.
- `write_source.py --remote neos`: after the command succeeds, reads `<folder>/neos_job.json`; `run.machine = {os: "NEOS server"}` (rest empty), `run.date = finished_utc`, `tool_version` = `gams_version` (it refuses a `--tool-version` that disagrees), a `neos:` block with the job fields; `neos_job.json` stays among the outputs.

- [ ] **Step 1: failing tests** with a fake proxy (a class with `ping`, `submitJob`, `getJobStatus`, `getFinalResults`, `getOutputFile`) serving files from the 2a 3x3 MCP folder: `test_neos_job_writes_outputs`, `test_neos_failure_leaves_no_folder` (status `Failed` → `NeosError`, `out` absent after `regen_gams_reference`), `test_neos_done_but_unsolved_is_refused` (`.lst` with `MODEL STATUS 5`), `test_neos_tool_version_from_lst` (`.lst` header `GAMS 52.5.0` → record `52.5.0`), `test_neos_record_has_no_password` (no file under the folder contains the fake password or the e-mail), `test_write_source_remote_neos` (machine, date, `neos:` block).
- [ ] **Step 2:** red. **Step 3:** implement. **Step 4:** green.
- [ ] **Step 5: one real smoke job** (maintainer present): `gtap7_3x3` gtap ifSUB=1 header on NEOS into a scratch folder; its `out.gdx` vs the local 2a one with `gdxdiff RelEps 1e-8`. Report the job number.
- [ ] **Step 6:** commit `feat(gtap): GAMS MCP references can be solved on NEOS and recorded as such`.

### Task 3: the job table with both variants

**Files:**
- Modify: `scripts/gtap/regenerate_gams.py`, `src/equilibria/sources.py` only if `reference()` needs it (it should not: it already refuses two matches)
- Test: `tests/scripts/test_regenerate_gams.py`, `tests/data/test_sources_reference.py`

**Interfaces:**
- `Job` gains `fbep: str | None` and `where: str` (`local`/`neos`); `jobs()` returns the 86 rows of the Jobs table (`nus333` and 9x10 come in Task 6); folder `gams_<form>_<mode>_ifsub<k>_fbep-<v>`; `job_command` adds `--variant fbep=<v>`, `--fbep <v>`, and for NEOS `--where neos` + `--remote neos`. New flag `--where {local,neos,all}` (default `local`) selects rows.

- [ ] **Step 1: failing tests:** `test_jobs_are_the_table` updated (counts per form/where/variant above), `test_neos_rows_carry_remote`, `test_reference_needs_fbep_when_both_exist` (two recorded folders differing only in `fbep` → `reference(ds, "gams", form=..., mode=..., ifsub=...)` raises, with `fbep=` it returns the one).
- [ ] **Step 2–4:** red, implement, green.
- [ ] **Step 5:** commit `feat(gtap): regenerate_gams lists both FBEP variants and the NEOS jobs`.

### Task 4: oracle, then the local runs

**Files:**
- Create: `tests/data/test_fbep_variants.py`
- Create: the 52 local folders; delete the 26 2a folders (replaced by renamed regenerations)

- [ ] **Step 1: the oracle, committed before any run** (`test(data): what each FBEP variant must hold`): for every recorded GAMS MCP/NLP result with `fbep` and a subsidised cell (FBEP ≠ 0 in its HAR), at `base`: `zero` → `pfa/pf == EVFP/EVFB` (rel 1e-6) and `ytax(fs) == 0`; `header` → `pfa/pf == 1 + (ftrv − FBEP)/EVFB` with `ftrv = FTRV` where the header is nonzero, else `EVFP − EVFB` (the injection overrides only nonzero cells), rel 1e-6. Skips while nothing is recorded; a test of the test feeds it a GDX of the wrong variant and expects red. Commit alone.
- [ ] **Step 2:** `uv run python scripts/gtap/regenerate_gams.py --tool-version 53.5.0 --where local` (log to the ledger folder); every job recorded.
- [ ] **Step 3:** the regenerated 2a variants (nl `zero`, nlp/mcp `header`) hash-identical to the 2a outputs (compare `outputs:` sha256 of old vs new `source.yaml`); then `git rm` the 26 2a folders.
- [ ] **Step 4:** oracle green on all local results; diff `header` vs `zero` per result (`gdxdiff RelEps 1e-6`, list of differing symbols) → `diff_summary.md`.
- [ ] **Step 5:** rewire the 2a consumers to say `fbep=` (`.nl` gate → `zero`, NLP/MCP gates → `header`); `_KNOWN_GAPS` unchanged. Commit results + rewiring together: `refactor(data): local GAMS results in both FBEP variants`.

### Task 5: the NEOS runs

- [ ] **Step 1:** `regenerate_gams.py --tool-version <NEOS GAMS> --where neos` (34 jobs; rerun until every row is recorded; a row that fails three times is reported, not forced).
- [ ] **Step 2:** `header` results vs the files they replace (`tests/fixtures/gtap7/<ds>/out_gtap_shock_ifsub<k>.gdx`, `tests/fixtures/gtap7_altertax/<ds>/out_altertax_ifsub<k>.gdx`): `gdxdiff RelEps 1e-8`; every difference explained in `diff_summary.md` (expected from 2a: `pwmg` bounds on no-margin routes, the iterloop change).
- [ ] **Step 3:** oracle green on the NEOS results; `header` vs `zero` diff into `diff_summary.md`.
- [ ] **Step 4:** rewire MCP consumers to `reference(..., fbep="header")` (`git grep -lE 'out_gtap_shock_ifsub|out_altertax_ifsub|comp_gtap7_'`), keep `data/gdx_tests/` as is, update `scripts/gtap/check_parity_gates_stamp.py` `INPUT_TREES`, delete the replaced fixtures and their `comp_*.gms`. Commit: `refactor(data): GAMS MCP references from NEOS, both FBEP variants`.

### Task 6: nus333 and 9x10

- [ ] **Step 1 (gate):** find how `output/nus333_neos/out.gdx` (commit `beec5a0`: GAMS 53.5 local, `bench_nus333_dual.py`) and `reference/gtap/comp/COMP_generated.csv`, `comp_neos/COMP.csv` (commit `a0cab08`, no generator named) are produced. If a 9x10 CSV has no reproducible generator, STOP and ask the maintainer (keep as `kind: data` with its provenance, or drop).
- [ ] **Step 2:** nus333 locally into `data/gtap7/nus333/gams_mcp_nus333/` (one variant: FBEP is zero in its data); diff vs the old GDX.
- [ ] **Step 3:** whatever Step 1 decides for 9x10 (both variants if regenerated: its GDX carries 32 FBEP cells).
- [ ] **Step 4:** rewire (`_parity_datasets.py`, `diff_nus333_full.py`, `test_nus333_neos_parity.py`, `gtap_parity_pipeline.py`, `gtap_model_equations.py:152-167`, `solver/path_capi.py:44`, `test_gtap_baseline_mirror.py`, `docs/site/guide/benchmarks.md`), drop `output/nus333_neos` and its `.gitignore` exception, `reference/gtap/comp*` and its `pyproject.toml` exclusion. Commit.

### Task 7: close

- [ ] **Step 1:** `data/README.md`: the `fbep` variant, NEOS results and their `neos:` block, the new folder names.
- [ ] **Step 2:** gates (`scripts/gtap/run_parity_gates.py`, GAMS 53 on PATH) green; full suite vs a fresh `before.txt` of `origin/main` (names, vanished, skip reasons; restore `Results.gdx`); Sphinx `-W`; `tests/docs`; `check_layout.py`; `prek run --all-files`.
- [ ] **Step 3:** commit `docs(data): map lists both FBEP variants and the NEOS results`.
- [ ] **Step 4:** one final reviewer; then push and PR only with the maintainer's permission. The PR body: `diff_summary.md`, the two-variant rationale (#187), NEOS job numbers.
