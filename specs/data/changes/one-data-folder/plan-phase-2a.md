# One data folder — Phase 2a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the GAMS GTAP v7 model into `data/gtap7/gams/` and regenerate on the Mac, with local GAMS 53, every GAMS result the Mac can produce (all `.nl` in ifSUB 0 and 1, every NLP/IPOPT solve, the two PATH solves under the demo cap) into `data/gtap7/<ds>/gams_<form>_<mode>_ifsub<k>/`, each with a `source.yaml`; tests ask for them through `reference()`.

**Architecture:** One driver, `scripts/gtap/regenerate_gams.py` (the Mac twin of `scripts/windows/regenerate_gempack.py`), lists the jobs and runs each under `write_source.py`. Each job calls one single-job entry point that writes into `--out DIR`: `nl_compare.py --gams-local` (CONVERT) for `.nl`, `regen_gams_reference.py` (new; replaces `regen_altertax_reference.py`) for GDX. Old fixtures are diffed against the new results (`gdxdiff`, `.nl` parser) before they are deleted; consumers switch from hand-built paths to `reference(dataset, "gams", form=..., mode=..., ifsub=...)`.

**Tech Stack:** Python ≥3.10, PyYAML, pytest, GAMS 53 (CONVERT, IPOPT, PATH ≤1000 rows), `gdxdiff`, `uv`, prek.

**Spec:** `specs/data/changes/one-data-folder/design.md` (phase 2; "When a regenerated file differs"; "Asking for a reference"), `docs/adr/0010-one-data-folder.md`. Phase 2 is split (maintainer, 2026-10-29): **2a** here, on the Mac; **2b** the PATH solves above the demo cap, on NEOS.

## Global Constraints

- No number changes by moving (ADR-0008). A regenerated result that differs from the file it replaces stops the phase for that file until the difference is explained in the PR; a test that turns red with the new file is a finding about the old one, never a reason to keep it or to loosen a tolerance.
- Local license, measured 2026-10-29: CONVERT and IPOPT have no size limit; PATH and CONOPT stop at 1000 rows (Demo; the personal license does not add PATH). Hence `gtap7_3x3` ifSUB=1 MCP is local (pure: 3 × Optimal; altertax: 804 rows) and ifSUB=0 is not (Licensing Problem) → 2b.
- GAMS binary: `/Library/Frameworks/GAMS.framework/Versions/53/Resources/gams`, `--tool-version 53.5.0` (the version `gams` prints, measured 2026-10-29; the driver fails and names it if it differs).
- GDX are never compared by bytes (the header carries a date): `gdxdiff OLD NEW RelEps=1e-9`; "No differences found" = identical.
- Variant keys (the design leaves the vocabulary open): `form` ∈ {`nl`, `nlp`, `mcp`}, `mode` ∈ {`gtap`, `altertax`} (the names `nl_compare --mode` uses), `ifsub` ∈ {0, 1}. Folder name `gams_<form>_<mode>_ifsub<k>`. File names inside: `gams_<phase>.{nl,row,col}` + `comp_<phase>.gms` (the exact `.gms` CONVERT read) for `nl`; `out.gdx` + `model.gms` (+ `path.opt` when used) for `nlp`/`mcp`.
- Each result folder stores the exact `.gms` that ran, so the run is replicable from the folder alone.
- Out of scope (2b): every other MCP/PATH GDX — pure (`tests/fixtures/gtap7/<ds>/out_gtap_shock_*` + `comp_*.gms`) and altertax (`tests/fixtures/gtap7_altertax/<ds>/out_altertax_*`, `ifMCP=1`, NEOS #24); `reference/gtap/comp*/`; `output/nus333_neos/`. Out of scope (later phases): `scripts/gtap/*/` comparison folders with `.gms` (phase 6), `results/` (PEP, phase 5), the dictionary pairing of the `.nl` gate (phase 7, branch `fix/23-nl-gate-vacuous`, which touches `nl_compare.py`: it rebases on this).
- Suite before/after compared by name, vanished names and skip reasons (`before.txt` from `origin/main`, GAMS 53 on `PATH`); restore `src/equilibria/templates/reference/pep2/scripts/Results.gdx` after each full run.
- Commits in English, `type(scope): description`, ending with the session's `Co-Authored-By` line. `ruff` hooks may rewrite on commit: re-add and recommit.

### Jobs (26)

| Family | Datasets | ifSUB | Folder | Entry point | Replaces |
|---|---|---|---|---|---|
| nl | 3x3, 3x4, 5x5, 10x7, 15x10 | 0, 1 | `gams_nl_altertax_ifsub<k>` | `nl_compare.py --gams-local --skip-python --ifsub k --phase <today's phases> altertax` | `tests/fixtures/gtap7/<ds>/gams_*.{nl,row,col}`, `gtap7_altertax/<ds>/gams_altertax.nl` |
| nlp altertax | 3x3, 3x4, 5x5, 10x7 | 0, 1 | `gams_nlp_altertax_ifsub<k>` | `regen_gams_reference.py --mode altertax --form nlp` | `gtap7_nlp/<ds>_altertax_ifsub<k>.gdx` |
| nlp gtap | 3x3, 5x5, 10x7 | 0, 1 | `gams_nlp_gtap_ifsub<k>` | `regen_gams_reference.py --mode gtap --form nlp` | `gtap7_nlp/<ds>_pure_ifsub<k>.gdx` |
| mcp gtap | 3x3 | 1 | `gams_mcp_gtap_ifsub1` | `regen_gams_reference.py --mode gtap --form mcp` | `gtap7/gtap7_3x3/{out_gtap_shock_ifsub1.gdx, comp_gtap7_3x3_gtap_shock_ifsub1.gms}` |
| mcp altertax | 3x3 | 1 | `gams_mcp_altertax_ifsub1` | `regen_gams_reference.py --mode altertax --form mcp` | `gtap7_altertax/gtap7_3x3/out_altertax_ifsub1.gdx` |

"Today's phases": `base check shock` for 3x3, 5x5, 10x7; `base shock` for 3x4, 15x10 (adding `check` to those two is phase 7). The two altertax GDX per dataset are two methods, kept apart: `gtap7_altertax/` is the MCP/PATH reference (`ifMCP=1`, used by the MCP gate) and `gtap7_nlp/*_altertax_*` the NLP reference (`ifMCP=0`, used by the NLP gate); 185 symbols differ by ~0.3 % at the shock (`xp[USA,a_Food,shock]` 1.5642 vs 1.5599, measured 2026-10-29). Each is regenerated with its own method.

## Review Focus

- A job that fails (GAMS error, model status ∉ {1, 2}, missing output) must leave no folder and no `source.yaml` → Task 3 `test_failed_job_leaves_no_folder`.
- A test that asks for `ifsub=0` where only `ifsub=1` exists must raise naming the folder, never fall back → Task 4 `test_nl_gate_asks_ifsub0` (the #23 failure mode).
- `--gams-local` must write the `ifSUB` it was given into the `.gms` (today the header hardcodes 0) → Task 2 `test_standalone_carries_ifsub`.
- A re-run after a crash must skip finished folders and redo unfinished ones → Task 3 `test_resume_skips_recorded`.
- The model move must not change a single `.gms` byte → Task 1 Step 3 (`git mv` only, `git diff -M100% --stat` shows renames only).

---

### Task 1: the GAMS model moves to `data/gtap7/gams/`

**Files:**
- Move (`git mv`): `src/equilibria/templates/reference/gtap/scripts/*` (19), `src/equilibria/templates/reference/gtap/gams/{altertax_config,altertax_postmodel}.gms`, `src/equilibria/templates/reference/gtap/tariff_sim_fixed.gms`, `scripts/cal_dump.gms`, `scripts/cal_dump_neos.gms` → `data/gtap7/gams/` (flat; no name collides — check before moving). The 6 GDX of `src/equilibria/templates/reference/gtap/data/` → `data/gtap7/9x10/` (they are 9x10 GAMS inputs).
- Create: `data/gtap7/gams/source.yaml` (`write_source.py --kind model --no-run --model gtap7 --source "GTAP Standard 7 GAMS model (van der Mensbrugghe), as committed in this repository; local edits are in git history"`); rewrite `data/gtap7/9x10/source.yaml` (`--kind data --no-run`, same `--source` as phase 1, the GDX now among its outputs).
- Modify: every `.gms` path in the consumers listed by `git grep -lE 'reference/gtap/(scripts|data|gams)|tariff_sim_fixed|cal_dump'` (≈20 files: builders' `COMP_GMS`/`SCRIPTS_DIR` constants, `parity_adapter.py:375-376`, comments citing `model.gms:53`).
- Test: `tests/data/test_gams_model_present.py`.

**Interfaces:**
- Produces: `GAMS_MODEL_DIR = DATA_ROOT / "gtap7" / "gams"` in `src/equilibria/sources.py`; every builder reads `comp.gms` and its includes from there.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("name", ["comp.gms", "model.gms", "cal.gms", "getData.gms", "altertax_config.gms", "cal_dump.gms"])
def test_gams_model_file_present(name):
    assert (GAMS_MODEL_DIR / name).is_file()

def test_model_folder_is_kind_model():
    assert load_source(GAMS_MODEL_DIR).kind == "model"
```

- [ ] **Step 2:** `uv run pytest tests/data/test_gams_model_present.py -q` → FAIL (`ImportError: GAMS_MODEL_DIR`).
- [ ] **Step 3:** Move, write both `source.yaml`, add the constant, rewrite consumers. `git diff -M100% --cached --stat` lists only renames for the `.gms`/`.gdx`.
- [ ] **Step 4:** Same test → PASS; `git grep -nE 'reference/gtap/(scripts|data|gams)|templates/reference/gtap/tariff' -- ':!specs/data' ':!docs/adr/0010*'` → no output; `uv run pytest tests/data tests/templates/gtap/test_ifsub_equivalence.py -q` → pass as on `origin/main`.
- [ ] **Step 5:** A builder still finds the model: `uv run python scripts/gtap/build_gtap7_pure_local_bundle.py --dataset gtap7_3x3 --ifsub 1 --out-dir $SCRATCH/b` → writes the bundle.
- [ ] **Step 6:** Commit `refactor(data): the GAMS GTAP model moves to data/gtap7/gams`.

### Task 2: single-job entry points write into `--out`

**Files:**
- Modify: `scripts/gtap/nl_compare.py` (`_build_standalone_gms` gains `ifsub: int`; new `--gams-local` runs local CONVERT with `convert.opt` = `AmplNL ampl.nl` in `--out-dir` instead of `_submit_and_fetch_nl`, writing `gams_<phase>.{nl,row,col}` and `comp_<phase>.gms`; `--ifsub` help drops "pure-gtap only").
- Create: `scripts/gtap/regen_gams_reference.py`. Delete: `scripts/gtap/regen_altertax_reference.py` (its "install into fixtures" job is gone; its subsidy check moves).
- Test: `tests/scripts/test_nl_compare_local.py`, `tests/scripts/test_regen_gams_reference.py`.

**Interfaces:**
- Produces: `nl_compare.py --dataset D --phase P... --ifsub K --gams-local --skip-python --out-dir DIR` (exit 0 only if every phase wrote its `.nl`); `regen_gams_reference.py --dataset D --mode {gtap,altertax} --form {nlp,mcp} --ifsub K --out DIR` → `DIR/out.gdx`, `DIR/model.gms` (+`path.opt`), exit ≠ 0 on any model status ∉ {1, 2} or missing GDX, leaving `DIR` without `out.gdx`. `--form nlp` = `ifMCP=0` + `option nlp=ipopt` (today's `regen_altertax_reference._solve(solver="nlp")` transform); `--form mcp` = the bundle as built. `--mode altertax` keeps the subsidy check (`ytax(ft)`, `ytax(fs)` nonzero).

- [ ] **Step 1: Write the failing tests**

```python
def test_standalone_carries_ifsub():
    for k in (0, 1):
        assert f"$setGlobal ifSUB     {k}" in nl_compare._build_standalone_gms("base", agg_gdx=AGG_3X3, ifsub=k)

@needs_gams
def test_gams_local_writes_nl(tmp_path):            # gtap7_3x3, phase base, ifsub 1
    run(["--dataset", "gtap7_3x3", "--phase", "base", "--ifsub", "1", "--gams-local", "--skip-python", "--out-dir", tmp_path])
    assert {p.name for p in tmp_path.iterdir()} >= {"gams_base.nl", "gams_base.row", "gams_base.col", "comp_base.gms"}

@needs_gams
def test_mcp_over_demo_cap_fails_clean(tmp_path):   # gtap7_3x3 gtap mcp ifsub 0 → Licensing Problem
    assert run_regen("gtap7_3x3", "gtap", "mcp", 0, tmp_path) != 0
    assert not (tmp_path / "out.gdx").exists()

@needs_gams
def test_nlp_altertax_writes_gdx(tmp_path):         # gtap7_3x3 altertax nlp ifsub 1
    assert run_regen("gtap7_3x3", "altertax", "nlp", 1, tmp_path) == 0
    assert (tmp_path / "out.gdx").is_file() and (tmp_path / "model.gms").is_file()
```

- [ ] **Step 2:** `uv run pytest tests/scripts/test_nl_compare_local.py tests/scripts/test_regen_gams_reference.py -q` → FAIL.
- [ ] **Step 3:** Implement. `needs_gams` = skip when the GAMS binary is absent (CI).
- [ ] **Step 4:** Same command → PASS (GAMS 53 on `PATH`).
- [ ] **Step 5:** Commit `feat(gtap): nl_compare and regen_gams_reference solve one GAMS job into a folder`.

### Task 3: the driver

**Files:**
- Create: `scripts/gtap/regenerate_gams.py`. Test: `tests/scripts/test_regenerate_gams.py` (subprocess stubbed, trees under `tmp_path`, `--data-root`).

**Interfaces:**
- Consumes: Task 2 entry points; `write_source.py` (`--kind result --tool gams --tool-version --solver {CONVERT,IPOPT,PATH} --variant form=… --variant mode=… --variant ifsub=… --phases … --model-file data/gtap7/gams/comp.gms --data-dir <dataset_dir>`).
- Produces: `regenerate_gams.py --tool-version V [--datasets …] [--only <folder name>] [--dry-run] [--data-root]`; `jobs() -> list[Job]` with the 26 jobs of the table, `Job.folder == dataset_dir(ds) / f"gams_{form}_{mode}_ifsub{k}"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_jobs_are_the_table():
    js = jobs()
    assert len(js) == 26
    assert sum(j.form == "nl" for j in js) == 10 and sum(j.form == "mcp" for j in js) == 2
    assert {j.folder.name for j in js if j.dataset == "gtap7_3x4"} == {f"gams_{f}_altertax_ifsub{k}" for f in ("nl", "nlp") for k in (0, 1)}  # no mcp: above the demo cap

def test_resume_skips_recorded(tmp_path): ...       # folder with source.yaml → not run
def test_failed_job_leaves_no_folder(tmp_path): ... # stub exits 1 → folder absent, driver exit ≠ 0, next jobs still run
def test_wrong_tool_version_stops(tmp_path): ...    # stub `gams` prints 53.6.0, --tool-version 53.5.0 → exit ≠ 0 naming 53.6.0
```

- [ ] **Step 2:** `uv run pytest tests/scripts/test_regenerate_gams.py -q` → FAIL.
- [ ] **Step 3:** Implement, mirroring `scripts/windows/regenerate_gempack.py` (skip recorded, delete a failed folder, run the rest, exit ≠ 0 if any failed).
- [ ] **Step 4:** Same command → PASS; `--dry-run` prints 26 lines.
- [ ] **Step 5:** Commit `feat(gtap): regenerate_gams.py records the local GAMS results into data/`.

### Task 4: run, diff against the old fixtures, rewire, delete

**Files:**
- Create: the 26 result folders (by running Task 3); `data/gdx_tests/` (the design's layout: a top-level folder of `data/`, next to `gtap7/`) with copies of the three GDX the reader test uses as format specimens (`tests/fixtures/gtap7/gtap7_3x3/out_gtap_shock_ifsub{0,1}.gdx`, `tests/fixtures/gtap7_nlp/gtap7_3x3_pure_ifsub0.gdx`) and their `.gdxdump.json.gz` oracles, `kind: data`, `--source "format specimen, copied from <old path> at <commit>"`. A reader specimen need not be a current result, so it is copied, not regenerated; the originals go with the phase that replaces them (2a or 2b).
- Modify: every consumer of the replaced files (`git grep -lE 'fixtures/gtap7_(nlp|altertax)|gams_(base|check|shock|altertax)\.nl|out_altertax_ifsub|_pure_ifsub|gtap7_3x3/out_gtap_shock_ifsub1'`): the `.nl`, NLP and MCP gate tests, `coverage_matrix.py`, `cascade_config.py`, `bench_nlp_timing.py`, `convert_gams.py`, `diff_altertax.py`, `tests/babel/gdx/test_reader_vs_gdxdump_snapshot.py` and the rest the grep lists → `reference(...)` (or `data/gdx_tests/` for the reader test). `scripts/gtap/check_parity_gates_stamp.py` `INPUT_TREES` unchanged (both fixture trees still hold 2b files).
- Delete: the replaced files of the Jobs table (and `tests/fixtures/gtap7_nlp/` once empty; `tests/fixtures/gtap7_altertax/` keeps its MCP GDX for 2b).
- Test: `tests/templates/gtap/test_gtap7_nl_parity.py::test_nl_gate_asks_ifsub0`.

- [ ] **Step 1: Write the failing test**

```python
def test_nl_gate_asks_ifsub0():
    # the gate builds Python with ifSUB=0; it must read the GAMS .nl recorded with ifSUB=0
    for ds in nl_datasets():
        folder = reference(ds, "gams", form="nl", mode="altertax", ifsub=0)
        assert load_source(folder).variant["ifsub"] == 0
```

and `nl_datasets()` lists datasets from `iter_sources()` (form `nl`, ifsub 0) instead of `tests/fixtures/gtap7/`.
- [ ] **Step 2:** Run → FAIL (`SourceError: no result for gtap7_3x3 gams …`).
- [ ] **Step 3:** `uv run python scripts/gtap/regenerate_gams.py --tool-version 53.5.0` (GAMS 53 on `PATH`; a failed job → stop and report it before going on). Then `uv run pytest tests/data -q` → green (`check_tree`: every file has a source, every hash matches).
- [ ] **Step 4: Diff every replaced file** (results into the PR as a table: file, identical / differs, what differs). GDX: `gdxdiff OLD NEW RelEps=1e-9`. `.nl`: `cmp`, and when different, `diff_nl_models(parse_nl(OLD), parse_nl(NEW))` counts (rows/cols only in one, coefficient diffs). Expected: ifSUB=1 `.nl` ≈ the old fixtures (#23 says they look like ifSUB=1); ifSUB=0 lacks nothing of `pm pefob pmcif pwmg xwmg xmgm pfa pfy pp`. Any other difference: stop and explain before Step 5.
- [ ] **Step 5:** Rewire consumers; delete the replaced files; `git grep` of Files/Modify → only spec/ADR hits.
- [ ] **Step 6:** `uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py tests/templates/gtap/test_gtap7_nlp_parity.py tests/templates/gtap/test_gtap7_mcp_parity.py tests/babel/gdx -q`. A gate that changes state with the regenerated reference is reported to the maintainer with the Step 4 diff; no tolerance or floor is edited in this phase.
- [ ] **Step 7:** Commit `refactor(data): GAMS results regenerated locally into data/, consumers ask reference()` (results and rewiring in one commit: between them the suite is red).

### Task 5: gates, map and phase close

**Files:**
- Modify: `data/README.md` (the `gams_<form>_<mode>_ifsub<k>` folders, `regenerate_gams.py`, the local/NEOS split), the stamp file.

- [ ] **Step 1:** `uv run python scripts/gtap/run_parity_gates.py` → green; figures identical to `origin/main` except any gate already reported in Task 4 Step 6. Inspect regenerated matrix docs before committing.
- [ ] **Step 2:** Full suite `uv run pytest -q -rA > after.txt` vs `before.txt`: by name, vanished names and skip reasons; only Tasks 1–4 tests are new; the 3 PEP Excel failures of `main` unchanged; restore `Results.gdx`.
- [ ] **Step 3:** Sphinx `-W` build; `uv run pytest tests/docs -q`; `uv run python scripts/dev/check_layout.py` → 0; `prek run --all-files` → Passed.
- [ ] **Step 4:** Commit `docs(data): map lists the local GAMS results; gate stamp follows them`.
- [ ] **Step 5:** One final reviewer (maintainer-approved), then push and PR only with the maintainer's permission.
