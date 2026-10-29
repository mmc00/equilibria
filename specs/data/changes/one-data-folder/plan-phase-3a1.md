# One data folder — Phase 3a-1 Implementation Plan (GEMPACK matrix on Windows)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One command on the Windows machine regenerates the GEMPACK linearization-study matrix (6 datasets × 11 configurations) straight into `data/gtap7/<ds>/gempack_<tag>/`, each folder with a `source.yaml` written by `write_source.py`.

**Architecture:** `run_gempack_matrix.py` gains a single-job mode (`--out DIR`) that solves one (dataset, shock, steps, closure) and leaves its results in `DIR` under fixed names. A new driver, `scripts/windows/regenerate_gempack.py`, lists the 66 jobs and runs each through `write_source.py`, which times the run and records it. Nothing old is deleted or rewired here: comparing against `tests/fixtures/gtap7_gempack/` and rewiring consumers is phase 3b, on the Mac.

**Tech Stack:** Python ≥3.10, pytest, `uv run` on Windows, RunGTAP `gtapv7.exe` and GEMPACK `sltoht.exe` (Windows only).

**Spec:** `specs/data/changes/one-data-folder/design.md` (phase 3 in "Phases", "`source.yaml`", "When a regenerated file differs from the old one").

## Global Constraints

- Phase 3 is split (ruling, 2026-10-29): 3a-1 this matrix; 3a-2 the `corridas_paper.ps1` steps (Burfisher sltoht, Horridge, six-variable levels, Burfisher Gragg); 3b (Mac) diff against the old files, rewire consumers, delete `tests/fixtures/gtap7_gempack` and `runs/`.
- The job set reproduces what is committed today, nothing more: for each of `gtap7_{3x3,3x4,5x5,10x7,15x10,20x41}` the 10 grid configurations of `_grid_configs()` with closure capFlex (RDLT=1; the July fixtures used the dataset's default `.prm`, which is RDLT=1) plus `tm10_s8-16-32` with capFix (RDLT=0). 66 jobs.
- Folder: `data/gtap7/<ds>/gempack_<config_tag(shock, steps, rordelta)>`, e.g. `gempack_tm10_s8-16-32_capflex`, `gempack_tm0p1_s8-16-32_capflex`, `gempack_tm10_s8-16-32_capfix`.
- Files in each folder (fixed names, so a test asks for `reference(...) / "updated.har"`): `updated.har`, `decomp.har`, `sl4dump.har`, `run.cmf`, `solve_console.txt`, `sltoht_console.txt`. Inputs (`basedata.har`, `sets.har`, the `.prm`) are not copied in: they are recorded by hash in `generated_from`.
- `source.yaml` variant keys: `shock` (number, percent: `10`, `0.1`), `steps` (string, `"8-16-32"`), `closure` (`capFlex` | `capFix`, equilibria's spelling). `tool: gempack`, `tool_version` from `--tool-version` (the driver's argument, default `"11.3"`), `phases: [shock]`, `solver: gtapv7`, `--data-dir data/gtap7/<ds>`.
- A failed job leaves no folder behind (its files would otherwise count as files of the dataset folder and break the next job's input hash and `check_tree`).
- Commit messages in English, `type(scope): description`, ending with the session's `Co-Authored-By` line.

## Review Focus

- A solve that does not report "completed without error" must exit non-zero, so `write_source.py` writes no `source.yaml` → Task 1 `test_failed_solve_exits_nonzero`.
- A failed job's folder must be removed before the next job hashes `data/gtap7/<ds>` → Task 2 `test_failed_job_folder_is_removed`.
- Rerunning the driver must skip folders that already have a `source.yaml` (a Windows run that dies at job 40 resumes at 41) → Task 2 `test_resume_skips_recorded_folders`.
- `--out` with more than one dataset, config or closure is ambiguous and must be refused → Task 1 `test_out_needs_exactly_one_job`.
- Paths on Windows: every path the driver passes is absolute (`write_source.py` runs the command with `cwd=folder`) → Task 2 `test_command_paths_are_absolute`.

---

### Task 1: single-job mode in `run_gempack_matrix.py`

**Files:**
- Modify: `scripts/gtap/run_gempack_matrix.py` (`main`, new `collect_job`)
- Test: `tests/scripts/test_run_gempack_matrix_out.py`

**Interfaces:**
- Consumes: `prepare(name, shock_pct, steps, rordelta) -> (run_dir, tag)`, `solve(run_dir, gtapv7, tag) -> (ok, res)`, `convert_sl4(run_dir, sltoht, tag) -> Path | None`, `config_tag`.
- Produces: CLI `run_gempack_matrix.py --datasets DS --shock-pct X --steps "S" --rordelta R --out DIR [--gtapv7 P --sltoht P]` → exit 0 and `DIR/{updated.har,decomp.har,sl4dump.har,run.cmf,solve_console.txt,sltoht_console.txt}`; exit 1 on a failed solve, 2 on bad arguments. `collect_job(run_dir: Path, tag: str, out: Path) -> list[str]` returns the names written.

- [ ] **Step 1: Write the failing tests** — use a fake `gtapv7` (a Python script made executable in `tmp_path` that writes `updated_<tag>.har`, `decomp.har`, `<tag>.sl4` and prints "completed without error") and a fake `sltoht` that writes `sl4dump_<tag>.har`; `RUN_ROOT` monkeypatched into `tmp_path`.

```python
def test_out_collects_fixed_names(tmp_path): ...
    # run main([... "--out", str(out)]) with the fakes → 0
    assert sorted(p.name for p in out.iterdir()) == [
        "decomp.har", "run.cmf", "sl4dump.har", "sltoht_console.txt",
        "solve_console.txt", "updated.har"]

def test_failed_solve_exits_nonzero(tmp_path): ...
    # fake gtapv7 prints "error" and writes nothing → main(...) == 1, out has no updated.har

def test_out_needs_exactly_one_job(tmp_path):
    assert main(["--datasets", "gtap7_3x3", "gtap7_3x4", "--out", str(tmp_path / "o")]) == 2
    assert main(["--grid", "--datasets", "gtap7_3x3", "--out", str(tmp_path / "o")]) == 2
    assert main(["--datasets", "gtap7_3x3", "--out", str(tmp_path / "o")]) == 2  # no --rordelta: two closures
```

- [ ] **Step 2:** `uv run pytest tests/scripts/test_run_gempack_matrix_out.py -q` → FAIL (`--out` unknown; `main` takes no argv).
- [ ] **Step 3:** `main(argv: list[str] | None = None) -> int`; with `--out`, validate one dataset × one config × one `--rordelta`, then prepare → clean stale → solve → on success `collect_job` (copy `updated_<tag>.har`→`updated.har`, `decomp.har`, `sl4dump_<tag>.har`→`sl4dump.har`, `<tag>.cmf`→`run.cmf`, the two consoles), exit 0; on failure exit 1. Without `--out`, behavior unchanged (fixtures dir, all jobs).
- [ ] **Step 4:** Same command → PASS.
- [ ] **Step 5:** Commit `feat(gempack): run_gempack_matrix --out solves one job into a folder`.

### Task 2: the driver `scripts/windows/regenerate_gempack.py`

**Files:**
- Create: `scripts/windows/regenerate_gempack.py`
- Test: `tests/scripts/test_regenerate_gempack.py`

**Interfaces:**
- Consumes: `config_tag`, `_grid_configs` (Task 1's module), `write_source.py` CLI.
- Produces: `jobs(datasets: list[str]) -> list[Job]` with `Job(dataset: str, shock: float, steps: str, closure: str, rordelta: int, folder: Path)`; `job_command(job, *, gtapv7: Path, sltoht: Path, tool_version: str) -> list[str]` (the full `write_source.py ... -- run_gempack_matrix.py ... --out <folder>` argv); `main(argv) -> int` with `--dry-run`, `--datasets`, `--only TAG` (repeatable, filters by `config_tag`), `--gtapv7`, `--sltoht`, `--tool-version`; exit 0 when every job is recorded, 1 otherwise; prints one line per job and a summary.

- [ ] **Step 1: Write the failing tests**

```python
def test_sixty_six_jobs():
    js = jobs(DATASETS)
    assert len(js) == 66
    assert sum(j.closure == "capFix" for j in js) == 6
    assert {j.folder.name for j in js if j.dataset == "gtap7_3x3"} >= {
        "gempack_tm10_s8-16-32_capflex", "gempack_tm0p1_s8-16-32_capflex",
        "gempack_tm10_s64_capflex", "gempack_tm10_s8-16-32_capfix"}

def test_command_records_the_variant():
    cmd = job_command(job_3x3_tm10_capfix, gtapv7=G, sltoht=S, tool_version="11.3")
    assert ["--variant", "shock=10", "--variant", "steps=8-16-32",
            "--variant", "closure=capFix"] == cmd[i:i+6]  # contiguous, in this order
    assert "--out" in cmd and cmd[cmd.index("--out") + 1] == str(job.folder)

def test_command_paths_are_absolute(): every path-like argument in job_command(...) is absolute
def test_resume_skips_recorded_folders(tmp_path): a job folder with source.yaml is not run (fake runner records calls)
def test_failed_job_folder_is_removed(tmp_path): fake runner returns 1 and leaves files → folder gone, main(...) == 1
def test_only_filters_by_tag(): main(["--dry-run", "--datasets", "gtap7_3x3", "--only", "tm10_s8-16-32_capflex"]) runs exactly 1 job
```

- [ ] **Step 2:** `uv run pytest tests/scripts/test_regenerate_gempack.py -q` → FAIL (module missing).
- [ ] **Step 3:** Implement; the runner is `subprocess.run(cmd)` behind a module-level function the tests replace (`run_job(cmd: list[str]) -> int`). `--dry-run` prints the 66 commands and exits 0.
- [ ] **Step 4:** Same command → PASS; `uv run python scripts/windows/regenerate_gempack.py --dry-run | tail -3` shows 66 jobs.
- [ ] **Step 5:** Commit `feat(gempack): regenerate_gempack.py records the GEMPACK matrix into data/`.

### Task 3: Windows instructions and a 1-job smoke run

**Files:**
- Modify: `scripts/windows/README.md` (new section "Phase 3a-1: GEMPACK matrix into data/").

- [ ] **Step 1:** Write the section: pull the branch; `uv sync --frozen`; smoke run `uv run python scripts/windows/regenerate_gempack.py --datasets gtap7_3x3 --only tm10_s8-16-32_capflex` then check that `data/gtap7/gtap7_3x3/gempack_tm10_s8-16-32_capflex/source.yaml` exists with `status: verified`; then the full run; then `git add data/gtap7 && git commit -m "data(gempack): regenerate the GEMPACK matrix on Windows" && git push`.
- [ ] **Step 2:** `uv run pytest tests/scripts -q` → PASS; `python scripts/dev/check_layout.py` → 0; `prek run --all-files` → Passed.
- [ ] **Step 3:** Commit `docs(windows): how to regenerate the GEMPACK matrix into data/`; push the branch only with the maintainer's permission, so the Windows clone can pull it.
