# One data folder — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the input datasets into `data/<model>/<dataset>/` with a `kind: data` `source.yaml` each, make every caller find them through `equilibria.sources`, ship nus333/9x10 in the wheel from `data/`, and delete the duplicate under `src/`.

**Architecture:** `equilibria.sources` gains `dataset_dir()` / `has_dataset()`, the only way code finds a dataset folder (the `kind: data` twin of `reference()`). Literal paths in prose, JSON metadata and shell are rewritten mechanically by a fixed old→new prefix table. `load_bundled` looks first for the copy hatch puts inside the wheel, then for the repository `data/`.

**Tech Stack:** Python ≥3.10, PyYAML, pytest, hatchling (`force-include`), `uv`, prek, GitHub Actions.

**Spec:** `specs/data/changes/one-data-folder/design.md` (phase 1 in "Phases"; "Packaging"; "Writing a `source.yaml`"), `docs/adr/0010-one-data-folder.md`.

## Global Constraints

- No number changes (ADR-0008): every moved file keeps its bytes; `git mv` so history follows.
- Folder names are the names the code uses (design: "a name found in code is found on disk as is"). The code calls the bundled dataset `"9x10"`, so it lives in `data/gtap7/9x10/` (the design's layout line `gtap7_9x10/` is read as this; the Packaging section already says `data/gtap7/9x10`).
- Input data is not regenerated: its `source.yaml` records origin and sha256, `status: verified`, written by `scripts/dev/write_source.py --kind data --no-run` (nobody types a hash).
- Out of scope, stays where it is: the GDX files in `src/equilibria/templates/reference/gtap/data/` (`9x10Dat.gdx`, `basedata-9x10.gdx`, …: GAMS inputs, phase 2), the `comp*` exclude in `pyproject.toml` (goes with the comp files, phase 2), PEP (phase 5).
- Suite before and after identical by test name except the tests this phase adds; the five parity gates give identical figures.
- Commits in English, `type(scope): description`, ending with the session's `Co-Authored-By` line.

### Moves (exact)

| From | To | `--model` | `--source` |
|---|---|---|---|
| `datasets/gtap7_{3x3,3x4,5x5,10x7,15x10,20x41}/` | `data/gtap7/gtap7_<ds>/` | `gtap7` | `"GTAP aggregation committed in 4ad8e06 (2026-06-10); upstream database and aggregation tool not recorded"` |
| `datasets/gtap6_{3x3,3x4,5x5,10x7,15x10,20x41}/` | `data/gtap6/gtap6_<ds>/` | `gtap6` | same text |
| `datasets/nus333/` (108 files) | `data/gtap7/nus333/` | `gtap7` | `"Burfisher, Introduction to Computable General Equilibrium Models, model package NUS333; byte-identical to the download (22f17d7)"` |
| `src/equilibria/templates/reference/gtap/data/9x10/` (3 files) | `data/gtap7/9x10/` | `gtap7` | `"GTAP Standard 7 GAMS distribution, inputs of convert.cmd (6bf7dd5)"` |
| `src/equilibria/templates/reference/gtap/data/nus333/` (4 files) | deleted (identical to `datasets/nus333`, md5 checked 2026-10-28) | | |

`--license` for every dataset: `"not stated by the source; to confirm"` (design: nus333 redistribution not stated). The maintainer may replace any `--source`/`--license` text before Task 2 is committed.

### Prefix rewrite table (Task 3)

Applied in this order (longest first), to every tracked file except `specs/data/**`, `docs/adr/0010-one-data-folder.md` and `data/README.md`:

| Old | New |
|---|---|
| `src/equilibria/templates/reference/gtap/data/9x10/` | `data/gtap7/9x10/` |
| `src/equilibria/templates/reference/gtap/data/nus333/` | `data/gtap7/nus333/` |
| `datasets/nus333` | `data/gtap7/nus333` |
| `datasets/gtap7_` | `data/gtap7/gtap7_` |
| `datasets/gtap6_` | `data/gtap6/gtap6_` |
| `datasets\nus333` | `data\gtap7\nus333` |
| `datasets\$ds` (ps1; `$ds` ranges over `gtap7_*` only — confirm at the loop near line 240) | `data\gtap7\$ds` |

## Review Focus

- A dataset name with no folder must raise and list the datasets that exist, never return a non-existent path → Task 1 `test_unknown_dataset_lists_known`.
- Two `kind: data` folders claiming the same dataset must raise and name both → Task 1 `test_two_folders_same_dataset_raise`.
- A result folder (`kind: result`) for a dataset must never be returned as its data folder → Task 1 `test_result_folder_is_not_data`.
- An installed wheel run from a directory with no repo must still load nus333 → Task 4 Step 6 (clean venv, run outside the repo).
- `EQUILIBRIA_NUS333_DIR` must keep overriding the default → Task 3 runs `tests/test_local_refs_nus333.py`.

---

### Task 1: `dataset_dir()` and `has_dataset()`

**Files:**
- Modify: `src/equilibria/sources.py` (after `reference`)
- Test: `tests/data/test_sources_dataset_dir.py` (trees built under `tmp_path` with `tests/data/_records.py` helpers)

**Interfaces:**
- Consumes: `iter_sources(root) -> Iterator[SourceRecord]`, `SourceRecord.{kind,dataset,path}`, `SourceError`, `DATA_ROOT`.
- Produces: `dataset_dir(dataset: str, *, root: Path = DATA_ROOT) -> Path` and `has_dataset(dataset: str, *, root: Path = DATA_ROOT) -> bool`. Only `kind == "data"` records count.

- [ ] **Step 1: Write the failing tests**

```python
def test_finds_the_data_folder(tmp_path):        # data/gtap7/gtap7_3x3 with kind: data
    assert dataset_dir("gtap7_3x3", root=tmp_path) == tmp_path / "gtap7" / "gtap7_3x3"
    assert has_dataset("gtap7_3x3", root=tmp_path)

def test_unknown_dataset_lists_known(tmp_path):  # tree holds gtap7_3x3 only
    with pytest.raises(SourceError, match=r"gtap7_9x9.*known: \['gtap7_3x3'\]"):
        dataset_dir("gtap7_9x9", root=tmp_path)
    assert not has_dataset("gtap7_9x9", root=tmp_path)

def test_two_folders_same_dataset_raise(tmp_path):  # gtap6/x and gtap7/x both dataset "x"
    with pytest.raises(SourceError) as e:
        dataset_dir("x", root=tmp_path)
    assert "gtap6" in str(e.value) and "gtap7" in str(e.value)

def test_result_folder_is_not_data(tmp_path):    # only a kind: result folder for gtap7_3x3
    assert not has_dataset("gtap7_3x3", root=tmp_path)
```

- [ ] **Step 2:** `uv run pytest tests/data/test_sources_dataset_dir.py -q` → FAIL (`ImportError: dataset_dir`).
- [ ] **Step 3:** Implement both in `src/equilibria/sources.py`; `has_dataset` counts matches, it does not catch `dataset_dir`'s error (two matches → `has_dataset` raises too).
- [ ] **Step 4:** Same command → PASS.
- [ ] **Step 5:** Commit `feat(sources): dataset_dir() finds a dataset folder by name`.

### Task 2: move the datasets and record where they come from

**Files:**
- Move: per the Moves table (`git mv`); delete the `src/` nus333 copy (`git rm -r`).
- Create: one `source.yaml` per moved folder (14 dataset folders: 6 gtap7, 6 gtap6, nus333, 9x10). `9x10` has no other copy, so it moves; only the nus333 duplicate is deleted.
- Modify: `.gitignore:404-413` (`datasets/*/X` → `data/gtap*/*/X`, same file list), `scripts/dev/check_layout.py:48` (drop `"datasets"` from `ROOT_DIRS_TEMPORARY`, and any `ALLOWED_EXACT` entry under `datasets/`).
- Test: `tests/data/test_data_tree.py` (exists: real tree must pass `check_tree`), new `tests/data/test_datasets_present.py`.

**Interfaces:**
- Consumes: `write_source.py --kind data --no-run --folder F --model M --dataset D --source S --license L`.
- Produces: the 14 folders above, each `has_dataset(<name>)`; Task 3 relies on the names in the Moves table.

- [ ] **Step 1: Write the failing test**

```python
GTAP = [f"gtap{v}_{s}" for v in (6, 7) for s in ("3x3", "3x4", "5x5", "10x7", "15x10", "20x41")]

@pytest.mark.parametrize("name", [*GTAP, "nus333", "9x10"])
def test_dataset_has_its_har_files(name):
    d = dataset_dir(name)
    for f in ("basedata.har", "sets.har", "default.prm"):
        assert (d / f).is_file(), f"{name}: {f} missing in {d}"
```

- [ ] **Step 2:** `uv run pytest tests/data -q` → FAIL (`SourceError: no data folder for dataset 'gtap6_3x3'`).
- [ ] **Step 3:** Before deleting, check `md5` of the 4 files of `src/.../gtap/data/nus333/` equals `datasets/nus333/` (stop if not). `git mv` per the table; `git rm -r` the copy; run `write_source.py` once per folder; edit `.gitignore` and `check_layout.py`.
- [ ] **Step 4:** `uv run pytest tests/data -q` → PASS; `python scripts/dev/check_layout.py` → 0; `ls datasets` → no such directory.
- [ ] **Step 5:** Commit `refactor(data): datasets move into data/<model>/<dataset> with a source.yaml`. Between this commit and Task 3's the rest of the suite is red (callers still say `datasets/`); the PR is merged as a whole.

### Task 3: every caller asks `equilibria.sources`

**Files:**
- Modify (Python, by hand): the 16 `DATASETS_DIR = ROOT / "datasets"` modules (`git grep -n 'DATASETS_DIR\s*='`), every `ROOT / "datasets" / X` and `REPO_ROOT / "datasets"` (`git grep -n '"datasets"'`), `src/equilibria/_local_refs.py:75` (default of `nus333_dir`), `scripts/gtap/_parity_datasets.py`.
- Modify (mechanical, by the prefix table): the rest of the 120 files `git grep -lE 'datasets[/\\]|gtap/data/(9x10|nus333)/'` lists (docs, JSON `"path"` metadata in `tests/babel/har/golden/`, `.ps1`, readmes).
- Modify: `.github/workflows/windows-20x41-diagnostic.yml:68` → `[ ! -d "data/${DATASET%%_*}/$DATASET" ]`.

**Interfaces:**
- Consumes: `dataset_dir`, `has_dataset` (Task 1); folder names (Task 2).
- Produces: no `datasets` path anywhere outside the excluded spec files.

- [ ] **Step 1:** Replace in Python: `DATASETS_DIR / X` and `ROOT / "datasets" / X` → `dataset_dir(X)`; `(… / X / "basedata.har").exists()` used as a skip guard → `has_dataset(X)`; drop the `DATASETS_DIR` definitions. `nus333_dir()` default → `str(dataset_dir("nus333"))` (env var still first).
- [ ] **Step 2:** Apply the prefix table with a throwaway script in the scratchpad (not committed).
- [ ] **Step 3:** `git grep -nE 'datasets[/\\]|"datasets"|DATASETS_DIR|gtap/data/(9x10|nus333)/' -- ':!specs/data' ':!docs/adr/0010-one-data-folder.md' ':!data/README.md'` → no output.
- [ ] **Step 4:** `uv run pytest tests/test_local_refs_nus333.py tests/babel/har tests/templates/gtap tests/blocks/gtap tests/compare tests/scripts -q` → same pass/skip as on `origin/main` for these paths (diff by name).
- [ ] **Step 5:** Commit `refactor(data): callers find datasets through equilibria.sources`.

### Task 4: packaging from `data/`

**Files:**
- Modify: `src/equilibria/datasets.py:24-28` (`_GTAP_DATASETS` lookup), `pyproject.toml:119` (`[tool.hatch.build.targets.wheel.force-include]`).
- Test: `tests/test_datasets_bundled.py`.

**Interfaces:**
- Consumes: `dataset_dir` (Task 1).
- Produces: `dataset_path("gtap", name)` = `Path(equilibria.__file__).parent / "_bundled" / "gtap" / name` when that folder exists (installed wheel), else `dataset_dir(name)`. `load_bundled`, `list_bundled`, `dataset_path` keep their signatures; `list_bundled("gtap") == ["9x10", "nus333"]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_checkout_reads_data_folder():
    assert dataset_path("gtap", "nus333") == dataset_dir("nus333")

def test_bundled_copy_wins(tmp_path, monkeypatch):  # fake package dir with _bundled/gtap/nus333
    monkeypatch.setattr(datasets, "_PACKAGE_DIR", tmp_path)
    (tmp_path / "_bundled" / "gtap" / "nus333").mkdir(parents=True)
    assert dataset_path("gtap", "nus333") == tmp_path / "_bundled" / "gtap" / "nus333"

def test_load_bundled_nus333_same_numbers():
    p = load_bundled("gtap", "nus333")
    # pin 3 values read on origin/main before the move (record them in Step 2)
```

- [ ] **Step 2:** On `origin/main` (other worktree), print 3 scalars of `load_bundled("gtap","nus333")` and of `"9x10"` (one `vfm`, one `vxsb`, one elasticity) and write them into the test with `==` (bytes are unchanged, so equality is exact). Run → FAIL (`_PACKAGE_DIR` missing).
- [ ] **Step 3:** Implement the lookup with module constant `_PACKAGE_DIR = Path(__file__).parent`. `force-include` maps each file: `data/gtap7/9x10/{basedata.har,sets.har,default.prm}` and `data/gtap7/nus333/{basedata.har,sets.har,default.prm,baserate.har}` → `equilibria/_bundled/gtap/<name>/<file>` (the 104 Burfisher experiment files stay out: the wheel ships what it ships today).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** `uv build` then `unzip -l dist/*.whl | grep -E '_bundled|reference/gtap/data/(9x10|nus333)/|comp'` → the 7 `_bundled` files, nothing else.
- [ ] **Step 6:** Clean venv: `uv venv $TMPDIR/whl && uv pip install --python $TMPDIR/whl dist/*.whl` then from `$TMPDIR`: `$TMPDIR/whl/bin/python -c 'from equilibria import load_bundled; print(load_bundled("gtap","nus333"))'` → loads, path under `site-packages/equilibria/_bundled`.
- [ ] **Step 7:** Commit `build(data): the wheel ships nus333 and 9x10 from data/`.

### Task 5: gates, map and phase close

**Files:**
- Modify: `scripts/gtap/check_parity_gates_stamp.py:28` (`INPUT_TREES` += `"data/gtap6"`, `"data/gtap7"`: gate results depend on them), the stamp file it writes, `data/README.md` (the map lists the 14 dataset folders and how to add one with `write_source.py --kind data`).

- [ ] **Step 1:** `uv run python scripts/gtap/run_parity_gates.py` → same figures as on `origin/main`; write the new stamp.
- [ ] **Step 2:** Full suite `uv run pytest -q -rA > after.txt` vs `before.txt` from `origin/main`, both with GAMS 53 on `PATH`: only the tests of Tasks 1, 2 and 4 are new; none changed state. The 3 PEP Excel failures of `main` (`test_pep2_gams_excel_parity` ×2, `test_pep2_python_excel_parity`) stay as they are; restore `src/equilibria/templates/reference/pep2/scripts/Results.gdx` after the run (those tests overwrite it).
- [ ] **Step 3:** Sphinx `-W` build; `uv run pytest tests/docs -q`; `python scripts/dev/check_layout.py` → 0; `prek run --all-files` → Passed.
- [ ] **Step 4:** Commit `docs(data): map lists the datasets; gate stamp covers data/`.
- [ ] **Step 5:** Push and PR only with the maintainer's permission.
