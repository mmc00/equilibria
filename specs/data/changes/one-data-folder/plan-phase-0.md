# One data folder — Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put in place everything the later phases rely on, without moving any file: the `source.yaml` schema and its checks, `reference()`, the script that writes a `source.yaml`, and the layout check (closed root, English lowercase names).

**Architecture:** One library module, `equilibria.sources`, owns the schema, the tree checks and `reference()`; tests and scripts only call it. `scripts/dev/write_source.py` runs a generating command and writes the record. `scripts/dev/check_layout.py` replaces `scripts/check_lowercase_filenames.py` and the inline "canonical scripts layout" CI step.

**Tech Stack:** Python ≥3.10, PyYAML (already a dependency), pytest, `uv run`, prek (pre-commit), GitHub Actions.

**Spec:** `specs/data/changes/one-data-folder/design.md` (and `proposal.md`, `specs/specs.md`, `docs/adr/0010-one-data-folder.md`).

## Global Constraints

- No file moves in this phase; `data/` holds only `README.md` at the end of it.
- Suite before and after identical except for the new tests (diff `pytest -q -rA` by name).
- Names of new files and folders: lowercase English.
- No host name or user name in any `source.yaml`.
- Commit messages in Spanish, `tipo(ámbito): descripción`, ending with the
  `Co-Authored-By` line the session gives.
- The check script lives at `scripts/dev/check_layout.py` (not `scripts/`: the
  repo forbids loose `scripts/*.py`); the design's `scripts/check_layout.py`
  means this path.

## Review Focus

- A `source.yaml` with a typo in a key (`varient:`) must be an error, not an ignored key → Task 1 `test_unknown_key_is_an_error`.
- Two folders matching the same request must raise and name both, never pick one → Task 2 `test_two_matches_raise_and_name_both`.
- A dataset in the request that has no folder at all must name what does exist → Task 2 `test_no_match_lists_candidates`.
- A command that fails must not leave a `source.yaml` that says `verified` → Task 4 `test_failed_command_writes_nothing`.
- A path with an uppercase-free Spanish word inside a longer English name (`decomp`, `design`) must NOT be flagged: tokens are whole words → Task 5 `test_word_inside_a_token_is_not_spanish`.

---

### Task 1: `source.yaml` schema

**Files:**
- Create: `src/equilibria/sources.py`
- Test: `tests/data/test_sources_schema.py` (+ `tests/data/__init__.py` if the tests tree needs it)

**Interfaces:**
- Produces:
  - `class SourceError(ValueError)`
  - `@dataclass(frozen=True) class SourceRecord` with fields: `kind: Literal["result","data","model"]`, `model: str`, `dataset: str | None`, `tool: str | None`, `variant: Mapping[str, object]` (default empty), `status: Literal["verified","pending"]`, `path: Path` (the folder), `raw: Mapping[str, object]`.
  - `load_source(path: Path) -> SourceRecord` — reads `<folder>/source.yaml`.
  - `sha256_file(path: Path) -> str`

Allowed top-level keys (anything else is `SourceError`): `kind model dataset tool tool_version variant phases generated_from solver command equilibria_commit exit outputs run status notes source license`. Required by kind: `result` → `model dataset tool variant generated_from command outputs run status`; `data` → `model dataset source license outputs status`; `model` → `model source outputs status`. `run` requires `date duration_s machine`; `machine` requires `os cpu cores ram_gb` and nothing else (no host, no user).

- [ ] **Step 1: Write the failing tests**

```python
def test_valid_result_loads(tmp_path): ...            # minimal valid result → SourceRecord(kind="result", variant={"ifsub": 0})
def test_unknown_key_is_an_error(tmp_path): ...       # "varient:" → SourceError mentioning "varient"
def test_missing_required_key_is_an_error(tmp_path): ...  # result without "command" → SourceError mentioning "command"
def test_machine_rejects_host_name(tmp_path): ...     # run.machine.host → SourceError
def test_status_must_be_verified_or_pending(tmp_path): ...
```

- [ ] **Step 2:** `uv run pytest tests/data/test_sources_schema.py -q` → FAIL (`ModuleNotFoundError: equilibria.sources`).
- [ ] **Step 3:** Implement `load_source`, `SourceRecord`, `SourceError`, `sha256_file` in `src/equilibria/sources.py` (YAML via `yaml.safe_load`; error messages name the file and the key).
- [ ] **Step 4:** Same command → PASS.
- [ ] **Step 5:** Commit `feat(sources): esquema de source.yaml (ADR-0010)`.

### Task 2: `reference()`

**Files:**
- Modify: `src/equilibria/sources.py`
- Test: `tests/data/test_sources_reference.py`

**Interfaces:**
- Consumes: `load_source`, `SourceRecord`, `SourceError` (Task 1).
- Produces:
  - `DATA_ROOT: Path` — `<repo>/data`, overridable by env `EQUILIBRIA_DATA_DIR`.
  - `iter_sources(root: Path = DATA_ROOT) -> Iterator[SourceRecord]` — every `source.yaml` under `root`.
  - `reference(dataset: str, tool: str, *, root: Path = DATA_ROOT, **variant) -> Path` — matching rule from the design: same `dataset` and `tool`, `kind == "result"`, and every requested key present in `variant` with an equal value; exactly one match, else `SourceError`.

- [ ] **Step 1: Write the failing tests** (build small trees under `tmp_path`)

```python
def test_single_match_returns_its_folder(tmp_path): ...
def test_omitted_keys_are_free(tmp_path): ...            # ask ifsub=0 only; folder has closure too → match
def test_two_matches_raise_and_name_both(tmp_path): ...  # capFix and capFlex, ask ifsub=0 → SourceError naming both folders
def test_wrong_variant_raises(tmp_path): ...             # folder ifsub=1, ask ifsub=0 → SourceError naming ifsub=1 folder as candidate
def test_no_match_lists_candidates(tmp_path): ...        # unknown dataset → message lists datasets that exist
```

- [ ] **Step 2:** run → FAIL (`ImportError: reference`).
- [ ] **Step 3:** Implement `DATA_ROOT`, `iter_sources`, `reference`.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** Commit `feat(sources): reference() pide una referencia por lo que es`.

### Task 3: tree checks and the repo test

**Files:**
- Modify: `src/equilibria/sources.py`
- Test: `tests/data/test_sources_checks.py` (the checks on built trees) and `tests/data/test_data_tree.py` (runs them on the real `data/`)

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: `check_tree(root: Path = DATA_ROOT) -> list[str]` — one message per problem, empty when clean. The four checks of the design: (1) every file under `root` is inside a folder (or ancestor) with a `source.yaml`, except `README.md` and `source.yaml`; (2) for `verified` results every `generated_from` sha256 matches the current file; (3) every file listed in `outputs` exists with that sha256 and the folder has no file missing from `outputs`; (4) no `pending` record (the repo test skips (4) only on a branch whose phase is open: env `EQUILIBRIA_ALLOW_PENDING=1`, never set in CI on `main`).

- [ ] **Step 1: Write the failing tests** — one per check, each built so that check (and only it) fails:

```python
def test_file_without_source_is_reported(tmp_path): ...
def test_changed_input_is_reported(tmp_path): ...       # write source, then edit the input file
def test_edited_output_is_reported(tmp_path): ...
def test_extra_output_file_is_reported(tmp_path): ...
def test_pending_is_reported(tmp_path): ...
def test_clean_tree_has_no_messages(tmp_path): ...
```

and in `test_data_tree.py`: `def test_data_tree_is_clean(): assert check_tree() == []`.

- [ ] **Step 2:** run both files → FAIL (`ImportError: check_tree`).
- [ ] **Step 3:** Implement `check_tree`. Create `data/README.md` (Task 6 fills it) so `data/` exists.
- [ ] **Step 4:** run → PASS (the real tree holds only `README.md`).
- [ ] **Step 5:** Commit `feat(sources): check_tree y test del árbol data/`.

### Task 4: `scripts/dev/write_source.py`

**Files:**
- Create: `scripts/dev/write_source.py`
- Test: `tests/scripts/test_write_source.py`

**Interfaces:**
- Consumes: `sha256_file`, `load_source` (to validate what it wrote).
- Produces (importable via `importlib`, like `tests/scripts/test_check_parity_gates_stamp.py` does):
  - `machine_info() -> dict` with exactly `os, cpu, cores, ram_gb` (macOS: `sysctl -n machdep.cpu.brand_string` / `hw.memsize`; Linux: `/proc/cpuinfo`, `/proc/meminfo`; Windows: `platform.processor()`, `GlobalMemoryStatusEx` via `ctypes`; `ram_gb` rounded to 1 decimal).
  - CLI: `write_source.py --folder DIR --kind result --model M --dataset D --tool T [--tool-version V] --variant k=v ... --model-file PATH --data-dir PATH [--solver NAME] [--equilibria-commit auto] -- <command...>`. It runs the command (cwd = `--folder`), times it (`time.perf_counter`), and only if the exit code is 0 writes `source.yaml` with `status: verified`, `outputs` = every file in the folder, `generated_from` hashes, `run.date` (ISO with offset), `run.duration_s`, `run.machine`, `equilibria_commit` (`git rev-parse --short HEAD`). Then it calls `load_source` on its own output.
  - `--no-run` mode for `kind: data|model`: hashes and writes, no command.

- [ ] **Step 1: Write the failing tests**

```python
def test_machine_info_has_exactly_the_four_keys(): ...
def test_runs_command_and_records_it(tmp_path): ...       # command writes out.txt → outputs has out.txt, duration_s > 0, status verified
def test_failed_command_writes_nothing(tmp_path): ...     # `false` → exit != 0, no source.yaml
def test_written_file_loads(tmp_path): ...                # load_source(folder).kind == "result"
```

- [ ] **Step 2:** run → FAIL (script missing).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** Commit `feat(dev): write_source.py corre, cronometra y registra la referencia`.

### Task 5: `scripts/dev/check_layout.py`

**Files:**
- Create: `scripts/dev/check_layout.py` (absorbs `scripts/check_lowercase_filenames.py`: same `ALLOWED_EXACT`/`ALLOWED_PREFIXES`, with paths updated)
- Delete: `scripts/check_lowercase_filenames.py`
- Modify: `.github/workflows/tests.yml:22-33` (one step: `python scripts/dev/check_layout.py`; the inline loose-scripts rule moves into the script), `.pre-commit-config.yaml` (local hook `check-layout`, `always_run: true`, `pass_filenames: false`)
- Test: `tests/scripts/test_check_layout.py`

**Interfaces:**
- Produces: `violations(paths: list[str]) -> list[str]` (pure, on a list of tracked paths) and `main() -> int` (runs it on `git ls-files`).

Rules, each with its own exception list:
1. Root: allowed directories `src data tests scripts examples docs specs .github`; allowed files `README.md CHANGELOG.md CONTRIBUTING.md ROADMAP.md GLOSSARY.md LICENSE NOTICE CITATION.cff Makefile pyproject.toml uv.lock .gitignore .pre-commit-config.yaml .readthedocs.yaml`. Temporary, removed by the phase that empties them: `datasets notebooks output reports results runs`.
2. Lowercase (existing rule and exceptions).
3. English: Spanish word list `de del la las el los sin con para por y que un una punto residuos bloques registro bienestar desempleo corridas resultado cada sobre entre cuando como donde fijo libre paso pasos precio precios ecuacion ecuaciones prueba pruebas datos semilla asentado cierre choque modelo nuevo viejo lectura escritura diccionario nombres ejemplos pagina cobertura salida entrada hecho mapa guia notas`; a path segment fails if a token (split on `_ - .`) is in the list or it contains `ñáéíóú`. Temporary exceptions, exactly these 16 (removed by the phase that renames them):
   `docs/architecture/monolito_vs_bloques.md`, `docs/architecture/registro_de_bloques.md`, `runs/gempack_20x41_validation/RESULTADO.md`, `scripts/windows/corridas_paper.ps1`, `tests/babel/gdx/test_gdxdump_sin_rutas_fijas.py`, `tests/blocks/test_sin_registro_de_bloques.py`, `tests/fixtures/gtap7_3x3_punto_sin_shock_3f3fe5f.json.gz`, `tests/fixtures/gtap7_3x3_residuos_sin_shock_3f3fe5f.json.gz`, `tests/fixtures/nus333_bienestar_gams_levels.json.gz`, `tests/fixtures/nus333_bienestar_gempack.json.gz`, `tests/fixtures/nus333_desempleo_gams_levels.json.gz`, `tests/templates/gtap/_desempleo.py`, `tests/templates/gtap/test_driver_default_es_bloques.py`, `tests/templates/gtap/test_nus333_bienestar.py`, `tests/templates/gtap/test_nus333_me9_pasos.py`, `tests/templates/gtap/test_nus333_tbl65b_desempleo.py`.
4. Loose scripts: no `scripts/*.py` at the top of `scripts/`.

Messages point to `data/README.md` (root rule) and name the rule broken.

- [ ] **Step 1: Write the failing tests**

```python
def test_new_root_folder_is_rejected(): assert violations(["newdir/x.py"])
def test_known_root_entries_pass(): assert violations(["src/a.py", "README.md", "data/README.md"]) == []
def test_spanish_name_is_rejected(): assert violations(["tests/test_sin_shock.py"])
def test_accent_is_rejected(): assert violations(["docs/notes/análisis.md"])
def test_word_inside_a_token_is_not_spanish(): assert violations(["tests/test_decomp_design.py"]) == []
def test_uppercase_still_rejected(): assert violations(["tests/Foo.py"])
def test_loose_script_rejected(): assert violations(["scripts/foo.py"])
def test_repo_passes_today(): assert main() == 0   # with the temporary exceptions
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** Implement; delete the old script; update CI and pre-commit.
- [ ] **Step 4:** run the tests → PASS; `python scripts/dev/check_layout.py` → exit 0; `prek run check-layout --all-files` → Passed.
- [ ] **Step 5:** Commit `feat(dev): check_layout.py — raíz cerrada y nombres en inglés`.

### Task 6: where things go

**Files:**
- Modify: `data/README.md` (the map: the layout of the design, the `source.yaml` fields, how to write one with `write_source.py`, how to ask with `reference()`, the two layout rules)
- Modify: `CONTRIBUTING.md` (a short "Where files go" section pointing to `data/README.md` and the root rule)
- Modify: `specs/areas.toml` (`data` area gains `src/equilibria/sources.py`; `[docs] data` gains `data/README.md`)
- Modify: `specs/data/context.md`, `specs/data/specs/specs.md` (fill the stubs; baseline gets no requirement until the phase that implements it)
- Modify (outside the repo, maintainer's tooling): `~/proyectos/dev-tools/equilibria-tools/agent-docs/CLAUDE.md` and `AGENTS.md` — a "Data and references" section: everything non-code is under `data/`, read `data/README.md`, ask with `reference()`, never add a root folder, names in English. Commit in the dev-tools repo separately.

- [ ] **Step 1:** Write the pages.
- [ ] **Step 2:** `uv run pytest tests/docs -q` (doc gate) and `python scripts/dev/spec_tools.py check-pr --base origin/main --labels ""` → no errors.
- [ ] **Step 3:** Commit `docs(data): data/README.md y punteros en CONTRIBUTING y specs`.

### Task 7: phase close

- [ ] Full suite: `uv run pytest -q -rA > after.txt`; compare by test name with the same run on `origin/main` (`before.txt`): only new tests added, none changed state.
- [ ] `python scripts/dev/check_layout.py` → 0; `prek run --all-files` → all Passed.
- [ ] Push and PR only with the maintainer's permission.
