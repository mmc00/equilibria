# One `data/` folder design

## Context

Measured on `main` (`e153062`, 2026-10-28):

| Where | What | Tracked files |
|---|---|---|
| `datasets/` | input data of 13 datasets (GTAP v6/v7 aggregations, nus333) | 162 |
| `tests/fixtures/gtap7/` | GAMS `.nl` (+ `.row`/`.col`), GDX `out_*_ifsub{0,1}`, generated `comp_*.gms`, per dataset | 58 |
| `tests/fixtures/gtap7_nlp/` | GAMS NLP GDX, `<ds>_{pure,altertax}_ifsub{0,1}.gdx` | 15 |
| `tests/fixtures/gtap7_altertax/` | GAMS altertax `.nl` + GDX per dataset | 15 |
| `tests/fixtures/gtap7_gempack/` | 239 loose GEMPACK files (`updated_*`, `sl4dump_*`, `decomp_*`, `sl4levels*`), `burfisher/` (121), `burfisher_gragg/` (173) | 533 |
| `tests/fixtures/` (root) | 37 small GDX/`.gms` for the GDX reader tests, 8 JSON snapshots | 45 |
| `tests/fixtures/{gdxdump,gtap_logvalue,pyomo_sol}/` | reader and solver fixtures | 5 |
| `tests/{babel/gdx,sam_tools,compare}/fixtures/` | more reader, SAM and comparison fixtures | 9 |
| `src/equilibria/templates/reference/` | GAMS models (`gtap/scripts`, `gtap/gams`, PEP `.gms`), bundled data (`gtap/data/{9x10,nus333}`, PEP SAM), comparison CSVs (`gtap/comp*`, excluded from the wheel by hand) | 49 `.gms`/`.gdx`/`.nl`, 10 MB |
| `runs/` | RunGTAP/GEMPACK comparison runs (nus333, 9x10, 20x41) and the whole `horridge_simple` model | 39 |
| `output/`, `results/` | one GDX each | 2 |

`datasets/nus333/` and `src/equilibria/templates/reference/gtap/data/nus333/`
are identical (same md5 for `basedata.har`, `sets.har`, `default.prm`).

Paths to move are cited in: `tests/fixtures` 78 files, `templates/reference`
80, `datasets/` 39 (+16 using `DATASETS_DIR`), `runs/` 16, `output/nus333` 7,
`docs/findings` 20, `docs/guides` 12, `docs/analysis` 8. The parity-gate stamp
hashes `tests/fixtures/gtap7`, `gtap7_altertax` and `gtap_logvalue`
(`scripts/gtap/check_parity_gates_stamp.py:28`). Sphinx reads the logos from
`docs/assets/` (`docs/site/conf.py:100`).

Constraints: no move may change a number (ADR-0008); every phase ends with the
test suite and the five parity gates (`scripts/gtap/run_parity_gates.py`)
exactly as before; `git mv` so history follows the files; PEP files only with
Ricardo's agreement.

## Approach

### Layout

```
data/
  README.md                    the map: what is where, how to regenerate it
  gtap7/
    gams/                      GAMS GTAP v7 model (today reference/gtap/scripts, gams/, tariff_sim_fixed.gms, scripts/cal_dump*.gms)
    gtap7_3x3/                 today datasets/gtap7_3x3/
      basedata.har sets.har default.prm baserate.har v7_consolidated.gdx
      source.yaml
      gams_ifsub0/             .nl/.row/.col per phase, out GDX, the generated comp_*.gms
      gams_ifsub1/
      gams_altertax_ifsub0/    today gtap7_altertax/ and gtap7_nlp/*_altertax_*
      gams_nlp_ifsub0/         today gtap7_nlp/*_pure_*
      gempack_tm10_s8-16-32/   today the loose updated_/sl4dump_/decomp_ files of this dataset
      equilibria_snapshots/    today the JSON snapshots (settled seed, punto/residuos sin shock)
    gtap7_10x7/ gtap7_15x10/ gtap7_3x4/ gtap7_5x5/ gtap7_20x41/ gtap7_9x10/
    nus333/
      ...
      gempack_burfisher/<experiment>/       today burfisher/
      gempack_burfisher_gragg/<experiment>/ today burfisher_gragg/
      rungtap_tm10/                         today runs/nus333_compare/
  gtap6/gtap6_<ds>/            today datasets/gtap6_*
  horridge_simple/             today runs/horridge_simple/
  pep/                         today reference/pep, reference/pep2, results/  (own phase, with Ricardo)
  gdx_tests/                   today the small GDX/.gms of tests/fixtures/ and gdxdump/
  solver_tests/                today pyomo_sol/
```

A dataset folder has the same name the code uses (`"gtap7_3x3"` lives in
`data/gtap7/gtap7_3x3/`; decided by the maintainer), so a name found in code is
found on disk as is. Results folders are named `<tool>_<variant>`.

### `source.yaml`

One per folder of results, one per dataset folder, one per model folder:

```yaml
kind: result              # result | data | model
model: gtap7
dataset: gtap7_3x3        # result and data only
tool: gams                # gams | neos-convert | gempack | rungtap | equilibria
tool_version: "53.1.0"
variant: {ifsub: 0, closure: capFix, mode: gtap}   # what a test can ask for
phases: [base, check, shock]
generated_from:           # paths relative to data/, with the sha256 of each file
  model: {path: gtap7/gams/comp.gms, sha256: ...}
  data: {path: gtap7/gtap7_3x3/, sha256: {basedata.har: ..., sets.har: ..., default.prm: ...}}
solver: {name: CONVERT, options: {}}                # solver and the options it ran with
command: "python scripts/gtap/nl_compare.py --dataset gtap7_3x3 --ifsub 0 ..."
equilibria_commit: 212a873                          # the scripts that drove the run
exit: {code: 0, model_status: "", solver_status: ""}  # as the tool reported them
outputs:                  # every file of the folder, with its sha256
  gams_base.nl: ...
run:                      # where and how long (requested by the maintainer)
  date: 2026-10-28T21:14:03-06:00
  duration_s: 412.7       # wall clock of the command
  machine:
    os: "macOS 26.2 (Darwin 25.2.0)"
    cpu: "Apple M3 Max"
    cores: 16
    ram_gb: 64
status: verified          # verified | pending (only while a phase is open)
notes: ""
```

`scripts/dev/write_source.py` fills `run.machine`, `run.duration_s` (it runs
the command itself and times it), `outputs` and `equilibria_commit`; nothing
of that is typed by hand. A NEOS run records NEOS as the machine
(`os: "NEOS server"`, the rest empty). The test checks `outputs` too: a result
edited after it was generated is red. No host name or user name is recorded.

For `kind: data`: `source` (URL or "aggregated from X with Y"), `license`
(nus333: public access, redistribution not stated by GTAP — to confirm).

### Asking for a reference

`equilibria.data` (or a test-only helper; builder's call) exposes
`reference(dataset, tool, **variant) -> Path`. A folder matches when its
`dataset` and `tool` are equal and every key of the request is in its
`variant` with the same value (keys the request omits are free). Exactly one
folder must match: none, or more than one, raises and names the candidates, so
a test that leaves out `closure` where two closures exist has to say which. Tests stop
building paths by hand. The `.nl` gate asks for
`reference("gtap7_3x3", "gams", ifsub=0)`; with today's fixtures that is an
error that names the mismatch instead of a green test.

### The test (`tests/data/test_sources.py`)

1. Every file under `data/` belongs to a folder with a `source.yaml`
   (`README.md` and the `source.yaml` files themselves excepted).
2. For `status: verified`, every `generated_from` sha256 matches the current
   file: a result whose inputs changed after it was made is red.
3. No folder is `pending` on `main` once its phase is merged: every result is
   regenerated by its recorded command (decided by the maintainer, 2026-10-28).
4. Every `source.yaml` parses against the schema (unknown keys are errors).

### Packaging

The wheel copies `data/gtap7/nus333` and `data/gtap7/9x10` into the package at
build time (hatch `force-include`), and `load_bundled` finds them both in an
installed wheel and in an editable checkout. The `exclude` of
`reference/gtap/comp*` in `pyproject.toml` goes away with the files. PEP keeps
shipping what it ships today until Ricardo decides (`load_bundled("pep")` is
known broken, R-PEP). `load_bundled` keeps its signature; only its path
lookup changes (the copy inside an installed wheel first, the repository
`data/` in a checkout). It does not go through `reference()`: bundled data is
`kind: data`, not a result.

### Writing a `source.yaml`

`scripts/dev/write_source.py` writes it: it fills `generated_from` with the
sha256 of the inputs it is given, so nobody computes hashes by hand. A result
is `verified` only when it was produced by the recorded command from the
recorded inputs during this change (or later); moving a file never makes it
verified. The parity-gate stamp (`check_parity_gates_stamp.py`) is a cache of
"gates ran on these inputs": `INPUT_TREES` is updated to the new paths in the
same PR, and the gates are re-run to write a new stamp, which is also how
each phase proves its figures are unchanged.

### Root and docs

- Root after the change: `src/ data/ tests/ scripts/ examples/ docs/ specs/`.
  `reports/` → `docs/archive/`; `notebooks/` → `examples/notebooks/`; the
  scripts in `runs/*/` move to `scripts/gtap/` or `data/` next to their run.
- `docs/` keeps `site/` (the only user docs; absorbs `assets/` into
  `site/_static/`), `adr/`, `agents/`, and gains `notes/` (all dated internal
  notes: today `findings/`, `analysis/`, `plans/`, `technical/`,
  `architecture/`, `reference/`, the loose `docs/*.md`) and `archive/`
  (what no longer applies). `docs/guides/` disappears: its MIP and simulation
  guides go to `site/guide/` or `archive/`; the seven `pep_*.md` move only as
  Ricardo decides (they are listed under `templates/pep` in `[docs]`).
- `specs/` stays. `specs/readme.md`, `specs/areas.toml` (`[docs]`, `unowned`),
  `CONTRIBUTING.md`, `AGENTS.md` and the maintainer's `CLAUDE.md`
  (`~/proyectos/dev-tools/equilibria-tools/agent-docs/`) point to `data/README.md`
  and the new `docs/` layout.

### Two layout rules, enforced (requested by the maintainer)

Both run in pre-commit and CI, in one script that replaces
`scripts/check_lowercase_filenames.py` (`scripts/dev/check_layout.py`; it keeps
the lowercase rule and its exceptions):

1. **No new root entries.** The root holds exactly: `src/ data/ tests/
   scripts/ examples/ docs/ specs/`, the dot entries (`.github/`, `.gitignore`,
   `.pre-commit-config.yaml`, `.readthedocs.yaml`), and the files `README.md
   CHANGELOG.md CONTRIBUTING.md ROADMAP.md GLOSSARY.md LICENSE NOTICE
   CITATION.cff Makefile pyproject.toml uv.lock`. Any other tracked entry at
   the root is red, with a message that says where things go (`data/README.md`).
   Changing the list is a reviewed edit of the script, never a side effect.
2. **File and folder names in English.** Only names (paths), not contents:
   comments, docstrings and docs keep their language. A name is red when a
   token of it (split on `_ - .`) is in a list of Spanish words kept in the
   script, or when it has an accent or `ñ`. The list cannot catch every
   Spanish word, so review still applies; it catches the common ones. Measured
   on `main`: 16 names fail today (`test_sin_registro_de_bloques.py`,
   `RESULTADO.md`, `nus333_bienestar_gempack.json.gz`, ...); they are renamed
   in the phase that moves their folder, or in phase 6. Names that come from an
   external source and must keep it (a GTAP or Burfisher file name) go in the
   exceptions with the reason.

### Phases (one PR each, suite + gates identical before and after)

0. ADR-0010, this change, the `source.yaml` schema, `reference()`, the test
   (empty `data/`, so it passes trivially; a test of the test proves each check
   can fail), and `scripts/dev/check_layout.py` with today's root and today's 16 Spanish
   names as the only allowed exceptions (each phase removes its own).
1. `datasets/` → `data/<model>/<dataset>/` with `kind: data` sources; packaging;
   delete the `src/` copies of nus333/9x10.
2. GTAP GAMS results (`tests/fixtures/gtap7*`, `reference/gtap/comp*`,
   `output/`) and the GAMS model (`reference/gtap/scripts`, `gams/`,
   `scripts/*.gms`). Every result is regenerated with local GAMS 53 (NEOS
   only if the local license is too small) and written with its
   `source.yaml`. Includes #23: the `.nl` fixtures in `ifSUB` 0 and 1.
3. GEMPACK and RunGTAP results (`gtap7_gempack/`, `runs/`), `horridge_simple`,
   regenerated on the Windows machine (`scripts/windows/corridas_paper.ps1`
   is the starting point).
4. Reader and solver fixtures, including `tests/{babel/gdx,sam_tools,compare}/fixtures/`
   (regenerated from their `generate_*.gms` where one exists),
   JSON snapshots (regenerated by equilibria at a named commit);
   `tests/fixtures/` disappears.
5. PEP (with Ricardo).
6. Root leftovers and `docs/`; doc pointers; `CLAUDE.md`.
7. #23: finish the dictionary pairing of the `.nl` gate (branch
   `fix/23-nl-gate-vacuo`, commit `212a873`) on the fixtures phase 2
   regenerated, until every row of every family the dictionary pairs is
   compared.

### When a regenerated file differs from the old one

Every regeneration is diffed against the file it replaces before the old one
is deleted, and the diff goes in the PR. Identical: nothing to say. Different:
the phase stops for that file and the difference is explained (what changed:
GAMS version, variant, inputs, closure) before anything else; a test that
turns red with the new file is a finding about the old one, never a reason to
keep it (ADR-0008). The `.nl` fixtures are expected to differ: they look like
`ifSUB=1` while the gate builds `ifSUB=0`.

Input data (`kind: data`) is not regenerated: GTAP aggregations and the
Burfisher package are external sources. Their `source.yaml` records where they
come from and their sha256, and that counts as verified.

## Alternatives considered

- **Keep the folders, add `source.yaml` and an index file.** Least churn, but
  the next file still has three plausible homes; rejected by the maintainer.
- **`references/` for results next to `datasets/` for inputs.** Two folders
  whose boundary is a judgement call; rejected as still complex.
- **Move `specs/` under `docs/`.** Breaks `spec_tools.py check-pr` and the CI
  workflow for no gain in order.

## Verification

- Every phase: `uv run pytest` (full suite, same pass/skip counts as before,
  diffed by name) and `scripts/gtap/run_parity_gates.py` (identical stamped
  figures); Sphinx `-W` build; `tests/docs/test_doc_examples.py`.
- `uv build` and install the wheel in a clean venv: `load_bundled("gtap",
  "nus333")` works and the wheel holds no comparison files.
- After phase 7, the `.nl` gate compares every row of every family the
  dictionary pairs, and a request for the wrong `ifSUB` is red.
