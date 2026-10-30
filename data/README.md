# data/

Everything equilibria reads or is compared against that is not code lives here
(ADR-0010): input datasets, the reference models that produce results (GAMS
`.gms`), the results of those models (GAMS `.nl`/GDX, GEMPACK HAR/SL4), and the
small files the readers are tested with. `src/` holds code, `tests/` holds
tests; neither holds data.

> Migration in progress (`specs/data/changes/one-data-folder/`). Until each
> phase lands, its files are still in their old places: `tests/fixtures/`,
> `src/equilibria/templates/reference/`, `runs/`. The input datasets moved in
> phase 1; the GAMS model and the GAMS results that run locally in phase 2a.
> The MCP results that need NEOS are phase 2b.

## Layout

One dataset is one folder, named exactly as the code names it. Its results sit
next to it, one folder per tool and variant.

```
data/
  gtap7/
    gams_jgea/                the official JGEA supplementary files, untouched
    gams/                     the GAMS GTAP v7 model that runs (derived_from gams_jgea)
    gtap7_3x3/
      basedata.har sets.har default.prm ...   + source.yaml (kind: data)
      gams_nl_altertax_ifsub0/   .nl per phase   + source.yaml (kind: result)
      gams_nlp_gtap_ifsub1/      GDX
      gams_mcp_gtap_ifsub1/      GDX
    gtap7_10x7/ ...
    nus333/ ...
  gtap6/ ...
  pep/ ...
  gdx_tests/                  GDX format specimens for the reader tests
```

## source.yaml

Every folder of results, every dataset folder and every model folder has one.
It is written by a script, never by hand:

```bash
python scripts/dev/write_source.py --folder data/gtap7/gtap7_3x3/gams_ifsub0 \
    --kind result --model gtap7 --dataset gtap7_3x3 --tool gams --tool-version 53.1.0 \
    --variant ifsub=0 --variant closure=capFix \
    --model-file data/gtap7/gams/comp.gms --data-dir data/gtap7/gtap7_3x3 \
    -- gams comp.gms --ifsub=0
```

The script runs the command in the folder, times it, and records the command,
the tool and solver, the exit code, the sha256 of every input and output, and
the machine it ran on (OS, CPU, cores, RAM; never host or user names). If the
command fails, nothing is written. Input data and models, which are not
generated, are recorded with `--no-run` and their `--source` (and `--license`).

`tests/data/test_data_tree.py` fails when a file has no `source.yaml`, when an
input changed after a result was generated, when an output was edited, or when
a record is still `pending`.

## Datasets present

| Folder | What |
|---|---|
| `gtap7/gtap7_{3x3,3x4,5x5,10x7,15x10,20x41}/` | GTAP v7 aggregations (HAR + `v7_consolidated.gdx`) |
| `gtap6/gtap6_{3x3,3x4,5x5,10x7,15x10,20x41}/` | GTAP v6 aggregations (HAR) |
| `gtap7/nus333/` | Burfisher's NUS333 package: HAR data and the book's experiments (`.exp`, `.shk`, `.prm`) |
| `gtap7/9x10/` | GTAP Standard 7 GAMS distribution, HAR inputs |

`nus333` and `9x10` also ship inside the wheel (`load_bundled("gtap", name)`).
Code finds a dataset by name:

```python
from equilibria.sources import dataset_dir

har = dataset_dir("gtap7_3x3") / "basedata.har"
```

To add a dataset: copy its files into `data/<model>/<name>/` and record it with
`write_source.py --kind data --no-run --model <model> --dataset <name> --source
"<where it comes from>" --license "<terms>"`.

## GAMS results

`data/gtap7/gams/` is the model that runs; `source.yaml` there lists every file
that differs from `gams_jgea/` with what changed and why. Each result folder is
`gams_<form>_<mode>_ifsub<k>`:

| Part | Values |
|---|---|
| `form` | `nl` (CONVERT `.nl` of each phase), `nlp` (IPOPT solve), `mcp` (PATH solve) |
| `mode` | `gtap`: the pure model (`comp.gms`), base then the tariff shock; `altertax`: `comp_altertax.gms`, whose `check` step uses the altertax CD elasticities before the shock |
| `ifsub<k>` | GAMS `ifSUB`: `0` keeps the substituted equations explicit, `1` substitutes them |

`scripts/gtap/regenerate_gams.py` regenerates them from `data/gtap7/gams/`, one
job per folder through `write_source.py`; a folder that already has a
`source.yaml` is skipped, so a stopped run resumes.

Local or NEOS: CONVERT and IPOPT have no size limit with the local license, so
every `nl` and `nlp` result is local. PATH stops at 1000 rows, so the only local
`mcp` results are `gtap7_3x3` ifSUB=1; the other MCP results come from NEOS
(phase 2b) and keep the NEOS run date and configuration in their `source.yaml`.

## Using a reference in a test

Ask for it by what it is, never by path:

```python
from equilibria.sources import reference

folder = reference("gtap7_3x3", "gams", form="nl", mode="altertax", ifsub=0)
```

`find_reference()` takes the same arguments and returns `None` instead of
raising, for a test that skips when the result is not there.

It raises when no folder matches (say, the result exists only for `ifsub=1`) or
when more than one does (two closures: the test has to say which).

## Rules

- No new folders or files at the repository root; `scripts/dev/check_layout.py`
  rejects them in pre-commit and CI.
- File and folder names are lowercase English. Names that come from an upstream
  source keep theirs, listed with the reason in `check_layout.py`.
- Commit messages are English (`scripts/dev/check_commit_msg.py`).
