# data/

Everything equilibria reads or is compared against that is not code lives here
(ADR-0010): input datasets, the reference models that produce results (GAMS
`.gms`), the results of those models (GAMS `.nl`/GDX, GEMPACK HAR/SL4), and the
small files the readers are tested with. `src/` holds code, `tests/` holds
tests; neither holds data.

> Migration in progress (`specs/data/changes/one-data-folder/`). Until each
> phase lands, its files are still in their old places: `datasets/`,
> `tests/fixtures/`, `src/equilibria/templates/reference/`, `runs/`.

## Layout

One dataset is one folder, named exactly as the code names it. Its results sit
next to it, one folder per tool and variant.

```
data/
  gtap7/
    gams/                     the GAMS GTAP v7 model (.gms) that generates the results
    gtap7_3x3/
      basedata.har sets.har default.prm ...   + source.yaml (kind: data)
      gams_ifsub0/            .nl, GDX ...    + source.yaml (kind: result)
      gams_ifsub1/
      gempack_tm10_s8-16-32/
    gtap7_10x7/ ...
    nus333/ ...
  gtap6/ ...
  pep/ ...
  gdx_tests/                  small GDX files for the reader tests
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

## Using a reference in a test

Ask for it by what it is, never by path:

```python
from equilibria.sources import reference

folder = reference("gtap7_3x3", "gams", ifsub=0)
```

It raises when no folder matches (say, the result exists only for `ifsub=1`) or
when more than one does (two closures: the test has to say which).

## Rules

- No new folders or files at the repository root; `scripts/dev/check_layout.py`
  rejects them in pre-commit and CI.
- File and folder names are lowercase English. Names that come from an upstream
  source keep theirs, listed with the reason in `check_layout.py`.
- Commit messages are English (`scripts/dev/check_commit_msg.py`).
