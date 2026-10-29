# Contributing to equilibria

Thanks for helping improve **equilibria**. This guide covers the dev setup and
the pre-commit hooks.

## Dev setup

```bash
uv sync --dev            # deps + tooling (ruff, ty, pytest)
uv tool install prek     # pre-commit runner in Rust (drop-in for pre-commit)
prek install             # install the git hooks (pre-commit and commit-msg)
```

## Pre-commit hooks

We use [prek](https://github.com/j178/prek), which reads the standard
`.pre-commit-config.yaml`. **Use `prek`, not `pre-commit`** — the config is
identical, only the runner differs.

```bash
prek run --all-files     # run every hook over the whole repo
prek run ruff-format     # run a single hook
```

Hooks (fast hygiene only, scoped to `src/` and `tests/`):

- **ruff** — lint (`--fix`) and format, reusing `[tool.ruff]` from `pyproject.toml`.
- **ty** — type check (Astral's checker), run as a *ratchet* (see below).
- **hygiene** — trailing-whitespace, end-of-file-fixer, check-yaml,
  check-added-large-files, check-merge-conflict.

These hooks **never run the solver or GAMS** — that is deliberately kept out of
pre-commit (see *Parity gates* below). Byte-sensitive data and oracles
(`templates/reference/`, `**/golden/`, `**/fixtures/`, `.diff/.gms/.inc/.gdx/.har`)
are excluded from every hook so they are never reformatted.

### The `ty` ratchet

`ty` runs in **ratchet** mode: it blocks a commit that introduces a type error
into a file that is currently clean, but excludes the files that already carry
diagnostics (frozen type debt). That exclude list is the backlog for the
type-cleanup phase.

When you clean the types of an excluded file, regenerate the list and paste it
into the `exclude:` block of the `ty` hook in `.pre-commit-config.yaml`:

```bash
uv run python scripts/dev/gen_ty_exclude.py          # prints the exclude: regex
uv run python scripts/dev/gen_ty_exclude.py --list    # prints the raw file list
```

## Where files go, names and commit messages

- **Data and references go under `data/`**, one dataset per folder, each result
  with a `source.yaml` written by `scripts/dev/write_source.py`. Read
  [`data/README.md`](data/README.md) before adding any non-code file (ADR-0010).
- **The root is closed**: `src/ data/ tests/ scripts/ examples/ docs/ specs/`
  and the usual top-level files. A new root entry fails `scripts/dev/check_layout.py`.
- **Everything is English**: file and folder names, code, comments, messages,
  docs and **commit messages**. `prek install` installs a `commit-msg` hook that
  rejects a Spanish message, and CI checks the commits of every PR.

## Parity gates (GTAP / PEP)

The Python↔GAMS parity sweeps (NLP-vs-NLP and MCP-vs-MCP) are **mandatory
before any push / PR** and are enforced by a separate hook
(`scripts/gtap/run_parity_gates.py`), not by pre-commit — they are slow and
require a local solver. Do not add solver/GAMS checks to `.pre-commit-config.yaml`.

## Tests

```bash
uv run pytest -m "not gams"          # unit + integration (no GAMS)
uv run pytest -m gams                # GAMS parity (needs a local GAMS install)
```

## Specs and changes

Behavior of each architecture area is specified under `specs/`, and larger
changes are proposed there before code. See [`specs/readme.md`](specs/readme.md)
for the flow, and fill in the [PR template](.github/pull_request_template.md):
the `spec-check` CI workflow checks the PR against the specs (warnings for
missing spec/docs updates; failures for inapplicable deltas, hand-edited
baselines, or syncs that differ from their delta). Adding or removing a label
re-runs it.
