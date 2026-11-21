# Working on equilibria

Guide for coding agents (Claude Code, Codex and others). equilibria is a Python
library of CGE models (GTAP, PEP, simple_open) that must give the same numbers as
their GAMS and GEMPACK references. Read the rules first: they are checked by hooks
and CI, and a change that breaks one is rejected.

## Rules

- **Everything in the repo is English**: file and folder names, identifiers,
  comments, docstrings, messages, docs and **commit messages**. The maintainer may
  talk to you in Spanish; that does not change the repo language. A `commit-msg`
  hook and a CI job (`scripts/dev/check_commit_msg.py`) reject a Spanish message.
- **The root is closed**: `src/ data/ tests/ scripts/ examples/ docs/ specs/`,
  `.github/`, the agent folders (`.agents/skills/`, `.claude/`) and the usual
  top-level files. `scripts/dev/check_layout.py` rejects anything else, a name with
  capitals or a Spanish word, and a loose `scripts/*.py` (scripts live in
  `scripts/<area>/`).
- **Every non-code file goes under `data/`**, one dataset per folder. Read
  [`data/README.md`](data/README.md) first. Each reference result carries a
  `source.yaml` (written by `scripts/dev/write_source.py`, ADR-0010), and tests ask
  for it with `equilibria.sources.reference(dataset, tool, **variant)`, never by path.
- **The reference models decide** (ADR-0008): when Python and GAMS or GEMPACK
  disagree, Python is wrong until shown otherwise. Do not loosen a tolerance or a
  gate to make a test pass; find the cause.
- **A model is a stack of blocks** (ADR-0001). Blocks hold the economics of one
  period and do not know periods (ADR-0006); a shock lives in the shock block
  (ADR-0003); each name is declared once (ADR-0005); the driver knows no model
  (ADR-0011).
- **Larger changes start as a spec** under `specs/` ([`specs/readme.md`](specs/readme.md)).
  Fill in the PR template; the `spec-check` workflow checks the PR against the specs.

## How to work

Setup and the full detail are in [`CONTRIBUTING.md`](CONTRIBUTING.md). In short:

```bash
uv sync --dev                      # dependencies and tooling
uv tool install prek && prek install   # git hooks (pre-commit, commit-msg, pre-push)
uv run pytest -m "not gams"        # tests without GAMS
uv run pytest -m gams              # GAMS parity tests (need a local GAMS)
```

- **Run Python through `uv run`**, never a bare `python` or a hard-coded `.venv` path.
- **Pre-commit is fast hygiene only** (ruff, `ty` as a ratchet, layout, commit
  message). Never add a solver or GAMS check to it. The pre-push hook runs the small
  family (JAX) datasets against the photos and needs a Python with jax and pymumps:
  set `EQUILIBRIA_FAMILY_PYTHON` to it (`scripts/dev/pre_push_family.sh`).
- **GTAP gates before every push or PR that touches GTAP**:
  `uv run python scripts/gtap/run_parity_gates.py` (about 20 min; `--quick` runs the
  gates without writing the stamp). It refuses a dirty tree, runs the MCP, NLP,
  `.nl` and coverage gates, regenerates the measured docs (commit them if they
  change) and writes `.git/gtap-parity-gates.stamp`. A Claude Code PreToolUse hook
  (`scripts/gtap/claude_hooks/block_push_without_gates.py`) blocks `git push` and
  `gh pr create|merge` while the stamp is missing or stale; other agents run
  `python3 scripts/gtap/check_parity_gates_stamp.py` themselves before pushing;
  pushes that do not touch the GTAP trees are never blocked. `GTAP_GATES_SKIP=1` is
  for emergencies only.
- **Every validation run happens at a clean commit**: commit first, then run, and
  quote the SHA (`git rev-parse --short HEAD`) with the result, a failing one too.
- **Commit messages** follow `type(area): what changed`, as in the log:
  `feat(family): ...`, `fix(gtap): ...`, `test(simple_open): ...`, `docs(skills): ...`.
  Branches are `<type>/<topic>` (`feat/simple-open-jax`, `refactor/gtap-on-core-scripts`).
- **A new test file** must be added to the CI groups:
  `uv run python scripts/dev/plan_test_groups.py --suite python --refresh-files`
  (and `--suite gdxdump`), then commit the result (see `CONTRIBUTING.md`).
- **`gtap7_20x41` runs locally only**: on the Linux CI it runs out of memory (over 11 GB).

## Build a model, in short

`simple_open` (Hosoe's standard CGE model, `src/equilibria/templates/simple_open/`)
is the smallest complete example and matches GAMS in every cell. A model is built
in this order:

1. **Data** under `data/<dataset>/` with its `source.yaml`, and the GAMS or GEMPACK
   result to compare against (`data/stdcge/` for simple_open).
2. **Calibration** reads the data into parameter values and a base point
   (`calibration.py`, `sam.py`).
3. **Blocks** come from `equilibria.blocks` (production, trade, demand,
   institutions, equilibrium). Write a model-specific block only for what no shared
   block covers (`blocks.py`: `StdcgeInvestment`).
4. **Template**, a `ModelTemplate` (`template.py`): `create_model` adds the sets
   and the blocks in order, the instruments a shock changes are fixed, and
   `@overwrite` blocks replace the block with their name (ADR-0004).
5. **Recipe** (`recipe.py`): the periods (`base`, `check`, `shock`), each with a
   kind (`calibrate`, `replicate`, `shock`, `step`), and per kind the steps before
   and after the solve and the solver. Shared steps live in `equilibria.core.steps`;
   model steps such as `apply_shock` in `steps.py`.
6. **Exercise**: `Exercise(template, recipe, shock=...).run()` is one run; read
   results with `exercise.model.value(name, index, period)`.
7. **Backend**: Pyomo by default; `backend="family"` runs the same blocks as JAX
   arrays (`template.build`). See the `port-to-jax` skill.
8. **Parity test** against the reference, cell by cell
   (`tests/templates/simple_open/test_gams_parity.py`, and
   `tests/family/test_simple_open_gams_parity.py` for JAX).
9. **When a number does not match**, see `docs/archive/findings/` and, for GTAP, the
   parity cascade under `scripts/gtap/`.
10. **Document it**: an entry in `docs/site/guide/templates.md`, new domain words in
    `docs/glossary.md`, and an ADR for a new decision.

The why and the pictures are in
[`docs/site/guide/architecture.md`](docs/site/guide/architecture.md) and
[ADR-0011](docs/adr/0011-one-driver-fed-by-a-recipe.md).

## Where things are

| To… | Read |
|---|---|
| Find your way in the docs | [`docs/readme.md`](docs/readme.md) |
| Know what a domain word means | [`docs/glossary.md`](docs/glossary.md) |
| Know why something is built the way it is | [`docs/adr/`](docs/adr/) |
| Know what works today and what is missing | [`docs/roadmap.md`](docs/roadmap.md) |
| Know where data and references go | [`data/README.md`](data/README.md) |
| Use the issue tracker, labels and domain docs | [`docs/agents/`](docs/agents/) |

## Skills

The repo's skills are in `.agents/skills/` (open Agent Skills format), listed in
[`.agents/skills/README.md`](.agents/skills/README.md). Codex reads that folder;
Claude Code reads `.claude/skills/<name>`, a link to the same folder.
