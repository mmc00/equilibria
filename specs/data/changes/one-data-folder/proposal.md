# One `data/` folder, with a `source.yaml` for every reference

## Why

Reference files (datasets, GAMS models, GAMS and GEMPACK results) are spread
over seven folders with three different sorting rules, one dataset is committed
twice, and no file records how it was produced. Three times a parity test
stayed green while reading a reference that was not what we believed. The
latest: the `.nl` gate compares Python `ifSUB=0` against GAMS fixtures that
look like `ifSUB=1` (ADR-0010). The repository root and `docs/` have the same
problem on a smaller scale: one-file folders (`output/`, `results/`), old
reports, and fifteen `docs/` subfolders mixing user docs, internal notes and
tooling.

## What changes

- One `data/` folder holds every non-code file a model reads or is compared
  against; one dataset is one folder; results sit next to their data.
- Every reference result folder has a `source.yaml`; a test enforces it and
  checks the variant a test asks for against the one recorded.
- `nus333` and `9x10` ship in the wheel from `data/`; the copies under
  `src/equilibria/templates/reference/gtap/data/` go away.
- `output/`, `results/`, `reports/`, `runs/` and `notebooks/` leave the root;
  the loose `.gms` in `scripts/` move to `data/`.
- `docs/` goes from fifteen subfolders to four (`site/`, `adr/`, `notes/`,
  `archive/`) plus `agents/`; `specs/` stays as it is.
- `CONTRIBUTING.md`, `AGENTS.md`, the maintainer's `CLAUDE.md`, `specs/readme.md`
  and `specs/areas.toml` point to the new places, and `data/README.md` is the
  map.

## Issues

- #23 (the `.nl` gate): its fixtures are regenerated in `ifSUB` 0 and 1 on top
  of this change, as the first results with a verified `source.yaml`.
- One issue per phase of `design.md`, opened when the design is approved.

## Impact (areas, ADRs)

- Areas: `data` (new baseline), `parity` (fixtures and gates), `templates/gtap`
  and `templates/pep` (paths only), docs.
- Creates ADR-0010. Relies on ADR-0008 and ADR-0009.
- PEP files (`reference/pep`, `reference/pep2`, `results/`) move only in their
  own phase, agreed with Ricardo, who owns R-PEP.
