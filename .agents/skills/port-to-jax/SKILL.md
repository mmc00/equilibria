---
name: port-to-jax
description: Use when moving an equilibria CGE model (a template built from blocks, or an original GAMS/GEMPACK model) to the JAX "family" backend so it runs with no Pyomo model, or when a family run of a non-GTAP template fails, is not square, or does not match its reference.
---

# Port a model to the JAX (family) backend

## Overview

Two stretches: **(1) original code → equilibria blocks** (parity against the original), **(2) blocks → JAX**. Stretch 2 is generic code that already exists: do not write a new period model, array builder or Newton. Worked example: `simple_open` (`templates/simple_open/template.py`, `tests/family/test_simple_open_*.py`).

## When to use

- A template has a non-Pyomo `create_model()` of blocks and runs through `Exercise` on Pyomo, and you want `backend="family"`.
- A model exists only as `.gms`/`.tab`: do stretch 1 first (section "Stretch 1").
- Not for GTAP: it has its own family route (`family/gtap_periods.py`, `family/run_setup.py`, `solver/newton_gtap.py`).

## Stretch 2: blocks → JAX (reuse, do not rewrite)

| Piece | Where | What it does |
|---|---|---|
| `build_arrays(model, instruments, periods)` | `family/generic_arrays.py` | records the blocks (`FamilyBackend`), lays every var/row out per period, instruments fixed + exogenous, **bounds from `StateModel.lower/upper`** |
| `GenericFamilyPeriodModel(mp, backend, state, periods)` | `family/generic_periods.py` | `PeriodModel` protocol + real `residual(period)` + `period_system()` |
| `newton_family(tol=1e-10)` | `solver/newton_family.py` | recipe solver: active period, square check, `newton_tr` with bounds |

Template wiring (copy `SimpleOpenEconomy`): a `backend: Literal["pyomo","family"]` field; override `build()` to return `GenericFamilyPeriodModel(*build_arrays(self.create_model(overwrite=...), INSTRUMENTS, periods), periods)` (fresh every call) and fall back to `super().build`; `default_recipe()` passes `solver=newton_family()`. Steps must use only the `PeriodModel` protocol (`core/period_model.py`); fix any step that touches `.pyomo`.

## Validation ladder (in order)

Reference data and compare map are the ones the template's Pyomo parity test already uses (`equilibria.sources.reference(...)`, `Mapping.builtin(<model>)`); they are stretch-1 outputs. If they do not exist, finish stretch 1 first.


1. `residual("base") < 1e-9` at the calibration: proves the recorder reproduced every row.
2. `check` solves back to `base` (every cell, rel/abs 1e-9).
3. No Pyomo **model** built: wrap the run in `_spies()` from `tests/family/test_family_e2e.py`, expect `{"ConcreteModel": 0, "PyomoNLP": 0, "identify_variables": 0}`. Do not assert `"pyomo" not in sys.modules`: the recorder imports `pyomo.environ`.
4. The template still defaults to `backend="pyomo"` (a test), and two runs from one template do not leak (different model objects, same results; a run with no shock returns the calibration).
5. **Measure before choosing a tolerance:** run both backends against the reference (`scripts/simple_open/measure_gams_gap.py` is the template) and report the worst gap per period, its variable, Walras and EV. The model owner sets the tolerance (simple_open: 1e-9 after a 4.2e-11 worst gap; a precedent, not a default); tighten a copy of the compare map inside the test (`dataclasses.replace(v, tol=Tolerance(...))`), never edit the shared yaml. Pyomo also gets measured, so a gap can be blamed on the reference, not the port.
6. Prove the parity test can fail (e.g. a different shock value fails the shock cells), then revert.

## Common mistakes

| Mistake | Fix |
|---|---|
| Tests import old code | `/opt/homebrew/envs/eq-mumps` has equilibria installed editable from another worktree: run `PYTHONPATH=src /opt/homebrew/envs/eq-mumps/bin/python -m pytest ...` from the worktree root. The uv `.venv` has no jax. |
| `fly` crashes with `NotImplementedError` in `residual` | `FamilyPeriodModel.residual` is a stub; use `GenericFamilyPeriodModel`. |
| Newton stops at 1e-7 | `newton_tr` reads `ftol` from `EQUILIBRIA_GTAP_TR_FTOL` (1e-7) when not passed; pass it. |
| Levels go to 0/negative | Level blocks bound `Reals` vars below by 1e-5 (`LEVEL_FLOOR`); GTAP's `MPArrays.build` only knows `NonNegativeReals`. Take bounds from the state. |
| "not square" | An instrument or the numeraire is free (or a row is missing). Count free cells vs active rows of the period; fix the closure in the template, not the solver. |
| Recorder fails on a block | It records `+ - * / ** neg sqrt log exp`, `math.prod`, `sum`. Rewrite the block with those; never fall back to Pyomo. |
| New test missing in CI | Add it by hand to `.github/test_files_python.txt`; never run `scripts/dev/plan_test_groups.py --refresh-files` (drops family files). |
| Commit fails once | pre-commit runs `ruff --fix` and `ty`: re-add and commit; guard `Optional` attributes in tests. |

## Stretch 1: original code → blocks (PENDING)

Not yet written from a real case. To be filled from the GTAP dynamic port. Until then: build the template from blocks in `equilibria.blocks.*` (a model-specific block only where no generic one fits, e.g. `StdcgeInvestment`), calibrate from the SAM, and reach cell parity with the original on Pyomo first (`equilibria-parity-debug`, `xmodel-parity`); then stretch 2.

## Claude Code tips

- Related skills: `equilibria-parity-debug`, `xmodel-parity` (parity gaps), `superpowers:test-driven-development`.
- Use a subagent to map an unfamiliar template; keep the ladder results in your own context.
