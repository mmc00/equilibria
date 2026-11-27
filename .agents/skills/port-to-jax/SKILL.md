---
name: port-to-jax
description: Use when moving an equilibria CGE model (a blocks template, or an original GAMS/GEMPACK model) to the JAX family backend with no Pyomo model, or when a family run of a non-GTAP template fails, is not square, or does not match its reference.
---

# Port a model to the JAX (family) backend

Two stretches: (1) original GAMS/GEMPACK code to equilibria blocks, with parity against the original; (2) blocks to JAX. Stretch 2 is generic code that already exists: do not write a new period model, array builder or Newton. The worked example is `simple_open` (`src/equilibria/templates/simple_open/template.py`, `tests/family/test_simple_open_*.py`). Not for GTAP, which has its own route (`family/gtap_periods.py`, `family/run_setup.py`, `solver/newton_gtap.py`).

Reference: [the three generic pieces and the template wiring](references/pieces.md).

## Steps

1. Check the template is ready for stretch 2: it has a blocks `create_model()` and a Pyomo parity test against its reference.
   ```bash
   uv run pytest -q -p no:cacheprovider -o addopts= tests/templates/simple_open/test_gams_parity.py
   ```
   (Use the template's own parity test; it reads the reference through `equilibria.sources.reference(...)` and `Mapping.builtin(<model>)`, never by path.) Check: it passes. If the template has no reference data or no Pyomo parity, stop: that is stretch 1 (`## Stretch 1` below), not this skill.

2. Install the `family` extra (jax is optional: without it the family tests skip and still exit 0), then run the worked example's family tests.
   ```bash
   uv sync --dev --extra family
   uv run pytest -q -p no:cacheprovider -o addopts= tests/family/test_generic_arrays.py tests/family/test_generic_periods.py tests/family/test_newton_family.py tests/family/test_simple_open_family.py
   ```
   Check: `18 passed` and no `skipped` in the last line. A skip means jax is missing: run the sync again. Do not use a Python from another checkout.

3. Wire the template as in `references/pieces.md`: a `backend: Literal["pyomo","family"]` field defaulting to `"pyomo"`, `build()` returning a fresh `GenericFamilyPeriodModel(*build_arrays(...), periods)`, `default_recipe()` with `solver=newton_family()`. Find any recipe step that touches `.pyomo`.
   ```bash
   grep -rn "\.pyomo" src/equilibria/templates/<model>/ src/equilibria/core/steps.py
   ```
   Check: no recipe step reads `.pyomo`. A hit in the template's `period_model()` or its imports (as `template.py` has) is not a step; the family steps use only the `PeriodModel` protocol (`src/equilibria/core/period_model.py`).

4. Rung 1 of the ladder: the residual at the calibration is below 1e-9. This proves the recorder reproduced every row.
   ```bash
   uv run pytest -q -p no:cacheprovider -o addopts= tests/family/test_<model>_family.py -k "base"
   ```
   Check: `residual("base") < 1e-9`. If not, see Errors (recorder, not square).

5. Rung 2: `check` solves back to `base`, every cell at rel/abs 1e-9 (copy `test_check_returns_the_base`).
   ```bash
   uv run pytest -q -p no:cacheprovider -o addopts= tests/family/test_<model>_family.py -k "check"
   ```
   Check: passes, and the run code for `check` and `shock` is `"optimal"`.

6. Rung 3: no Pyomo model is built in a full run. Wrap the run in `_spies()` from `tests/family/test_family_e2e.py` and expect `{"ConcreteModel": 0, "PyomoNLP": 0, "identify_variables": 0}`. Do not assert `"pyomo" not in sys.modules`: the recorder imports `pyomo.environ`.
   ```bash
   uv run pytest -q -p no:cacheprovider -o addopts= tests/family/test_<model>_family.py -k "no_pyomo"
   ```
   Check: the spy counts are all 0 for your template's test (worked example: `test_simple_open_family.py`).

7. Rung 4: the template still defaults to `backend="pyomo"`, and two runs from one template do not leak (different model objects, same results; a run with no shock returns the calibration).
   ```bash
   uv run pytest -q -p no:cacheprovider -o addopts= tests/family/test_<model>_family.py -k "leak or default"
   ```
   Check: both pass for your template (worked example: `test_simple_open_family.py`). `build()` must record anew on every call.

8. Rung 5: measure before choosing a tolerance. Run both backends against the reference and report the worst gap per period, its variable, Walras and EV. `scripts/simple_open/measure_gams_gap.py` is the template; copy it for your model.
   ```bash
   uv run python scripts/simple_open/measure_gams_gap.py
   ```
   Check: you have a table with the worst gap per backend and period (simple_open: 4.2e-11 at shock `E[MLK]`, EV diff 1.7e-10, Pyomo and family alike). Give it to the model owner: they set the tolerance (simple_open uses 1e-9, a precedent, not a default). Pyomo is measured too, so a gap can be blamed on the reference and not the port.

9. Write the parity test at the owner's tolerance. Tighten a copy of the compare map inside the test (`dataclasses.replace(v, tol=Tolerance(...))`, as `tests/family/test_simple_open_gams_parity.py` does); never edit the shared yaml.
   ```bash
   uv run pytest -q -p no:cacheprovider -o addopts= tests/family/test_<model>_gams_parity.py
   ```
   Check: passes (the worked example has 6 tests), and `git diff -- src/equilibria/compare/maps` is empty.

10. Rung 6: prove the parity test can fail. Change the shock value (for example `{"tau_m": 0.5}` in the test), run it, then revert.
    ```bash
    uv run pytest -q -p no:cacheprovider -o addopts= tests/family/test_<model>_gams_parity.py
    ```
    Check: it fails on the shock cells with the wrong shock and passes again after the revert (`git diff` shows no leftover change).

11. List the new test files in the CI groups. The refresh rewrites both lists from a collection and drops every file that skips here, so restore what it dropped: only your own additions may remain.
    ```bash
    uv run python scripts/dev/plan_test_groups.py --refresh-files
    git diff .github/test_files_python.txt .github/test_files_gdxdump.txt
    ```
    If the diff has any `-` line, undo the refresh (`git checkout -- .github/test_files_python.txt .github/test_files_gdxdump.txt`) and add your files by hand, sorted as Python sorts:
    ```bash
    uv run python -c "from pathlib import Path; p = Path('.github/test_files_python.txt'); p.write_text(''.join(f + chr(10) for f in sorted({*p.read_text().split(), 'tests/family/test_<model>_family.py', 'tests/family/test_<model>_gams_parity.py'})))"
    ```
    Check: `git diff .github/test_files_python.txt` shows only `+` lines, for your new `tests/family/test_<model>_*.py` files.

## Errors

| Failure | Fix |
|---|---|
| `NotImplementedError` in `residual` when the family model runs | `FamilyPeriodModel.residual` is a stub: build `GenericFamilyPeriodModel`, not the GTAP class. |
| `residual` is nonzero after you deactivate a row | It counted every row. `GenericFamilyPeriodModel.residual` takes the max over active rows only (`ps.system.F(ps.x0())[ps.active]`), as the solver measures it (4cc279cb). Do not answer from reading: run `uv run pytest -q -p no:cacheprovider -o addopts= tests/family/test_generic_periods.py` and say what it shows. |
| Newton stops at 1e-7 | `newton_tr` reads `ftol` from `EQUILIBRIA_GTAP_TR_FTOL` (1e-7) when it is not passed: `newton_family(tol=...)` passes it explicitly. |
| Levels go to 0 or negative | Level blocks bound `Reals` vars below by `LEVEL_FLOOR` = 1e-5. GTAP's `MPArrays.build` only knows `NonNegativeReals`; use `build_arrays`, which takes bounds from `StateModel.lower/upper`. `newton_family` passes those `lb`/`ub` to `newton_tr` (`src/equilibria/solver/newton_family.py`), so the family Newton keeps the floor: the bug is in how the arrays were built, not in Newton. |
| `ValueError: not square: N rows, M free cells` | An instrument or the numeraire is free, or a row is missing. Count free cells against active rows of the period and fix the closure in the template (the `INSTRUMENTS` tuple), not the solver. `tests/family/test_newton_family.py::test_not_square_raises` shows the case. |
| Recorder fails on a block | It records `+ - * / ** neg sqrt log exp`, `math.prod` and `sum`. Rewrite the block with those; never fall back to Pyomo. |
| The no-Pyomo test fails on `"pyomo" in sys.modules` | The recorder imports `pyomo.environ`. Assert the `_spies()` counts (0 ConcreteModel, 0 PyomoNLP, 0 identify_variables) instead. |
| The second run of one template gives different numbers | `build()` reused arrays. Record fresh on every call: `build_arrays(self.create_model(...), ...)` inside `build()`, nothing cached on the template. |
| Parity fails at 1e-9 but Pyomo is the same distance from the reference | The gap is the reference's precision, not the port: report both measured gaps to the owner (step 8); do not loosen the tolerance yourself. |
| Family tests all skip and pytest exits 0, or `import jax` fails | jax is the `family` extra, not in `uv sync --dev`. Run `uv sync --dev --extra family` and check the line says `passed`, not `skipped`. |
| Tests import old code | You ran a Python from another checkout. Run through `uv run` from the worktree root. |
| `--refresh-files` removes test files from `.github/test_files_python.txt` | It collects the tests and a file that skips drops out: `tests/family/` files without jax (22 files), and those with `pytest.importorskip("mumps")` (`test_altertax_on_core.py`, `test_mar246_headers_golden.py`, `tests/compare/test_gtap7_side_golden.py`: no pymumps in the local venv, CI installs it). It also ignores `--suite` and rewrites both lists. Undo it with `git checkout` and add your lines by hand (step 11). |
| Commit fails once | pre-commit runs `ruff --fix` and `ty`: re-add and commit again after reading what it changed; guard `Optional` attributes in tests. |

## Stretch 1: original code to blocks (pending)

Not yet written from a real case; it will be filled from the GTAP dynamic port. Until then: build the template from blocks in `equilibria.blocks.*` (a model-specific block only where no shared one fits, as `StdcgeInvestment` in `src/equilibria/templates/simple_open/blocks.py`), calibrate from the SAM, and reach cell parity with the original on Pyomo first (see the `equilibria-parity-debug` and `xmodel-parity` skills); then do the steps above.

## Claude Code tips

- Use a subagent to map an unfamiliar template; keep the ladder results in your own context.
- `superpowers:test-driven-development` fits steps 4 to 10: write each rung's test first.
