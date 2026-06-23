# Altertax multi-period parity test (CI)

## Goal

Add a solving parity gate for the GTAP altertax multi-period pipeline (approach A,
`gtap_multiperiod_driver.solve_multiperiod`), covering **both ifSUB modes** (0 and 1),
with the GAMS reference GDX files committed as fixtures.

The existing `test_gtap7_nl_parity.py` is a *no-solve* gate: it diffs Python `.nl`
coefficients against GAMS `.nl` fixtures. It catches equation/parameter/structure
regressions but cannot catch a regression that **converges to wrong values** (e.g. the
`save<0` bug that silently dropped gtap7_3x4 to 94% while still reporting `code=1`).
This new test closes that gap by actually solving and comparing solved values.

## Scope

- **Datasets:** gtap7_3x3, gtap7_5x5, gtap7_10x7.
- **Modes:** ifSUB=0 and ifSUB=1.
- **Matrix:** 3 datasets × 2 modes = **6 parametrized cases**.
- Out of scope: gtap7_15x10 (2.5 MB × 2 fixtures, slow solve), gtap7_3x4 (secondary),
  gtap7_20x41 (NEOS-only ref). These remain manual (run by passing the dataset/mode
  explicitly); the test file supports them if their fixtures are added, but they are not
  in the committed CI matrix.

## What the test asserts (per case)

For each `(dataset, if_sub)`:

1. **Convergence:** base, check, shock periods all return `code == 1`.
2. **Real-cell match:** match of solved shock-period variables vs the GAMS GDX is
   `≥ 98.0%`, where the denominator **excludes** fixed parameters and ifSUB report/margin
   variables (the same exclusion set the session's measurement harness used). 98% is ~1pt
   below the measured values (3x3 ~99%, 5x5 99.3–100%, 10x7 ~99%) — it catches real
   regressions without false-failing on the known ~1% boundary/basin micro-cell noise.

Measured baselines (must hold): 3x3 ~99%, 5x5 99.3–100%, 10x7 ~99%, all 3 periods code=1,
both modes.

## Design

### Test file

`tests/templates/gtap/test_altertax_multiperiod_parity.py`

- `@pytest.mark.parametrize("dataset,if_sub", [...6 combos...])`.
- Helper `_solve_and_match(dataset, if_sub) -> (codes, match_pct)` that replicates the
  session's measurement harness:
  - Build `GTAPMultiPeriodModel` with `GTAPClosureConfig(if_sub=...)`.
  - `seed_all_periods(m, FIXTURE_GDX)`.
  - `solve_multiperiod(m, params, closure, ref_gdx=FIXTURE_GDX, skip_base_solve=True,
    mute_welfare=True, seed_from_prior=False, holdfix_cd=True)`.
  - Compare shock-period vars vs `gams_levels(REF, ...)` with the `a_/c_/f_/r_` prefix
    strip and the `_GAMS_TO_PY`/ALIAS name map, **excluding** `SKIP`
    (walras/ev/cv/uh/u/ug/us) and `RF` (pfa/pfy/pm/pmcif/pefob/pwmg/pp/pdp/pmp/xwmg/xmgm/
    lambdamg/imptx/exptx) from the denominator.
- `assert all(code == 1 for code in codes)` and `assert match_pct >= 98.0`.

### Skip conditions (so the file is safe to run anywhere)

The test **skips** (not fails) when either is true:

1. The fixture GDX is missing (mirrors `test_gtap7_nl_parity._available_datasets`).
2. `path_capi_python` cannot be imported (the PATH solver is a local package at
   `/Users/marmol/proyectos/path-capi-python/src`, absent on `ubuntu-latest`).

This means: on the automatic `ubuntu-latest` CI the whole file skips cleanly (no solver);
on the `self-hosted` runner (which has PATH + GAMS) it actually runs.

### Marker

Mark the module with `@pytest.mark.altertax_solve` (registered in `pyproject.toml`/
`pytest.ini`) so it can be selected/deselected by marker, and is **not** collected by the
default `ubuntu-latest` test run.

### Fixtures

Copy the 6 reference GDX files (already validated this session) to:

```
tests/fixtures/gtap7_altertax/gtap7_3x3/out_altertax_ifsub0.gdx
tests/fixtures/gtap7_altertax/gtap7_3x3/out_altertax_ifsub1.gdx
tests/fixtures/gtap7_altertax/gtap7_5x5/out_altertax_ifsub0.gdx
...
tests/fixtures/gtap7_altertax/gtap7_10x7/out_altertax_ifsub1.gdx
```

Total ~3.4 MB, committed to git. Source GDX live at
`/Users/marmol/proyectos2/equilibria_refs/<dataset>_altertax_cd/`.

### CI wiring

Add a step to the **existing `gams-tests` job** (`.github/workflows/tests.yml`,
`runs-on: self-hosted`, triggered by `workflow_dispatch` + `inputs.run_gams`):

```yaml
- name: Run altertax multi-period parity tests
  run: |
    uv run pytest tests/templates/gtap/test_altertax_multiperiod_parity.py -v --tb=short
```

The automatic `ubuntu-latest` jobs are **not** touched (the file self-skips there for lack
of solver, but to be explicit we do not add it to those jobs).

## Why this design

- **Follows the repo's existing pattern**: solving tests live on `self-hosted`/manual
  (`gams-tests`), no-solve gates run automatically (`gtap7-nl-parity`). Verified that
  `ubuntu-latest` does not install the PATH solver and pyproject has no `path_capi_python`.
- **Catches both regression classes**: equations (via match%) and convergence (via code=1).
  The `save<0`/EGY-class bug (converges code=1 but to wrong values) is exactly what the 98%
  match gate catches and the `.nl` gate cannot.
- **Reuses the validated measurement harness** → numbers stay consistent with what was
  measured all session.
- **Bounded CI weight**: 3 small datasets, ~3.4 MB fixtures, ~1–2 min solve for 6 cases.

## Non-goals

- Not replacing the `.nl` gate (complementary — that one is fast and runs on every push).
- Not adding 15x10/3x4/20x41 to the committed matrix (manual; fixtures too large or ref
  not cleanly convergible via public NEOS).
- Not making the test run on `ubuntu-latest` (no solver there).
