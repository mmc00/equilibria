# Altertax multi-period parity test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a solving parity gate for the GTAP altertax multi-period pipeline covering gtap7_3x3/5x5/10x7 × ifSUB{0,1}, asserting all 3 periods converge (code=1) and real-cell shock match ≥98% vs committed GAMS GDX fixtures.

**Architecture:** A single parametrized pytest module that builds `GTAPMultiPeriodModel`, seeds from a committed GDX fixture, solves via `solve_multiperiod`, and compares solved shock-period variables against the GDX (reusing the session's measurement harness). It self-skips when the PATH solver or a fixture is absent, so it is safe on `ubuntu-latest` (no solver) and actually runs on the `self-hosted`/manual `gams-tests` CI job.

**Tech Stack:** Python 3.11, pytest, pyomo, gams.transfer (via `_diff_core`), path_capi_python (local solver, self-hosted only).

## Global Constraints

- Datasets in committed matrix: gtap7_3x3, gtap7_5x5, gtap7_10x7 only.
- Modes: ifSUB=0 and ifSUB=1 (`GTAPClosureConfig(if_sub=False/True)`).
- Match threshold: `≥ 98.0%` of real cells (denominator excludes fixed params and ifSUB report/margin vars).
- Convergence: all 3 periods (base/check/shock) return `code == 1`.
- `solve_multiperiod` call signature (keyword-only after `closure`): `solve_multiperiod(m, params, closure, ref_gdx=REF, skip_base_solve=True, mute_welfare=True, seed_from_prior=False, holdfix_cd=True)`.
- Test marker: reuse the existing `gams` marker (registered in `pyproject.toml`; `--strict-markers` is on, so do NOT invent a new marker).
- The test imports `_diff_core` (`gams_levels`, `list_populated_vars`, `split_t`) from `scripts/gtap` via `sys.path.insert(0, str(ROOT / "scripts/gtap"))` — same as `test_gtap7_nl_parity.py`.
- `ROOT = Path(__file__).resolve().parents[3]`.
- Exclusion sets (from the session harness):
  - `SKIP = {"walras","ev","cv","uh","u","ug","us"}`
  - `RF = {"pfa","pfy","pm","pmcif","pefob","pwmg","pp","pdp","pmp","xwmg","xmgm","lambdamg","imptx","exptx"}`
  - `ALIAS = {"xa":"xaa","xd":"xda","xm":"xma","pp":"pp_rai","p":"p_rai","ytaxInd":"ytax_ind","ytaxind":"ytax_ind"}`
- Prefix strip: drop a leading `a_/c_/f_/r_` (`s[1]=="_" and s[0] in "acfr"`).

## File Structure

- `tests/fixtures/gtap7_altertax/<dataset>/out_altertax_ifsub{0,1}.gdx` — committed GAMS reference GDX (6 files, ~3.4 MB).
- `tests/templates/gtap/test_altertax_multiperiod_parity.py` — the parametrized solving parity test.
- `.github/workflows/tests.yml` — add a step to the existing `gams-tests` job.

---

### Task 1: Commit the GDX fixtures

**Files:**
- Create: `tests/fixtures/gtap7_altertax/gtap7_3x3/out_altertax_ifsub0.gdx`
- Create: `tests/fixtures/gtap7_altertax/gtap7_3x3/out_altertax_ifsub1.gdx`
- Create: `tests/fixtures/gtap7_altertax/gtap7_5x5/out_altertax_ifsub0.gdx`
- Create: `tests/fixtures/gtap7_altertax/gtap7_5x5/out_altertax_ifsub1.gdx`
- Create: `tests/fixtures/gtap7_altertax/gtap7_10x7/out_altertax_ifsub0.gdx`
- Create: `tests/fixtures/gtap7_altertax/gtap7_10x7/out_altertax_ifsub1.gdx`

**Interfaces:**
- Produces: the 6 fixture GDX paths that Task 2 reads via `FIXTURES_DIR / dataset / f"out_altertax_ifsub{0 if not if_sub else 1}.gdx"`.

- [ ] **Step 1: Copy the 6 reference GDX into the fixtures dir**

```bash
cd /Users/marmol/.superset/worktrees/b14cb643-ee65-449d-b3f0-be8003b60783/debug-gtap7-check-income
for ds in gtap7_3x3 gtap7_5x5 gtap7_10x7; do
  mkdir -p tests/fixtures/gtap7_altertax/$ds
  for m in 0 1; do
    cp /Users/marmol/proyectos2/equilibria_refs/${ds}_altertax_cd/out_altertax_ifsub${m}.gdx \
       tests/fixtures/gtap7_altertax/$ds/out_altertax_ifsub${m}.gdx
  done
done
```

- [ ] **Step 2: Verify all 6 exist and are non-empty**

Run:
```bash
ls -la tests/fixtures/gtap7_altertax/*/out_altertax_ifsub*.gdx
```
Expected: 6 files, each 250 KB – 1 MB, none zero bytes.

- [ ] **Step 3: Confirm not gitignored, then commit**

Run:
```bash
git check-ignore tests/fixtures/gtap7_altertax/gtap7_3x3/out_altertax_ifsub0.gdx; echo "exit=$?"
```
If `exit=0` (ignored), use `git add -f`; otherwise `git add`. Then:
```bash
git add -f tests/fixtures/gtap7_altertax/
git commit -m "test(gtap-altertax): add GDX fixtures for 3x3/5x5/10x7 ifSUB{0,1}"
```

---

### Task 2: The parametrized solving parity test

**Files:**
- Create: `tests/templates/gtap/test_altertax_multiperiod_parity.py`

**Interfaces:**
- Consumes: the 6 fixture GDX from Task 1.
- Produces: pytest cases `test_altertax_multiperiod_parity[gtap7_3x3-ifsub0]` … `[gtap7_10x7-ifsub1]` (6 total), each asserting convergence + match ≥98%.

- [ ] **Step 1: Write the full test module**

Create `tests/templates/gtap/test_altertax_multiperiod_parity.py` with exactly this content:

```python
"""GTAP altertax multi-period SOLVING parity gate (both ifSUB modes).

For each (dataset, ifSUB) this builds the multi-period model, seeds it from the
committed GAMS GDX fixture, solves base->check->shock via solve_multiperiod, and
asserts:
  1. all 3 periods converge (termination code == 1), and
  2. the shock-period real-cell match vs the GDX is >= 98%.

Unlike test_gtap7_nl_parity.py (a no-solve .nl coefficient diff), this catches
regressions that CONVERGE to wrong values (e.g. the save<0 bug that silently
dropped gtap7_3x4 to 94% while still reporting code=1).

The test SKIPS (not fails) when either the fixture GDX is missing or the local
PATH solver (path_capi_python) is unavailable -- so it self-skips on ubuntu-latest
and actually runs on the self-hosted gams-tests job.

Run:
    uv run pytest tests/templates/gtap/test_altertax_multiperiod_parity.py -v
    uv run pytest tests/templates/gtap/test_altertax_multiperiod_parity.py -v \
        -k "gtap7_3x3 and ifsub1"
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = ROOT / "tests/fixtures/gtap7_altertax"
DATASETS_DIR = ROOT / "datasets"

sys.path.insert(0, str(ROOT / "scripts/gtap"))

DATASETS = ["gtap7_3x3", "gtap7_5x5", "gtap7_10x7"]
MATCH_THRESHOLD = 98.0

# Exclusion sets (denominator), identical to the session measurement harness.
SKIP = {"walras", "ev", "cv", "uh", "u", "ug", "us"}
RF = {
    "pfa", "pfy", "pm", "pmcif", "pefob", "pwmg", "pp", "pdp", "pmp",
    "xwmg", "xmgm", "lambdamg", "imptx", "exptx",
}
ALIAS = {
    "xa": "xaa", "xd": "xda", "xm": "xma", "pp": "pp_rai", "p": "p_rai",
    "ytaxInd": "ytax_ind", "ytaxind": "ytax_ind",
}


def _has_path_solver() -> bool:
    """The PATH MCP solver lives in a local package, absent on ubuntu-latest."""
    return importlib.util.find_spec("path_capi_python") is not None


def _strip(s):
    if isinstance(s, str) and len(s) > 2 and s[1] == "_" and s[0] in "acfr":
        return s[2:]
    return s


def _fixture_gdx(dataset: str, if_sub: bool) -> Path:
    suffix = "ifsub1" if if_sub else "ifsub0"
    return FIXTURES_DIR / dataset / f"out_altertax_{suffix}.gdx"


def _solve_and_match(dataset: str, if_sub: bool):
    """Build, seed, solve, compare. Returns (codes_dict, match_pct, total)."""
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.altertax import apply_altertax_elasticities
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_model_multiperiod import (
        GTAPMultiPeriodModel,
        PERIODS,
    )
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
    from pyomo.environ import value as V
    from _diff_core import gams_levels, list_populated_vars, split_t

    ref = _fixture_gdx(dataset, if_sub)
    d = DATASETS_DIR / dataset
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    rr = list(p.sets.r)[-1]
    pa = apply_altertax_elasticities(p, in_place=False)
    ac = GTAPClosureConfig(
        name="altertax", closure_type="MCP", capital_mobility="mobile",
        fix_endowments=False, fix_taxes=True, fix_technology=True,
        if_sub=if_sub, numeraire="pnum",
    )
    mp = GTAPMultiPeriodModel(pa.sets, pa, ac, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    for per in PERIODS:
        mp.build_equations_intra(m, per)
    mp.build_equations_fisher(m)
    m._residual_region = rr
    mp.seed_all_periods(m, ref)

    res = solve_multiperiod(
        m, p, ac, ref_gdx=ref,
        skip_base_solve=True, mute_welfare=True,
        seed_from_prior=False, holdfix_cd=True,
    )
    codes = {k: res[k]["code"] for k in res}

    tot = match = 0
    for vn in list_populated_vars(ref):
        if vn.lower() in SKIP or vn.lower() in RF:
            continue
        try:
            g = gams_levels(ref, vn)
        except Exception:
            continue
        pv = getattr(m, ALIAS.get(vn, vn), None) or getattr(m, vn.lower(), None)
        if pv is None:
            continue
        for fk, gval in g.items():
            try:
                body, t = split_t(fk)
            except Exception:
                continue
            if t != "shock":
                continue
            st = tuple(_strip(x) for x in body)
            if not st:
                idx = ("shock",)
            elif len(st) == 1:
                idx = (st[0], "shock")
            else:
                idx = (*st, "shock")
            val = None
            for cand in [idx, (*body, "shock") if body else ("shock",)]:
                try:
                    val = float(V(pv[cand]))
                    break
                except Exception:
                    pass
            if val is None:
                continue
            tot += 1
            d_abs = abs(val - gval)
            rel = d_abs / abs(gval) if abs(gval) > 1e-12 else (
                0.0 if d_abs < 1e-6 else 9e9
            )
            if d_abs <= 1e-6 or rel <= 1e-2:
                match += 1

    match_pct = 100.0 * match / max(tot, 1)
    return codes, match_pct, tot


@pytest.mark.gams
@pytest.mark.parametrize("if_sub", [False, True], ids=["ifsub0", "ifsub1"])
@pytest.mark.parametrize("dataset", DATASETS)
def test_altertax_multiperiod_parity(dataset: str, if_sub: bool) -> None:
    if not _has_path_solver():
        pytest.skip("path_capi_python (PATH solver) not available")
    ref = _fixture_gdx(dataset, if_sub)
    if not ref.exists():
        pytest.skip(f"fixture GDX missing: {ref}")
    if not (DATASETS_DIR / dataset / "basedata.har").exists():
        pytest.skip(f"dataset HAR missing for {dataset}")

    codes, match_pct, tot = _solve_and_match(dataset, if_sub)

    mode = "ifSUB=1" if if_sub else "ifSUB=0"
    assert all(c == 1 for c in codes.values()), (
        f"[{dataset}/{mode}] not all periods converged: {codes}"
    )
    assert tot > 0, f"[{dataset}/{mode}] no comparable cells found"
    assert match_pct >= MATCH_THRESHOLD, (
        f"[{dataset}/{mode}] real-cell match {match_pct:.2f}% "
        f"< {MATCH_THRESHOLD}% (over {tot} cells)"
    )
```

- [ ] **Step 2: Run the test to verify it PASSES (solver present locally)**

Run:
```bash
uv run pytest tests/templates/gtap/test_altertax_multiperiod_parity.py -v --tb=short
```
Expected: 6 PASSED (gtap7_3x3/5x5/10x7 × ifsub0/ifsub1). If `path_capi_python` is not importable in this venv, the 6 cases SKIP instead — in that case run with the solver on the path (the session ran solves all along, so it is available); a SKIP here is acceptable only if the solver is genuinely absent.

- [ ] **Step 3: Verify the assertion message is correct by spot-running one case**

Run:
```bash
uv run pytest "tests/templates/gtap/test_altertax_multiperiod_parity.py::test_altertax_multiperiod_parity[gtap7_5x5-ifsub1]" -v
```
Expected: PASS (5x5 ifSUB=1 measured ~99.5%).

- [ ] **Step 4: Commit**

```bash
git add tests/templates/gtap/test_altertax_multiperiod_parity.py
git commit -m "test(gtap-altertax): solving parity gate for 3x3/5x5/10x7 ifSUB{0,1}"
```

---

### Task 3: Wire into the self-hosted CI job

**Files:**
- Modify: `.github/workflows/tests.yml` (the `gams-tests` job)

**Interfaces:**
- Consumes: the test module from Task 2.
- Produces: an extra step in the manual `gams-tests` job that runs the altertax parity tests on the self-hosted runner.

- [ ] **Step 1: Read the gams-tests job to find the last step**

Run:
```bash
grep -n "gams-tests:" .github/workflows/tests.yml
sed -n '/gams-tests:/,/^  [a-z]/p' .github/workflows/tests.yml
```
Note the indentation and the last `- name:` step in that job.

- [ ] **Step 2: Add the altertax parity step at the end of the gams-tests job's `steps:`**

Append (matching the existing step indentation) after the last step in the `gams-tests` job:

```yaml
      - name: Run altertax multi-period parity tests
        run: |
          uv run pytest tests/templates/gtap/test_altertax_multiperiod_parity.py -v \
            --tb=short
```

- [ ] **Step 3: Validate the workflow YAML parses**

Run:
```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/tests.yml')); print('YAML OK')"
```
Expected: `YAML OK`

- [ ] **Step 4: Confirm the new step is inside gams-tests (not a new job)**

Run:
```bash
grep -n "altertax multi-period parity" .github/workflows/tests.yml
```
Expected: one line, indented under `gams-tests` steps (6 spaces for `- name:`).

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/tests.yml
git commit -m "ci(gtap-altertax): run multi-period parity tests in gams-tests job"
```

---

### Task 4: Non-regression sanity + final verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full altertax parity suite once more**

Run:
```bash
uv run pytest tests/templates/gtap/test_altertax_multiperiod_parity.py -v --tb=short
```
Expected: 6 PASSED (or 6 SKIPPED if solver absent — but locally the solver is present, so expect PASSED).

- [ ] **Step 2: Confirm the marker selection works (deselect on a non-gams run)**

Run:
```bash
uv run pytest tests/templates/gtap/test_altertax_multiperiod_parity.py -v -m "not gams"
```
Expected: 6 DESELECTED (the `gams` marker excludes them from a normal run).

- [ ] **Step 3: Confirm the .nl gate still passes (untouched, sanity)**

Run:
```bash
uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py -q
```
Expected: 5 passed.

- [ ] **Step 4: Confirm git is clean and fixtures are tracked**

Run:
```bash
git status --short
git ls-files tests/fixtures/gtap7_altertax/ | wc -l
```
Expected: clean tree (after commits); `6` tracked fixture files.

---

## Self-Review notes

- **Spec coverage:** matrix (3×2), threshold 98%, convergence code=1, exclusion sets, skip-on-missing-solver/fixture, self-hosted wiring, `gams` marker reuse, fixtures dir — all present (Tasks 1–3); verification in Task 4.
- **Placeholders:** none — full test code in Task 2 Step 1, exact YAML in Task 3 Step 2.
- **Type/name consistency:** `_solve_and_match` returns `(codes, match_pct, tot)` and the test unpacks exactly those three; `_fixture_gdx(dataset, if_sub)` used in both helper and test; `solve_multiperiod` kwargs match the verified signature; marker `gams` matches the registered one.
