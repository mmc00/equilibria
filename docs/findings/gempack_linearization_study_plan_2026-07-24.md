# Against-GEMPACK Linearization Study — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the mac-side tooling for the 4-evidence against-GEMPACK linearization study (shock sweep + Gragg refinement + ifSUB fidelity gate + welfare from decomp.har), so the ~50-run GEMPACK grid can be driven on Windows and the results consolidated into a docs page + finding.

**Architecture:** Extend existing infra (`run_gempack_matrix.py`, `gempack_reference.py`) rather than greenfield. Two Python-side fidelity gates run on mac *before* the Windows phase; the runner emits per-(dataset×config) `.cmf` files with unique names so the grid doesn't overwrite itself; the comparator gains a welfare reader and a per-config table emitter; a generator produces the docs page.

**Tech Stack:** Python 3, Pyomo (existing GTAP model), `equilibria.babel.har.reader` (HAR I/O), pytest (`-m integration` for solve-based tests), the GTAP multiperiod driver.

## Global Constraints

- **Spec:** `docs/findings/gempack_linearization_study_spec_2026-07-24.md` — this plan implements it.
- **FIDELITY IS SUPREME.** Never inflate a match% by excluding divergent cells or forcing a value; GAMS/GEMPACK is the source of truth. A gate that fails means fix the cause, not the gate.
- **Metric:** absolute percentage points on %-changes (`|Δ(%change)| ≤ 1pp`), NOT relative tol. GEMPACK output is %-change; Python %-change = `s/b - 1`.
- **gitignore allowlist:** `scripts/gtap/*` is ignored by default. Every NEW script under `scripts/gtap/` needs a `!scripts/gtap/<file>.py` line added to `.gitignore` or it won't be tracked.
- **Datasets:** `gtap7_3x3`, `gtap7_3x4`, `gtap7_5x5`, `gtap7_10x7`, `gtap7_15x10`. (`gtap7_20x41` is known-blocked in GEMPACK — attempted for the record, no fixture expected.)
- **prek pre-commit** runs ruff (format + lint --fix) and a ty ratchet on `^(src|tests)/`. Scripts under `scripts/` are outside the ty scope but ruff still formats staged Python. Expect the double-commit pattern if ruff reformats.
- **Parity gates stamp:** touching `scripts/gtap/*` or `tests/...` invalidates the stamp; a mandatory `run_parity_gates.py` sweep + stamp is required before any push/PR (enforced by the PreToolUse hook). Doc-only edits do not.
- **No model-equation changes** in this plan. If Gate 3 (Task 1) exposes an ifSUB condensation bug, STOP and open a separate debugging task via the `equilibria-parity-debug` cascade — do not patch the model inside this study.

---

### Task 1: ifSUB fidelity gate (evidence 3, mac gate #1)

The hardest fidelity gate, and it runs entirely on mac. In Python, solve each dataset's shock at `ifSUB=1` and `ifSUB=0` and assert the post-shock **levels** agree cell-by-cell. `ifSUB` is model condensation (van der Mensbrugghe Table D.1) — it must NOT change the economics, so the two solves must land on the same levels point. If they diverge, condensation has a bug → blocks the whole study.

**Files:**
- Create: `scripts/gtap/verify_ifsub_equivalence.py`
- Modify: `.gitignore` (add `!scripts/gtap/verify_ifsub_equivalence.py`)
- Test: `tests/templates/gtap/test_ifsub_equivalence.py`

**Interfaces:**
- Consumes: `test_gtap7_gempack_parity._solve_shock(dataset, ifsub) -> (model, code)` pattern (replicate its build/seed/solve sequence; it is a test-module function, so the script re-implements the same sequence rather than importing it).
- Produces: `verify_ifsub_equivalence.compare_levels(dataset: str, tol_rel: float = 1e-4) -> dict` returning `{"n_cells": int, "n_agree": int, "frac_agree": float, "worst": list[tuple[str, tuple, float, float, float]]}` where `worst` is the top divergent `(varname, key, v_if1, v_if0, rel)` rows. A `main()` prints a per-dataset table and exits non-zero if any dataset is below the floor.

- [ ] **Step 1: Write the failing test**

```python
# tests/templates/gtap/test_ifsub_equivalence.py
"""ifSUB is model condensation (van der Mensbrugghe Table D.1), not economics —
Python ifSUB=1 and ifSUB=0 must solve to the SAME post-shock levels. LOCAL-only,
gated on the shock GDX fixtures being present."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/gtap"))
sys.path.insert(0, str(ROOT / "src"))

DATASETS = ["gtap7_3x3", "gtap7_3x4", "gtap7_5x5", "gtap7_10x7", "gtap7_15x10"]
FLOOR = 0.999  # 99.9% of level cells agree at rel 1e-4


@pytest.mark.parametrize("dataset", DATASETS)
def test_ifsub_levels_equivalence(dataset):
    from verify_ifsub_equivalence import compare_levels
    d = ROOT / "datasets" / dataset / "basedata.har"
    g1 = ROOT / f"tests/fixtures/gtap7/{dataset}/out_gtap_shock_ifsub1.gdx"
    g0 = ROOT / f"tests/fixtures/gtap7/{dataset}/out_gtap_shock_ifsub0.gdx"
    if not (d.exists() and g1.exists() and g0.exists()):
        pytest.skip(f"missing inputs for {dataset}")
    r = compare_levels(dataset, tol_rel=1e-4)
    assert r["frac_agree"] >= FLOOR, (
        f"[{dataset}] ifSUB=1 vs ifSUB=0 levels agree only "
        f"{r['frac_agree']*100:.2f}% < {FLOOR*100:.1f}% — condensation bug. "
        f"worst: {r['worst'][:5]}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/templates/gtap/test_ifsub_equivalence.py -v -m integration`
Expected: FAIL/ERROR with `ModuleNotFoundError: verify_ifsub_equivalence` (or skips if GDX fixtures absent — if it skips everywhere, note that and proceed; the script is still needed for the CLI gate).

- [ ] **Step 3: Write the implementation**

```python
# scripts/gtap/verify_ifsub_equivalence.py
"""Verify Python ifSUB=1 ≡ ifSUB=0 in post-shock LEVELS, per dataset.

ifSUB ("if SUBstitution", van der Mensbrugghe Table D.1) condenses the model by
substituting variables out as linear expressions — it must NOT change the
economics. This is mac gate #1 of the against-GEMPACK linearization study: if the
two modes diverge, condensation has a bug and the study pauses to fix it.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DATASETS = ["gtap7_3x3", "gtap7_3x4", "gtap7_5x5", "gtap7_10x7", "gtap7_15x10"]


def _solve(dataset: str, ifsub: int):
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_model_multiperiod import (
        PERIODS, GTAPMultiPeriodModel,
    )
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    d = ROOT / "datasets" / dataset
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har", sets_path=d / "sets.har",
        default_path=d / "default.prm", baserate_path=d / "baserate.har",
    )
    rr = list(p.sets.r)[-1]
    ac = GTAPClosureConfig(
        name="base", closure_type="MCP", capital_mobility="sluggish",
        fix_endowments=False, fix_taxes=False, fix_technology=False,
        if_sub=bool(ifsub), numeraire="pnum",
    )
    gdx = ROOT / f"tests/fixtures/gtap7/{dataset}/out_gtap_shock_ifsub{ifsub}.gdx"
    mp = GTAPMultiPeriodModel(p.sets, p, ac, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    for per in PERIODS:
        mp.build_equations_intra(m, per)
    mp.build_equations_fisher(m)
    m._residual_region = rr
    mp.seed_all_periods(m, gdx)
    res = solve_multiperiod(
        m, p, ac, ref_gdx=gdx, skip_base_solve=True, mute_welfare=True,
        seed_from_prior=False, holdfix_cd=True, mode="gtap",
    )
    return m, int(res["shock"]["code"])


def _shock_levels(m) -> dict[tuple[str, tuple], float]:
    """All Var 'shock'-slice values, keyed by (varname, body-index)."""
    from pyomo.environ import Var, value as V
    out: dict[tuple[str, tuple], float] = {}
    for v in m.component_objects(Var, active=True):
        bn = v.name.split("[")[0]
        for idx in v:
            if not (isinstance(idx, tuple) and idx and idx[-1] == "shock"):
                continue
            try:
                out[(bn, idx[:-1])] = float(V(v[idx]))
            except (ValueError, TypeError):
                continue
    return out


def compare_levels(dataset: str, tol_rel: float = 1e-4) -> dict:
    m1, c1 = _solve(dataset, 1)
    m0, c0 = _solve(dataset, 0)
    if c1 != 1 or c0 != 1:
        return {"n_cells": 0, "n_agree": 0, "frac_agree": 0.0,
                "worst": [("SOLVE", ("code",), float(c1), float(c0), 9e9)]}
    L1, L0 = _shock_levels(m1), _shock_levels(m0)
    common = L1.keys() & L0.keys()
    worst: list[tuple[str, tuple, float, float, float]] = []
    n_agree = 0
    for k in common:
        a, b = L1[k], L0[k]
        denom = max(abs(a), abs(b), 1e-9)
        rel = abs(a - b) / denom
        if rel <= tol_rel:
            n_agree += 1
        else:
            worst.append((k[0], k[1], a, b, rel))
    worst.sort(key=lambda t: -t[4])
    n = len(common)
    return {"n_cells": n, "n_agree": n_agree,
            "frac_agree": (n_agree / n if n else 0.0), "worst": worst[:20]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--tol-rel", type=float, default=1e-4)
    args = ap.parse_args()
    bad = []
    print(f"{'dataset':14s} {'cells':>7s} {'agree%':>8s}  worst")
    for ds in args.datasets:
        if not (ROOT / "datasets" / ds / "basedata.har").exists():
            print(f"{ds:14s} {'--':>7s} {'skip':>8s}  (no dataset HAR)")
            continue
        r = compare_levels(ds, args.tol_rel)
        w = r["worst"][0] if r["worst"] else None
        wtxt = f"{w[0]}{w[1]} rel={w[4]:.1e}" if w else "-"
        flag = "" if r["frac_agree"] >= 0.999 else "  <<< BELOW 99.9%"
        print(f"{ds:14s} {r['n_cells']:7d} {r['frac_agree']*100:7.2f}%  {wtxt}{flag}")
        if r["frac_agree"] < 0.999:
            bad.append(ds)
    if bad:
        print(f"\nFAIL: ifSUB condensation diverges on {bad} — investigate before Windows.")
        return 1
    print("\nOK: ifSUB=1 ≡ ifSUB=0 in levels on all datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add the gitignore allowlist entry**

Add near the other `!scripts/gtap/` lines in `.gitignore`:
```
!scripts/gtap/verify_ifsub_equivalence.py
```

- [ ] **Step 5: Run the CLI gate on the smallest dataset first**

Run: `.venv/bin/python scripts/gtap/verify_ifsub_equivalence.py --datasets gtap7_3x3`
Expected: a table row for gtap7_3x3 with `agree% ≥ 99.90%` and exit 0. **If it is below 99.9%, STOP** — this is a real condensation-fidelity finding; open a debugging task (`equilibria-parity-debug`, drift-test / closure-diff class) and do not proceed to Windows. Record the worst cells.

- [ ] **Step 6: Run the full gate + the test**

Run: `.venv/bin/python scripts/gtap/verify_ifsub_equivalence.py`
Run: `.venv/bin/python -m pytest tests/templates/gtap/test_ifsub_equivalence.py -v -m integration`
Expected: all 5 datasets ≥ 99.9%, exit 0; pytest PASS (or skip where GDX fixtures are absent).

- [ ] **Step 7: Commit**

```bash
git add scripts/gtap/verify_ifsub_equivalence.py tests/templates/gtap/test_ifsub_equivalence.py .gitignore
git commit -m "gtap(F5): ifSUB fidelity gate — Python if1 ≡ if0 in levels (mac gate #1)"
```

---

### Task 2: Parameterize the runner for the shock×config grid

The runner writes a single fixed `tm10.cmf` per dataset and hardcodes `Steps = 8 16 32`. For the grid we need (a) a `--steps` flag (Gragg-refinement axis, evidence 2), (b) unique `.cmf` + output names per (dataset×config) so the 50-run grid does not overwrite itself, (c) a batch generator that lays out the whole grid.

**Files:**
- Modify: `scripts/gtap/run_gempack_matrix.py:82` (`make_cmf` signature + `Steps` line), `:124` (`prepare`), `:182` (`main` args + grid loop)
- Test: `tests/templates/gtap/test_gempack_runner_grid.py`

**Interfaces:**
- Consumes: `regions(ds_dir) -> list[str]` (unchanged).
- Produces:
  - `make_cmf(name, regs, shock_pct=10.0, steps="8 16 32", updated_name="updated.har") -> str` — `steps` becomes the `Steps = {steps} ;` line; `updated_name` becomes the `Updated file GTAPDATA = {updated_name} ;` line so each config writes a distinct updated HAR.
  - `config_tag(shock_pct, steps) -> str` — stable filename tag, e.g. `tm10_s8-16-32` → `tm3` / `s4-8-16-32-64`; used to name `.cmf` and outputs.
  - `prepare(name, shock_pct=10.0, steps="8 16 32") -> Path` — writes `<tag>.cmf` (not the fixed `tm10.cmf`) into the run dir.

- [ ] **Step 1: Write the failing test**

```python
# tests/templates/gtap/test_gempack_runner_grid.py
"""The GEMPACK runner must emit distinct .cmf content per (shock, steps) config
so the sweep grid does not overwrite itself."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/gtap"))
sys.path.insert(0, str(ROOT / "src"))


def test_shock_pct_appears_in_cmf():
    from run_gempack_matrix import make_cmf
    regs = ["USA", "EU", "ROW"]
    c3 = make_cmf("gtap7_3x3", regs, shock_pct=3.0)
    assert "Shock tm = uniform 3 ;" in c3
    assert "uniform 3%" in c3


def test_steps_flag_drives_gragg_line():
    from run_gempack_matrix import make_cmf
    regs = ["USA", "EU", "ROW"]
    c = make_cmf("gtap7_3x3", regs, shock_pct=10.0, steps="4 8 16 32 64")
    assert "Steps  = 4 8 16 32 64 ;" in c
    assert "Steps  = 8 16 32 ;" not in c


def test_updated_name_is_config_specific():
    from run_gempack_matrix import make_cmf, config_tag
    regs = ["USA", "EU", "ROW"]
    tag = config_tag(3.0, "8 16 32")
    c = make_cmf("gtap7_3x3", regs, shock_pct=3.0, updated_name=f"updated_{tag}.har")
    assert f"Updated file GTAPDATA = updated_{tag}.har ;" in c


def test_config_tag_is_stable_and_distinct():
    from run_gempack_matrix import config_tag
    assert config_tag(10.0, "8 16 32") != config_tag(3.0, "8 16 32")
    assert config_tag(10.0, "8 16 32") != config_tag(10.0, "4 8 16 32 64")
    # deterministic
    assert config_tag(0.1, "8 16 32") == config_tag(0.1, "8 16 32")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/templates/gtap/test_gempack_runner_grid.py -v`
Expected: FAIL — `make_cmf` has no `steps`/`updated_name` kwargs; `config_tag` does not exist.

- [ ] **Step 3: Implement `config_tag` and extend `make_cmf`**

In `scripts/gtap/run_gempack_matrix.py`, add above `make_cmf`:

```python
def config_tag(shock_pct: float, steps: str) -> str:
    """Stable, filesystem-safe tag for a (shock, steps) config.
    e.g. (10, "8 16 32") -> "tm10_s8-16-32"; (0.1, ...) -> "tm0p1_s...".
    """
    s = f"{shock_pct:g}".replace(".", "p")
    st = steps.replace(" ", "-")
    return f"tm{s}_s{st}"
```

Change `make_cmf`'s signature and the two affected lines:

```python
def make_cmf(name: str, regs: list[str], shock_pct: float = 10.0,
             steps: str = "8 16 32", updated_name: str = "updated.har") -> str:
    ...
    # was: Updated file GTAPDATA = updated.har ;
    Updated file GTAPDATA = {updated_name} ;
    ...
    # was: Steps  = 8 16 32 ;
    Steps  = {steps} ;
```

(Interpolate `updated_name` and `steps` into the f-string. Leave everything else — EXOG_BLOCK, swaps, Shock line — unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/templates/gtap/test_gempack_runner_grid.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Extend `prepare` + `main` to lay out the grid**

Change `prepare` to write a config-tagged `.cmf` and return the tag too:

```python
def prepare(name, shock_pct=10.0, steps="8 16 32"):
    ds_dir = DATA_DIR / name
    regs = regions(ds_dir)
    run_dir = RUN_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in INPUT_FILES:
        shutil.copy2(ds_dir / f, run_dir / f)
    tag = config_tag(shock_pct, steps)
    cmf = make_cmf(name, regs, shock_pct, steps, updated_name=f"updated_{tag}.har")
    (run_dir / f"{tag}.cmf").write_text(cmf, encoding="ascii")
    print(f"  {name:14s} {tag:22s} residual={regs[-1]:<10s} regions={len(regs):2d}")
    return run_dir, tag
```

In `main`, add a `--grid` mode that expands the two study axes and calls `prepare` per config. Add args:

```python
ap.add_argument("--grid", action="store_true",
                help="emit the full linearization-study grid: shock sweep "
                     "(10/3/1/0.3/0.1%% at default steps) + Gragg sweep "
                     "(steps 4/8/16/32/64 at 10%%). Overrides --shock-pct/--steps.")
ap.add_argument("--steps", type=str, default="8 16 32",
                help="Gragg Steps line for a single-config run (default '8 16 32')")
SWEEP_SHOCKS = [10.0, 3.0, 1.0, 0.3, 0.1]
GRAGG_STEPS = ["4", "8", "16", "32", "64"]
```

Grid expansion (in `main`, before the solve loop):

```python
if args.grid:
    configs = [(s, "8 16 32") for s in SWEEP_SHOCKS] + \
              [(10.0, st) for st in GRAGG_STEPS]
    # de-dup the (10.0, "8 16 32") overlap if present
    configs = list(dict.fromkeys(configs))
else:
    configs = [(args.shock_pct, args.steps)]

jobs = []  # (name, run_dir, tag, shock_pct, steps)
for name in args.datasets:
    for shock_pct, steps in configs:
        run_dir, tag = prepare(name, shock_pct, steps)
        jobs.append((name, run_dir, tag, shock_pct, steps))
```

Update the solve loop to iterate `jobs`, solve each with `-cmf {tag}.cmf`, and collect `updated_{tag}.har` → `FIXTURES/updated_{name}_{tag}.har` (and the sl4 → `sl4dump_{name}_{tag}.har`). Solve/convert take the `.cmf`/output names from the tag. Keep the SKIP-if-no-gtapv7 behaviour.

- [ ] **Step 6: Verify grid generation on mac (no-solve, no GEMPACK needed)**

Run: `.venv/bin/python scripts/gtap/run_gempack_matrix.py --grid --no-solve --datasets gtap7_3x3`
Expected: prints 9 config rows for gtap7_3x3 (5 shock + 5 gragg − 1 dedup overlap = 9), each with a distinct tag; 9 `.cmf` files land in `runs/gempack_matrix/gtap7_3x3/`. Inspect two: `tm10_s8-16-32.cmf` and `tm0p1_s8-16-32.cmf` differ only in the `Shock` line + `uniform` comment; `tm10_s4-8-16-32-64.cmf` differs only in the `Steps` line.

- [ ] **Step 7: Commit**

```bash
git add scripts/gtap/run_gempack_matrix.py tests/templates/gtap/test_gempack_runner_grid.py
git commit -m "gtap(F5): runner --grid — shock sweep + Gragg refinement, config-tagged .cmf"
```

---

### Task 3: Welfare reader from decomp.har (evidence 4)

Add a welfare reader to the comparator: read GEMPACK's `decomp.har` (WELVIEW) EV decomposition and expose it next to Python's EV. Diagnostic only — no floor (welfare `u` is a sign-flipping second-order quantity, per `gempack_welfare_not_cellwise_2026-07-23.md`).

**Files:**
- Modify: `scripts/gtap/gempack_reference.py` (add reader function near `gempack_levels`)
- Test: `tests/templates/gtap/test_gempack_welfare_reader.py`

**Interfaces:**
- Consumes: `gempack_reference._cells(header)` and `read_har` (already imported in the module).
- Produces: `gempack_welfare_ev(decomp_har_path: str) -> dict[str, dict[str, float]]` returning `{region: {branch: ev_dollars}}` for branches `{"alloc", "tot", "invsav", "total"}` — reading the WELVIEW decomposition header by NAME (defensive; RunGTAP versions differ), summing sub-components into the 3 canonical branches, or returning `{}` if the header is absent.

- [ ] **Step 1: Inspect a real decomp.har header layout (discovery, not a test)**

If a `decomp.har` exists from an earlier run, list its headers:
```bash
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); from equilibria.babel.har.reader import read_har; import glob; \
h=read_har(glob.glob('runs/**/decomp.har', recursive=True)[0]); print(list(h.keys()))"
```
If none exists yet (likely — decomp.har is Windows-produced), skip this step and write the reader defensively against the documented WELVIEW structure (regional EV with allocative-efficiency / terms-of-trade / investment-savings components, the Huff–McDougall decomposition). The test uses a synthetic HAR so it does not depend on a real file.

- [ ] **Step 2: Write the failing test (synthetic decomp HAR)**

```python
# tests/templates/gtap/test_gempack_welfare_reader.py
"""The welfare reader pulls the EV decomposition from a decomp.har (WELVIEW) by
header NAME, sums into 3 canonical branches, and is a clean no-op when absent."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/gtap"))
sys.path.insert(0, str(ROOT / "src"))


def _write_synth_decomp(path: Path):
    """Minimal HAR with a named EV header (regions × components)."""
    from equilibria.babel.har.writer import write_har  # if a writer exists
    # regions R1,R2 ; components alloc,tot,invsav ; EV$ values
    arr = np.array([[1.0, -0.5, 0.2], [-0.3, 0.4, 0.1]], dtype="float32")
    write_har(str(path), {"A": {"array": arr,
              "sets": [["R1", "R2"], ["alloc", "tot", "invsav"]]}})


def test_welfare_reader_absent_is_empty():
    from gempack_reference import gempack_welfare_ev
    assert gempack_welfare_ev(str(ROOT / "does_not_exist.har")) == {}


def test_welfare_reader_sums_branches(tmp_path):
    pytest.importorskip("equilibria.babel.har.writer")
    from gempack_reference import gempack_welfare_ev
    p = tmp_path / "decomp.har"
    _write_synth_decomp(p)
    ev = gempack_welfare_ev(str(p))
    # returns per-region branch dict with a 'total' == sum of branches
    assert set(ev) == {"R1", "R2"}
    assert ev["R1"]["total"] == pytest.approx(1.0 - 0.5 + 0.2, abs=1e-5)
```

Note: if `equilibria.babel.har.writer` does not exist, replace `_write_synth_decomp` with a fixture-based test that `pytest.skip`s when no real `decomp.har` fixture is present, and keep only `test_welfare_reader_absent_is_empty` as the always-on assertion. Check for the writer in Step 3 before finalizing.

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/templates/gtap/test_gempack_welfare_reader.py -v`
Expected: FAIL — `gempack_welfare_ev` does not exist.

- [ ] **Step 4: Implement the reader**

```python
# in scripts/gtap/gempack_reference.py, after gempack_levels(...)
# Candidate WELVIEW header names for the regional EV decomposition, most specific
# first. RunGTAP versions differ; read by name and fall back gracefully.
_EV_HEADERS = ("A", "EV", "WEV", "CNTeqEV", "TOTe")
_BRANCH_ALIASES = {
    "alloc": ("alloc", "allocative", "alle"),
    "tot":   ("tot", "terms", "termsoftrade", "tote"),
    "invsav": ("invsav", "savinv", "is", "cgds"),
}


def gempack_welfare_ev(decomp_har_path: str) -> dict[str, dict[str, float]]:
    """Read the regional EV decomposition ($) from a decomp.har (WELVIEW).

    Returns {region: {"alloc": .., "tot": .., "invsav": .., "total": ..}}.
    Empty dict if the file or a recognizable EV header is absent (diagnostic use;
    welfare is deliberately not a floor-gate — see gempack_welfare_not_cellwise).
    """
    from pathlib import Path as _P
    if not _P(decomp_har_path).exists():
        return {}
    try:
        h = read_har(decomp_har_path)
    except Exception:
        return {}
    hdr = next((h[k] for k in _EV_HEADERS if k in h), None)
    if hdr is None:
        return {}
    arr = hdr.array
    sets = getattr(hdr, "sets", None)
    if sets is None or arr.ndim != 2:
        return {}
    regions = [str(x).strip() for x in sets[0]]
    comps = [str(x).strip().lower() for x in sets[1]]

    def branch_of(comp: str) -> str | None:
        for br, aliases in _BRANCH_ALIASES.items():
            if any(comp.startswith(a) for a in aliases):
                return br
        return None

    out: dict[str, dict[str, float]] = {}
    for ri, reg in enumerate(regions):
        d = {"alloc": 0.0, "tot": 0.0, "invsav": 0.0}
        for ci, comp in enumerate(comps):
            br = branch_of(comp)
            if br:
                d[br] += float(arr[ri, ci])
        d["total"] = d["alloc"] + d["tot"] + d["invsav"]
        out[reg] = d
    return out
```

Adjust `_EV_HEADERS` / `_BRANCH_ALIASES` in Step 1's discovery once a real `decomp.har` is available; the synthetic test only exercises header `"A"` with the canonical component names.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/templates/gtap/test_gempack_welfare_reader.py -v`
Expected: PASS (2 tests, or 1 pass + reader test skipped if no HAR writer).

- [ ] **Step 6: Commit**

```bash
git add scripts/gtap/gempack_reference.py tests/templates/gtap/test_gempack_welfare_reader.py
git commit -m "gtap(F5): welfare EV reader from decomp.har (WELVIEW), name-based + defensive"
```

---

### Task 4: Consolidation report — the docs page generator

Produce the study's output page from whatever grid fixtures are present. Reads `updated_{ds}_{tag}.har` / `sl4dump_{ds}_{tag}.har`, solves Python once per (dataset, shock) to get the pp match, and emits a markdown page with: per-dataset shock-sweep curve, Gragg-convergence table, measured non-linearity column, and the welfare diagnostic section. When a fixture is absent, the cell reads "—" (pending Windows) — the page always generates.

**Files:**
- Create: `scripts/gtap/gen_linearization_study.py`
- Modify: `.gitignore` (add `!scripts/gtap/gen_linearization_study.py`)
- Create (generated): `docs/site/guide/gtap7_gempack_linearization_study.md`
- Test: `tests/templates/gtap/test_linearization_study_gen.py`

**Interfaces:**
- Consumes: `run_gempack_matrix.config_tag`, `gempack_reference.gempack_qty_pct` / `Q_TO_VAR` / `gempack_welfare_ev`, and the `_solve_shock`/`_measure_pp` pattern (re-implemented locally, taking a shock-pct argument so it can solve Python at each sweep magnitude).
- Produces: `gen_linearization_study.build_page(fixtures_dir: Path, out_md: Path) -> str` returning the markdown text (also writing it). A `main()` wraps it.

- [ ] **Step 1: Write the failing test**

```python
# tests/templates/gtap/test_linearization_study_gen.py
"""The study page generator must always produce a page (marking absent fixtures
'—'), list the compared variables, and cite the source."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/gtap"))
sys.path.insert(0, str(ROOT / "src"))


def test_page_generates_with_no_fixtures(tmp_path):
    from gen_linearization_study import build_page
    empty = tmp_path / "fixtures"
    empty.mkdir()
    out = tmp_path / "page.md"
    md = build_page(empty, out)
    assert out.exists()
    # scope + provenance always present
    assert "linearization" in md.lower()
    assert "van der Mensbrugghe" in md
    # the shock-sweep axis is documented even with no data
    for pct in ("10", "3", "1", "0.3", "0.1"):
        assert pct in md
    # absent data is an em-dash, not a crash or a fake number
    assert "—" in md


def test_variable_list_is_shown(tmp_path):
    from gen_linearization_study import build_page
    out = tmp_path / "page.md"
    md = build_page(tmp_path, out)
    # the 15 mapped quantity vars are named
    for gv in ("qfd", "qxs", "qgdp", "qva"):
        assert gv in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/templates/gtap/test_linearization_study_gen.py -v`
Expected: FAIL — `gen_linearization_study` does not exist.

- [ ] **Step 3: Implement the generator**

Write `scripts/gtap/gen_linearization_study.py` with `build_page(fixtures_dir, out_md)`:
- Header + scope note (5 datasets; nus333/9x10 out) + provenance (van der Mensbrugghe *Standard GTAP Model in GAMS v7*, Table D.1; ifSUB = condensation; Horridge SIMPLE 100% reference from PR #40).
- A "Variables compared" section listing the 15 `Q_TO_VAR` keys with their Python var + measured base pp.
- **Shock-sweep table:** rows = datasets, columns = `10 / 3 / 1 / 0.3 / 0.1 %`, cells = `within-1pp %` (from solving Python at that shock vs `sl4dump_{ds}_{tag}.har`); a trailing **non-linearity** column = `|match(10%) − match(0.1%)|`. Absent fixture → "—".
- **Gragg-convergence table:** rows = datasets, columns = `Steps 4/8/16/32/64` at 10%, cells = within-1pp %. Absent → "—".
- **Welfare section:** for each dataset with a `decomp.har`/EV fixture, print `gempack_welfare_ev` EV$ by region+branch next to Python EV, with the explicit sign-flip caveat and a link to `gempack_welfare_not_cellwise_2026-07-23.md`. No floor.
- The Python solve per (dataset, shock) reuses the `_solve_shock` sequence but must apply the shock magnitude; if driving Python at arbitrary shock pct is not wired, solve only at the 10% baseline the model already supports and mark other shock columns "— (py-shock pending)" — do NOT fabricate. Note this limitation in the page.

Keep the generator import-light: it must `build_page` an empty dir without importing Pyomo (guard the solve behind "fixtures present").

- [ ] **Step 4: Add the gitignore allowlist entry**

```
!scripts/gtap/gen_linearization_study.py
```

- [ ] **Step 5: Run test + generate the page**

Run: `.venv/bin/python -m pytest tests/templates/gtap/test_linearization_study_gen.py -v`
Expected: PASS (2 tests).
Run: `.venv/bin/python scripts/gtap/gen_linearization_study.py`
Expected: writes `docs/site/guide/gtap7_gempack_linearization_study.md` with all cells "—" (no Windows fixtures yet) but full structure, variable list, and provenance.

- [ ] **Step 6: Wire the page into the docs toctree**

Add `gtap7_gempack_linearization_study` to the guide index/toctree next to `gtap7_coverage_matrix_gempack` (find it: `grep -rn "gtap7_coverage_matrix_gempack" docs/site`). Verify docs build:
Run: `.venv/bin/python -m sphinx -b html -W docs/site docs/site/_build/html 2>&1 | tail -20` (or the repo's `make -C docs/site html` equivalent).
Expected: build succeeds; the new page appears.

- [ ] **Step 7: Commit**

```bash
git add scripts/gtap/gen_linearization_study.py docs/site/guide/gtap7_gempack_linearization_study.md .gitignore docs/site/<index-touched>
git commit -m "gtap(F5): linearization-study page generator + skeleton page (cells pending Windows)"
```

---

### Task 5: Windows batch driver + the reproduction guide

Emit the single `.bat` the user runs on Windows to drive the whole grid (solve every `.cmf`, run sltoht, leave the fixtures where the generator reads them), plus a short guide section on the round-trip.

**Files:**
- Create: `runs/gempack_matrix/run_study_windows.bat` (generated by the runner's `--grid --no-solve` on Windows, or committed as a static driver)
- Modify: `runs/gempack_updated_har_guide.md` (add a "§10 linearization study" section)

**Interfaces:**
- Consumes: the config-tagged `.cmf` files from Task 2's `--grid` generation.
- Produces: nothing importable — an operator artifact.

- [ ] **Step 1: Add a `--emit-bat` option to the runner (or write the static .bat)**

Simplest: extend `run_gempack_matrix.py`'s `--grid --no-solve` path to also write `run_study_windows.bat` into `RUN_ROOT`, looping over every generated `(dataset, tag)` and calling `gtapv7.exe -cmf <tag>.cmf` then the sltoht chain, then copying `updated_<tag>.har`/`sl4dump.har` to the fixtures dir. Guard it behind a new `--emit-bat` flag so the mac test run doesn't create a Windows script unexpectedly.

```python
ap.add_argument("--emit-bat", action="store_true",
                help="with --grid, also write run_study_windows.bat driving every config")
```

- [ ] **Step 2: Generate + eyeball the .bat on mac**

Run: `.venv/bin/python scripts/gtap/run_gempack_matrix.py --grid --no-solve --emit-bat --datasets gtap7_3x3 gtap7_15x10`
Expected: `runs/gempack_matrix/run_study_windows.bat` exists and contains one solve+convert+copy block per (dataset, tag); paths use the `%GTAPV7%`/`%SLTOHT%` env vars with sensible `C:\...` defaults.

- [ ] **Step 3: Document the round-trip**

Add "§10 — Linearization study (5 datasets × shock sweep × Gragg)" to `runs/gempack_updated_har_guide.md`: the three commands (`--grid --no-solve --emit-bat` on either OS to lay out the grid; `run_study_windows.bat` on Windows; `gen_linearization_study.py` on either OS to build the page), the expected ~50 runs, and that a small shock should push the match toward ~99% (the whole point). Cross-reference the spec + PR #40's SIMPLE 100% reference.

- [ ] **Step 4: Commit**

```bash
git add scripts/gtap/run_gempack_matrix.py runs/gempack_updated_har_guide.md
git commit -m "gtap(F5): Windows batch driver for the study grid + guide §10"
```

---

### Task 6: Mandatory parity gates + finding + PR

Per the repo rule, `scripts/gtap/*` + `tests/...` changes require a full parity-gate sweep + stamp before push. Then write the consolidated finding and open the PR.

**Files:**
- Create: `docs/findings/gempack_linearization_study_2026-07-24.md` (consolidated finding, supersedes the single-shock one's "how measured" section)
- Run: `scripts/gtap/run_parity_gates.py`

- [ ] **Step 1: Run the full parity-gate sweep + stamp**

Run: `.venv/bin/python scripts/gtap/run_parity_gates.py`
Expected: NLP-vs-NLP + MCP-vs-MCP sweeps pass, measured docs regenerated, input-tree stamp refreshed. This is MANDATORY and the push hook enforces it. If any gate regresses, STOP and investigate — do not stamp over a red gate.

- [ ] **Step 2: Write the consolidated finding**

`docs/findings/gempack_linearization_study_2026-07-24.md`: the design's 4 evidences, the ifSUB=condensation correction (with the Table D.1 citation), what mac gate #1 measured (ifSUB if1≡if0 %), the shock-sweep + Gragg tables (or "pending Windows" if not yet run), and the conclusion linking back to PR #40's SIMPLE 100%. Add a one-line pointer in the roadmap memory.

- [ ] **Step 3: Full test run**

Run: `.venv/bin/python -m pytest tests/templates/gtap/ -v -k "gempack or ifsub or linearization"`
Expected: all green or cleanly skipped (Windows-fixture-gated tests skip; mac gates pass).

- [ ] **Step 4: Push + open PR**

```bash
git push -u origin gtap/gempack-linearization-study
gh pr create --base main --title "gtap(F5): against-GEMPACK linearization study — tooling + ifSUB fidelity gate" --body "<summary of the 4 evidences, mac gate results, and the pending Windows grid>"
```

- [ ] **Step 5: Finish the branch**

REQUIRED SUB-SKILL: superpowers:finishing-a-development-branch.

---

## Self-Review

**Spec coverage:** All four spec evidences map to tasks — evidence 3 (ifSUB) → Task 1; evidences 1+2 (shock sweep + Gragg) → Task 2 (runner) + Task 4 (measurement/page); evidence 4 (welfare) → Task 3 (reader) + Task 4 (report). The two mac fidelity gates are Task 1 (ifSUB) and the local test steps throughout. The Windows phase is Task 5. Acceptance gates: Gate 3 is Task 1's floor (99.9%); Gates 1+2 are encoded in Task 4's page (match(0.1%)>match(10%), monotone in Steps) and asserted once real fixtures arrive; welfare-no-floor is Task 3/4.

**Placeholder scan:** The only deferred concretes are the real `decomp.har` header names (Task 3, Step 1 — discovered from a real file, with a defensive fallback + synthetic test) and the exact docs toctree file (Task 4, Step 6 — located via grep). Both are explicit discovery steps, not hand-waves. No "add error handling"/"write tests for the above" placeholders.

**Type consistency:** `config_tag(shock_pct, steps)` is defined in Task 2 and consumed by name in Tasks 4/5. `make_cmf` new kwargs (`steps`, `updated_name`) are used consistently. `gempack_welfare_ev` return shape `{region: {branch: float}}` (Task 3) matches its consumer in Task 4. `compare_levels` return dict keys (`frac_agree`, `worst`) match the test in Task 1. `_solve_shock` is a test-module function (not importable as a library) — the plan re-implements its sequence in each script rather than importing it, noted in Tasks 1/4.
