# probe.py Cached Parity Tool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/parity/probe.py`, a cached, flag-driven parity probe that skips the expensive solver re-run between hypotheses, reports GAMS-seed coverage, and does commit A/B attribution — without duplicating the adapter's build/solve logic.

**Architecture:** A thin CLI over `GTAPParityAdapter`. The model is always *built* from current source (cheap, ~3-5s); only the *solve point* is cached in `~/.cache/equilibria_probe/<key>.pkl`, keyed by a hash of the equation-source files + dataset + scenario + closure, so an equation edit invalidates it. Queries (`--show`, `--residuals`, `--seed-gams`, `--compare-ref`) run against the value-injected model. Reuses `_adapter_protocol`, `_triage_steps`, `_diff_core`.

**Tech Stack:** Python 3.12, Pyomo 6.9, pytest, `uv run` for execution, existing equilibria GTAP parity modules.

---

## File Structure

- Create: `scripts/parity/_probe_cache.py` — cache key + load/store of the solve point (pure, no Pyomo). One responsibility: cache I/O + invalidation.
- Create: `scripts/parity/_probe_queries.py` — the four query implementations operating on a built+valued Pyomo model. One responsibility: query logic.
- Create: `scripts/parity/probe.py` — CLI: arg parsing, orchestration (build → cache-or-solve → dispatch query). One responsibility: wiring.
- Create: `tests/parity/test_probe.py` — tests for cache, queries, coverage gating.

Rationale: cache I/O is pure and unit-testable without a solver; query logic needs a model but not the CLI; the CLI is thin glue. Splitting by responsibility keeps each file focused and testable in isolation.

---

## Conventions (read once)

- Run everything with `uv run` from the worktree root `/Users/marmol/.superset/worktrees/b14cb643-ee65-449d-b3f0-be8003b60783/alter-tax-gtap-7`.
- PATH must include gdxdump for `--seed-gams`: `export PATH="/Library/Frameworks/GAMS.framework/Versions/53/Resources:$PATH"`.
- Reference GDX for gtap7_3x3: `output/gtap7_3x3_altertax_neos_bundle/out_local.gdx`.
- Reused real signatures (verified):
  - `AdapterRegistry.get(template_name: str) -> type` (classmethod) in `scripts/parity/_adapter_protocol.py`; instantiate with `adapter_cls()`.
  - `GTAPParityAdapter.build_solved_model(dataset: str, scenario: str)` and `.build_warmstarted_model(dataset, scenario)` in `src/equilibria/templates/gtap/parity_adapter.py`.
  - `adapter.enumerate_combinations()` returns the valid `(dataset, scenario)` tuples.
  - `GTAPSolver.apply_solution_hint(hint) -> int` in `src/equilibria/templates/gtap/gtap_solver.py` (returns number of cells set).
  - `_gams_snapshot_from_altertax_gdx(gdx_path: Path, period: str)` in `parity_adapter.py` builds the hint snapshot.
  - `gams_levels(gdx_path: Path, symbol: str) -> dict` in `scripts/gtap/_diff_core.py`.

---

## Task 1: Cache key + store/load (`_probe_cache.py`)

**Files:**
- Create: `scripts/parity/_probe_cache.py`
- Test: `tests/parity/test_probe.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/parity/test_probe.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "parity"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from _probe_cache import compute_cache_key, store_solution, load_solution, KEY_FILES


def test_cache_key_changes_when_key_file_changes(tmp_path):
    f = tmp_path / "eq.py"
    f.write_text("x = 1")
    k1 = compute_cache_key("gtap7_3x3", "altertax_check", "altertax", [f])
    f.write_text("x = 2")
    k2 = compute_cache_key("gtap7_3x3", "altertax_check", "altertax", [f])
    assert k1 != k2
    assert len(k1) == 16


def test_cache_key_stable_for_same_inputs(tmp_path):
    f = tmp_path / "eq.py"
    f.write_text("x = 1")
    k1 = compute_cache_key("gtap7_3x3", "altertax_check", "altertax", [f])
    k2 = compute_cache_key("gtap7_3x3", "altertax_check", "altertax", [f])
    assert k1 == k2


def test_store_and_load_roundtrip(tmp_path):
    sol = {"pf": {("EU_28", "UnSkLab", "Mnfcs"): 1.45213}, "pi": {("ROW",): 1.0}}
    cache_dir = tmp_path / "cache"
    store_solution("abc123", sol, cache_dir=cache_dir)
    loaded = load_solution("abc123", cache_dir=cache_dir)
    assert loaded == sol


def test_load_missing_returns_none(tmp_path):
    assert load_solution("nope", cache_dir=tmp_path) is None


def test_key_files_are_real_paths():
    for p in KEY_FILES:
        assert p.exists(), f"KEY_FILES entry missing: {p}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/parity/test_probe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named '_probe_cache'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/parity/_probe_cache.py
"""Cache key + store/load for the probe's solve point. Pure (no Pyomo)."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]
_GTAP = _ROOT / "src" / "equilibria" / "templates" / "gtap"

# Equation-source files whose edits must invalidate the cached solve point.
KEY_FILES = [
    _GTAP / "gtap_model_equations.py",
    _GTAP / "altertax" / "parameter_overrides.py",
    _GTAP / "altertax" / "calibration_sequence.py",
    _GTAP / "altertax" / "postmodel.py",
    _GTAP / "gtap_parameters.py",
]

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "equilibria_probe"


def compute_cache_key(dataset: str, scenario: str, closure_name: str,
                      key_files: list[Path] | None = None) -> str:
    files = KEY_FILES if key_files is None else key_files
    h = hashlib.sha256()
    h.update(dataset.encode())
    h.update(b"\0")
    h.update(scenario.encode())
    h.update(b"\0")
    h.update(closure_name.encode())
    for p in files:
        h.update(b"\0")
        try:
            h.update(Path(p).read_bytes())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()[:16]


def store_solution(key: str, solution: dict, cache_dir: Path | None = None) -> Path:
    d = cache_dir or DEFAULT_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{key}.pkl"
    with path.open("wb") as fh:
        pickle.dump(solution, fh)
    return path


def load_solution(key: str, cache_dir: Path | None = None) -> Optional[dict]:
    d = cache_dir or DEFAULT_CACHE_DIR
    path = d / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None  # corrupt cache → treat as miss


def clear_cache(cache_dir: Path | None = None) -> int:
    d = cache_dir or DEFAULT_CACHE_DIR
    if not d.exists():
        return 0
    n = 0
    for p in d.glob("*.pkl"):
        p.unlink()
        n += 1
    return n
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/parity/test_probe.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/parity/_probe_cache.py tests/parity/test_probe.py
git commit -m "feat(probe): cache key + store/load with equation-source invalidation"
```

---

## Task 2: Extract & inject solution values (`_probe_queries.py` part 1)

**Files:**
- Create: `scripts/parity/_probe_queries.py`
- Test: `tests/parity/test_probe.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/parity/test_probe.py
from _probe_queries import extract_solution, inject_solution


class _FakeVar:
    def __init__(self, value):
        self._value = value
        self.fixed = False
    def __call__(self):  # pyomo value() compat not needed; we read .value
        return self._value


def test_extract_inject_roundtrip_with_real_model():
    import pytest
    pytest.importorskip("pyomo")
    from _adapter_protocol import AdapterRegistry
    adapter = AdapterRegistry.get("gtap")()
    if ("gtap7_3x3", "altertax_check") not in adapter.enumerate_combinations():
        pytest.skip("gtap7_3x3 not available")
    m = adapter.build_warmstarted_model("gtap7_3x3", "altertax_check")
    sol = extract_solution(m)
    assert "pf" in sol and len(sol["pf"]) > 0
    # zero out one var, then re-inject, confirm restored
    from pyomo.environ import value
    key = next(iter(sol["pf"]))
    original = sol["pf"][key]
    m.pf[key].set_value(0.0)
    inject_solution(m, sol)
    assert abs(value(m.pf[key]) - original) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/parity/test_probe.py::test_extract_inject_roundtrip_with_real_model -v`
Expected: FAIL with `ModuleNotFoundError: No module named '_probe_queries'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/parity/_probe_queries.py
"""Query implementations operating on a built+valued Pyomo GTAP model."""
from __future__ import annotations

from typing import Any

from pyomo.environ import Var, Constraint, value


def _idx_key(idx):
    if idx is None:
        return (None,)
    if isinstance(idx, tuple):
        return idx
    return (idx,)


def extract_solution(model: Any) -> dict:
    """Return {var_name: {idx: value}} for all active Var components."""
    sol: dict[str, dict] = {}
    for v in model.component_objects(Var, active=True):
        cells = {}
        for idx in v:
            try:
                val = v[idx].value
                if val is not None:
                    cells[_idx_key(idx)] = float(val)
            except Exception:
                pass
        if cells:
            sol[v.local_name] = cells
    return sol


def inject_solution(model: Any, solution: dict) -> int:
    """Set Var values from a solution dict. Returns cells set."""
    n = 0
    for name, cells in solution.items():
        comp = getattr(model, name, None)
        if comp is None:
            continue
        for idx, val in cells.items():
            pyidx = idx[0] if (isinstance(idx, tuple) and len(idx) == 1 and idx[0] is None) else idx
            try:
                item = comp[pyidx]
                if hasattr(item, "fixed") and item.fixed:
                    continue
                item.set_value(float(val))
                n += 1
            except Exception:
                pass
    return n
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/parity/test_probe.py::test_extract_inject_roundtrip_with_real_model -v`
Expected: PASS (or SKIP if fixture absent — acceptable)

- [ ] **Step 5: Commit**

```bash
git add scripts/parity/_probe_queries.py tests/parity/test_probe.py
git commit -m "feat(probe): extract/inject solution values to/from Pyomo model"
```

---

## Task 3: `--show` and `--residuals` queries (`_probe_queries.py` part 2)

**Files:**
- Modify: `scripts/parity/_probe_queries.py`
- Test: `tests/parity/test_probe.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/parity/test_probe.py
from _probe_queries import query_show, query_residuals


def test_query_show_filters_by_region():
    import pytest
    pytest.importorskip("pyomo")
    from _adapter_protocol import AdapterRegistry
    adapter = AdapterRegistry.get("gtap")()
    if ("gtap7_3x3", "altertax_check") not in adapter.enumerate_combinations():
        pytest.skip("gtap7_3x3 not available")
    m = adapter.build_warmstarted_model("gtap7_3x3", "altertax_check")
    rows = query_show(m, ["pi"], region="ROW")
    assert all(r["var"] == "pi" for r in rows)
    assert all("ROW" in str(r["idx"]) for r in rows)
    assert len(rows) >= 1


def test_query_residuals_sorted_desc():
    import pytest
    pytest.importorskip("pyomo")
    from _adapter_protocol import AdapterRegistry
    adapter = AdapterRegistry.get("gtap")()
    if ("gtap7_3x3", "altertax_check") not in adapter.enumerate_combinations():
        pytest.skip("gtap7_3x3 not available")
    m = adapter.build_warmstarted_model("gtap7_3x3", "altertax_check")
    rows = query_residuals(m, top_n=5)
    assert len(rows) <= 5
    vals = [r["resid"] for r in rows]
    assert vals == sorted(vals, reverse=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/parity/test_probe.py::test_query_show_filters_by_region tests/parity/test_probe.py::test_query_residuals_sorted_desc -v`
Expected: FAIL with `ImportError: cannot import name 'query_show'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/parity/_probe_queries.py

def query_show(model, var_names, region=None, index_filter=None):
    """Return [{var, idx, value}] for the named vars, optionally filtered."""
    rows = []
    for name in var_names:
        comp = getattr(model, name, None)
        if comp is None:
            continue
        for idx in comp:
            key = _idx_key(idx)
            if region is not None and region not in (str(k) for k in key):
                continue
            if index_filter is not None and index_filter not in (str(k) for k in key):
                continue
            try:
                val = comp[idx].value
            except Exception:
                val = None
            rows.append({"var": name, "idx": key, "value": val})
    return rows


def _family(name: str) -> str:
    for sep in ("[", "("):
        i = name.find(sep)
        if i != -1:
            return name[:i]
    return name


def query_residuals(model, top_n=15, family=None):
    """Return [{eq, idx, resid}] = |body - target|, sorted desc."""
    rows = []
    for c in model.component_objects(Constraint, active=True):
        if family is not None and c.local_name != family:
            continue
        for idx in c:
            con = c[idx]
            try:
                body = value(con.body)
                lo = con.lower
                up = con.upper
                tgt = value(lo) if lo is not None else (value(up) if up is not None else 0.0)
                rows.append({"eq": c.local_name, "idx": idx, "resid": abs(body - tgt)})
            except Exception:
                pass
    rows.sort(key=lambda r: r["resid"], reverse=True)
    return rows[:top_n]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/parity/test_probe.py -k "query_show or query_residuals" -v`
Expected: PASS (or SKIP if fixture absent)

- [ ] **Step 5: Commit**

```bash
git add scripts/parity/_probe_queries.py tests/parity/test_probe.py
git commit -m "feat(probe): --show and --residuals query functions"
```

---

## Task 4: `--seed-gams` with coverage gate (`_probe_queries.py` part 3)

**Files:**
- Modify: `scripts/parity/_probe_queries.py`
- Test: `tests/parity/test_probe.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/parity/test_probe.py
from _probe_queries import seed_gams_point


def test_seed_gams_reports_coverage():
    import pytest
    pytest.importorskip("pyomo")
    from _adapter_protocol import AdapterRegistry
    ref = ROOT / "output" / "gtap7_3x3_altertax_neos_bundle" / "out_local.gdx"
    if not ref.exists():
        pytest.skip("reference GDX absent")
    adapter = AdapterRegistry.get("gtap")()
    if ("gtap7_3x3", "altertax_check") not in adapter.enumerate_combinations():
        pytest.skip("gtap7_3x3 not available")
    m = adapter.build_warmstarted_model("gtap7_3x3", "altertax_check")
    result = seed_gams_point(m, ref, "base")
    assert result["cells_set"] > 100
    assert 0.0 <= result["coverage"] <= 1.0
    assert isinstance(result["below_threshold"], bool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/parity/test_probe.py::test_seed_gams_reports_coverage -v`
Expected: FAIL with `ImportError: cannot import name 'seed_gams_point'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/parity/_probe_queries.py
from pathlib import Path


def seed_gams_point(model, gdx_path: Path, period: str, threshold: float = 0.95):
    """Seed model with the GAMS point for `period` via apply_solution_hint.

    Returns {cells_set, total_cells, coverage, below_threshold}. Coverage is
    cells_set / total free Var cells in the model.
    """
    from equilibria.templates.gtap.parity_adapter import (
        _gams_snapshot_from_altertax_gdx,
    )
    from equilibria.templates.gtap.gtap_solver import GTAPSolver

    snap = _gams_snapshot_from_altertax_gdx(Path(gdx_path), period)
    helper = GTAPSolver(model, solver_name="path")
    cells_set = helper.apply_solution_hint(snap)

    total = 0
    for v in model.component_objects(Var, active=True):
        for idx in v:
            try:
                if not v[idx].fixed:
                    total += 1
            except Exception:
                total += 1
    coverage = (cells_set / total) if total else 0.0
    return {
        "cells_set": cells_set,
        "total_cells": total,
        "coverage": coverage,
        "below_threshold": coverage < threshold,
    }
```

Note: if `GTAPSolver(model, solver_name="path")` raises because it needs more
constructor args, inspect the real signature with
`uv run python -c "import inspect, equilibria.templates.gtap.gtap_solver as s; print(inspect.signature(s.GTAPSolver.__init__))"`
and pass the minimal required args (the adapter calls it as
`GTAPSolver(m_chk, closure=alt_closure, solver_name="path", params=p_alt)`; if a
bare construction fails, thread `closure` and `params` through `seed_gams_point`
as parameters from the caller in Task 6).

- [ ] **Step 4: Run test to verify it passes**

Run: `export PATH="/Library/Frameworks/GAMS.framework/Versions/53/Resources:$PATH" && uv run pytest tests/parity/test_probe.py::test_seed_gams_reports_coverage -v`
Expected: PASS (or SKIP if GDX absent)

- [ ] **Step 5: Commit**

```bash
git add scripts/parity/_probe_queries.py tests/parity/test_probe.py
git commit -m "feat(probe): --seed-gams with coverage gate"
```

---

## Task 5: CLI orchestration (`probe.py`) — build, cache-or-solve, dispatch

**Files:**
- Create: `scripts/parity/probe.py`
- Test: `tests/parity/test_probe.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/parity/test_probe.py
import subprocess


def test_cli_show_runs_and_caches(tmp_path):
    import pytest
    ref = ROOT / "output" / "gtap7_3x3_altertax_neos_bundle" / "out_local.gdx"
    # not strictly needed for --show, but the build path is the gate
    env_cache = tmp_path / "cache"
    cmd = [
        "uv", "run", "python", "scripts/parity/probe.py",
        "--template", "gtap", "--dataset", "gtap7_3x3",
        "--scenario", "altertax_check", "--show", "pi", "--region", "ROW",
        "--cache-dir", str(env_cache),
    ]
    r1 = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    if "not available" in (r1.stdout + r1.stderr) or "Skip" in r1.stdout:
        pytest.skip("gtap7_3x3 not available")
    assert r1.returncode == 0, r1.stderr
    assert "pi" in r1.stdout
    # second run should be a cache hit (fast); assert marker present
    r2 = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r2.returncode == 0, r2.stderr
    assert "cache hit" in r2.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/parity/test_probe.py::test_cli_show_runs_and_caches -v`
Expected: FAIL (probe.py does not exist → returncode != 0)

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/parity/probe.py
#!/usr/bin/env python
"""Cached, flag-driven parity probe. See docs/superpowers/specs/2026-06-12-probe-cached-parity-tool-design.md."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "parity"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from _adapter_protocol import AdapterRegistry  # noqa: E402
from _probe_cache import (  # noqa: E402
    compute_cache_key, load_solution, store_solution, clear_cache,
)
from _probe_queries import (  # noqa: E402
    extract_solution, inject_solution, query_show, query_residuals,
    seed_gams_point,
)

# Closure name per scenario (matches GTAPParityAdapter altertax closures).
_CLOSURE_NAME = {
    "altertax_check": "altertax",
    "altertax_shock": "altertax",
    "baseline": "base",
    "shock_tm10": "base",
}


def _build_and_value(adapter, dataset, scenario, cache_dir, no_cache):
    """Build the model (always) and supply solved values from cache or solve."""
    closure = _CLOSURE_NAME.get(scenario, scenario)
    key = compute_cache_key(dataset, scenario, closure)
    # build_solved_model both builds AND solves; build_warmstarted_model builds
    # without the PATH solve. We want: build now, then either inject cache or solve.
    cached = None if no_cache else load_solution(key, cache_dir=cache_dir)
    if cached is not None:
        model = adapter.build_warmstarted_model(dataset, scenario)
        n = inject_solution(model, cached)
        print(f"[cache hit] key={key}  injected {n} cells")
        return model, key
    t0 = time.time()
    model = adapter.build_solved_model(dataset, scenario)
    sol = extract_solution(model)
    store_solution(key, sol, cache_dir=cache_dir)
    print(f"[solved + cached {time.time()-t0:.0f}s] key={key}")
    return model, key


def _print_show(rows):
    for r in rows:
        v = r["value"]
        vs = f"{v:.6f}" if isinstance(v, float) else str(v)
        print(f"  {r['var']:<10s} {str(r['idx']):<32s} {vs}")


def _print_residuals(rows):
    for r in rows:
        print(f"  {r['eq']:<28s} idx={str(r['idx']):<28s} resid={r['resid']:.4e}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Cached parity probe")
    ap.add_argument("--template", default="gtap")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--show", help="comma-separated var names")
    ap.add_argument("--region")
    ap.add_argument("--index")
    ap.add_argument("--residuals", action="store_true")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--family")
    ap.add_argument("--seed-gams", dest="seed_gams", help="period: base|check|shock")
    ap.add_argument("--seed-threshold", type=float, default=0.95)
    ap.add_argument("--gdx-ref")
    ap.add_argument("--compare-ref", dest="compare_ref")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--clear-cache", action="store_true")
    ap.add_argument("--cache-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.clear_cache:
        n = clear_cache(cache_dir=args.cache_dir)
        print(f"cleared {n} cache files")
        return 0

    try:
        adapter = AdapterRegistry.get(args.template)()
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if (args.dataset, args.scenario) not in adapter.enumerate_combinations():
        print(f"error: ({args.dataset!r}, {args.scenario!r}) not available; "
              f"have {adapter.enumerate_combinations()}", file=sys.stderr)
        return 2

    model, key = _build_and_value(adapter, args.dataset, args.scenario,
                                  args.cache_dir, args.no_cache)

    if args.seed_gams:
        if not args.gdx_ref:
            print("error: --seed-gams requires --gdx-ref", file=sys.stderr)
            return 2
        res = seed_gams_point(model, Path(args.gdx_ref), args.seed_gams,
                              threshold=args.seed_threshold)
        print(f"[seed-gams {args.seed_gams}] set {res['cells_set']}/"
              f"{res['total_cells']} cells ({res['coverage']:.0%})")
        if res["below_threshold"]:
            print(f"  WARNING: coverage {res['coverage']:.0%} < "
                  f"{args.seed_threshold:.0%} — results may be unreliable")

    if args.show:
        rows = query_show(model, args.show.split(","), region=args.region,
                          index_filter=args.index)
        print(f"=== show {args.show} ===")
        _print_show(rows)
    if args.residuals:
        rows = query_residuals(model, top_n=args.top, family=args.family)
        print(f"=== residuals (top {args.top}) ===")
        _print_residuals(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/parity/test_probe.py::test_cli_show_runs_and_caches -v`
Expected: PASS (or SKIP if fixture absent)

- [ ] **Step 5: Commit**

```bash
git add scripts/parity/probe.py tests/parity/test_probe.py
git commit -m "feat(probe): CLI orchestration — build, cache-or-solve, --show/--residuals/--seed-gams"
```

---

## Task 6: `--compare-ref` commit A/B via temp worktree

**Files:**
- Modify: `scripts/parity/probe.py`
- Test: `tests/parity/test_probe.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/parity/test_probe.py
def test_compare_ref_runs_against_head_itself(tmp_path):
    import pytest
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    head = r.stdout.strip()
    cmd = [
        "uv", "run", "python", "scripts/parity/probe.py",
        "--template", "gtap", "--dataset", "gtap7_3x3",
        "--scenario", "altertax_check", "--show", "pi", "--region", "ROW",
        "--compare-ref", head, "--cache-dir", str(tmp_path / "c"),
    ]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    if "not available" in (out.stdout + out.stderr):
        pytest.skip("gtap7_3x3 not available")
    assert out.returncode == 0, out.stderr
    # comparing HEAD to HEAD → delta column all ~0
    assert "HEAD" in out.stdout and "Δ" in out.stdout
    # no leftover temp worktrees
    wl = subprocess.run(["git", "worktree", "list"], cwd=ROOT,
                        capture_output=True, text=True)
    assert "probe_compare_" not in wl.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/parity/test_probe.py::test_compare_ref_runs_against_head_itself -v`
Expected: FAIL (no `--compare-ref` handling → no "Δ" in output / nonzero exit)

- [ ] **Step 3: Write minimal implementation**

Add a `_run_compare_ref` function and wire it into `main()` before the normal
query dispatch. Insert the function above `main()`:

```python
# add to scripts/parity/probe.py (above main)
import json
import subprocess
import tempfile


def _query_to_dict(model, args):
    """Run the requested read-only query and return {label: {key: value}}."""
    out = {}
    if args.show:
        for r in query_show(model, args.show.split(","), region=args.region,
                            index_filter=args.index):
            out[f"{r['var']}{r['idx']}"] = r["value"]
    if args.residuals:
        for r in query_residuals(model, top_n=args.top, family=args.family):
            out[f"{r['eq']}{r['idx']}"] = r["resid"]
    return out


def _run_compare_ref(adapter, args) -> int:
    """A/B the query: HEAD vs args.compare_ref, side-by-side."""
    model, _ = _build_and_value(adapter, args.dataset, args.scenario,
                                args.cache_dir, args.no_cache)
    head_vals = _query_to_dict(model, args)

    tmp = Path(tempfile.mkdtemp(prefix="probe_compare_"))
    wt = tmp / "wt"
    try:
        subprocess.run(["git", "worktree", "add", "--detach", str(wt),
                        args.compare_ref], cwd=ROOT, check=True,
                       capture_output=True, text=True)
        # run probe in the ref worktree as JSON-emitting subprocess
        sub_cmd = [
            "uv", "run", "python", "scripts/parity/probe.py",
            "--template", args.template, "--dataset", args.dataset,
            "--scenario", args.scenario, "--emit-json",
            "--cache-dir", str(tmp / "refcache"),
        ]
        if args.show:
            sub_cmd += ["--show", args.show]
            if args.region:
                sub_cmd += ["--region", args.region]
        if args.residuals:
            sub_cmd += ["--residuals", "--top", str(args.top)]
        sub = subprocess.run(sub_cmd, cwd=wt, capture_output=True, text=True,
                             timeout=600)
        ref_vals = {}
        for line in sub.stdout.splitlines():
            if line.startswith("__JSON__"):
                ref_vals = json.loads(line[len("__JSON__"):])
                break
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)],
                       cwd=ROOT, capture_output=True, text=True)

    keys = sorted(set(head_vals) | set(ref_vals))
    print(f"{'key':<40s} {'HEAD':>14s} {args.compare_ref[:10]:>14s} {'Δ':>14s}")
    for k in keys:
        h = head_vals.get(k)
        rv = ref_vals.get(k)
        delta = (h - rv) if (isinstance(h, (int, float)) and isinstance(rv, (int, float))) else None
        hs = f"{h:.6f}" if isinstance(h, float) else str(h)
        rs = f"{rv:.6f}" if isinstance(rv, float) else str(rv)
        ds = f"{delta:.6f}" if isinstance(delta, float) else "—"
        print(f"  {k:<38s} {hs:>14s} {rs:>14s} {ds:>14s}")
    return 0
```

Then add `--emit-json` to the parser and, in `main()`, handle both new paths.
Insert near the other `ap.add_argument` calls:

```python
    ap.add_argument("--emit-json", action="store_true",
                    help="emit query result as __JSON__-prefixed line (for --compare-ref subprocess)")
```

And in `main()`, immediately after the combos check and BEFORE `_build_and_value`,
add the compare-ref branch:

```python
    if args.compare_ref:
        return _run_compare_ref(adapter, args)
```

And when `--emit-json` is set, after building/valuing the model, emit JSON instead
of tables. Replace the tail of `main()` (the `if args.show` / `if args.residuals`
block) with:

```python
    if args.emit_json:
        print("__JSON__" + json.dumps(_query_to_dict(model, args)))
        return 0

    if args.show:
        rows = query_show(model, args.show.split(","), region=args.region,
                          index_filter=args.index)
        print(f"=== show {args.show} ===")
        _print_show(rows)
    if args.residuals:
        rows = query_residuals(model, top_n=args.top, family=args.family)
        print(f"=== residuals (top {args.top}) ===")
        _print_residuals(rows)
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/parity/test_probe.py::test_compare_ref_runs_against_head_itself -v`
Expected: PASS (or SKIP if fixture absent)

- [ ] **Step 5: Commit**

```bash
git add scripts/parity/probe.py tests/parity/test_probe.py
git commit -m "feat(probe): --compare-ref commit A/B via temp worktree"
```

---

## Task 7: Cache invalidation integration test + docs pointer

**Files:**
- Modify: `tests/parity/test_probe.py`
- Modify: `CLAUDE.md` (add probe.py to the debug tools table)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/parity/test_probe.py
from _probe_cache import compute_cache_key as _cck


def test_cache_key_invalidates_on_real_key_file_touch(tmp_path):
    # Simulate an edit by hashing a copy with different content.
    f = tmp_path / "gtap_model_equations.py"
    f.write_text("# v1")
    k1 = _cck("gtap7_3x3", "altertax_check", "altertax", [f])
    f.write_text("# v2  (an equation changed)")
    k2 = _cck("gtap7_3x3", "altertax_check", "altertax", [f])
    assert k1 != k2, "editing a key equation file must invalidate the cache"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/parity/test_probe.py::test_cache_key_invalidates_on_real_key_file_touch -v`
Expected: PASS immediately (logic already exists from Task 1) — if it FAILS, the
Task 1 key function is wrong; fix `compute_cache_key` to include file contents.

- [ ] **Step 3: Add docs pointer**

Add a row to the debug-tools cascade table in `CLAUDE.md` (the section
"Herramientas de debug parity (cascade de 4)"). Insert after the table:

```markdown

**Iteración rápida:** `scripts/parity/probe.py` — probe cacheado. Construye el
modelo siempre (refleja el código actual) y cachea sólo el punto resuelto
(invalidado por hash de las ecuaciones). Flags: `--show <vars> [--region R]`,
`--residuals [--top N] [--family F]`, `--seed-gams <period> --gdx-ref <gdx>`
(con gate de cobertura), `--compare-ref <commit>` (A/B automático). No reemplaza
la cascada de 4 — acelera la iteración de hipótesis y la atribución.
```

- [ ] **Step 4: Run the full probe test module**

Run: `export PATH="/Library/Frameworks/GAMS.framework/Versions/53/Resources:$PATH" && uv run pytest tests/parity/test_probe.py -v`
Expected: all PASS or SKIP (skips only where fixture GDX absent)

- [ ] **Step 5: Commit**

```bash
git add tests/parity/test_probe.py CLAUDE.md
git commit -m "test(probe): cache invalidation test + CLAUDE.md tool pointer"
```

---

## Task 8: Regression gate + end-to-end smoke

**Files:** none new

- [ ] **Step 1: Run the parity regression gate (must stay green)**

Run: `uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py -q`
Expected: `5 passed`

- [ ] **Step 2: End-to-end smoke — reproduce a session query**

Run:
```bash
export PATH="/Library/Frameworks/GAMS.framework/Versions/53/Resources:$PATH"
uv run python scripts/parity/probe.py --dataset gtap7_3x3 --scenario altertax_check \
  --show pf,pfa,pi --region ROW
uv run python scripts/parity/probe.py --dataset gtap7_3x3 --scenario altertax_check \
  --seed-gams base --gdx-ref output/gtap7_3x3_altertax_neos_bundle/out_local.gdx \
  --residuals --top 10
```
Expected: first run `[solved + cached Ns]`, second `--seed-gams` prints coverage
≥ ~80% and top residuals headed by `eq_regy` / `eq_x[*,Svces,Svces]`.

- [ ] **Step 3: Verify cache hit speed**

Run (immediately after the above):
```bash
time uv run python scripts/parity/probe.py --dataset gtap7_3x3 \
  --scenario altertax_check --show pi --region ROW
```
Expected: prints `[cache hit]`, wall-time dominated by build (~5s), not solve.

- [ ] **Step 4: Commit (if any doc/notes changed)**

```bash
git commit --allow-empty -m "chore(probe): end-to-end smoke verified (gtap7_3x3 altertax)"
```
