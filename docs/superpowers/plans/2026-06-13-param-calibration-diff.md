# probe.py --params (Param/calibration diff) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 5th parity-debug tool, `probe.py --params`, that compares all Pyomo `Param`s of the built model against the GAMS GDX (reporting diverge/ok/no-match), plus `--params-compare-builds` to auto-detect Params that change with/without `t0_snapshot`.

**Architecture:** Pure comparison logic in a new `scripts/parity/_probe_params.py`; `probe.py` gains `--params` / `--params-compare-builds` flags that build the model (via the adapter for `--params`; directly via `GTAPModelEquations` for compare-builds) and dispatch to the new module. Reuses `_diff_core.gams_levels` for GAMS reads and the snapshot's index normalization.

**Tech Stack:** Python 3.12, Pyomo 6.9, pytest, `uv run`, existing equilibria GTAP parity modules.

---

## File Structure

- Create: `scripts/parity/_probe_params.py` — pure: extract Params from a model, resolve GAMS symbols (with alias map), diff cell-by-cell into 3 groups, and diff two models' Params. One responsibility: Param comparison.
- Modify: `scripts/parity/probe.py` — add `--params` / `--params-compare-builds` flags + dispatch + printing.
- Modify: `tests/parity/test_probe.py` — tests for the new query.
- Modify: `CLAUDE.md` — register the 5th tool in the debug-tools table.

---

## Conventions (read once)

- Run with `uv run` from worktree root `/Users/marmol/.superset/worktrees/b14cb643-ee65-449d-b3f0-be8003b60783/alter-tax-gtap-7`.
- `--params` needs gdxdump on PATH: `export PATH="/Library/Frameworks/GAMS.framework/Versions/53/Resources:$PATH"`.
- Reference GDX: `output/gtap7_3x3_altertax_neos_bundle/out_local.gdx`.
- Verified real interfaces:
  - `gams_levels(gdx_path: Path, symbol: str) -> dict` in `scripts/gtap/_diff_core.py` — keys are tuples ending in the period (e.g. `('USA','c_Mnfcs','USA','base')`), values are floats.
  - `adapter.build_warmstarted_model(dataset, scenario)` builds the model (Params set at build; no solve needed).
  - `_idx_key(idx)` in `scripts/parity/_probe_queries.py` normalizes a Pyomo index to a tuple — reuse it.
  - Params iterate via `model.component_objects(Param, active=True)`; each `p` is indexed, `p[idx].value` (or `pyomo value(p[idx])`) gives the constant.
  - Build-with/without-t0 pattern: `_build_altertax_check_model` in `parity_adapter.py:263` builds `m_b = GTAPModelEquations(...).build_model()` then `m_chk = GTAPModelEquations(..., t0_snapshot=m_b).build_model()`.

---

## Task 1: Extract Params + GAMS symbol resolution (`_probe_params.py`)

**Files:**
- Create: `scripts/parity/_probe_params.py`
- Test: `tests/parity/test_probe.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/parity/test_probe.py
from _probe_params import extract_params, ALIAS_MAP, resolve_gams_symbol


def test_extract_params_with_real_model():
    import pytest
    pytest.importorskip("pyomo")
    from _adapter_protocol import AdapterRegistry
    adapter = AdapterRegistry.get("gtap")()
    if ("gtap7_3x3", "altertax_check") not in adapter.enumerate_combinations():
        pytest.skip("gtap7_3x3 not available")
    m = adapter.build_warmstarted_model("gtap7_3x3", "altertax_check")
    params = extract_params(m)
    # pf0 is a build-time Param (Fisher anchor); kappaf is a tax Param
    assert "pf0" in params and len(params["pf0"]) > 0
    assert all(isinstance(v, float) for v in params["pf0"].values())


def test_alias_map_and_resolve():
    # pf0 maps to GAMS pf at the base period
    assert ALIAS_MAP["pf0"] == "pf@base"
    name, period = resolve_gams_symbol("pf0")
    assert name == "pf" and period == "base"
    # un-aliased param: same name, period from caller
    name, period = resolve_gams_symbol("kappaf")
    assert name == "kappaf" and period is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/parity/test_probe.py::test_alias_map_and_resolve -v`
Expected: FAIL with `ModuleNotFoundError: No module named '_probe_params'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/parity/_probe_params.py
"""Param/calibration diff: compare a built model's Pyomo Params vs the GAMS GDX.

Covers the cascade blind spot — build-time calibrated constants (pf0, base_rgdpmp,
p_gf, betap) that the .nl / residual / warm-start / closure tools take as literals.
"""
from __future__ import annotations

from typing import Any, Optional

from pyomo.environ import Param, value

from _probe_queries import _idx_key  # reuse index normalization

# Python Param name -> GAMS symbol, optionally "@<period>" for a different period.
# pf0/xf0 are the base-period pf/xf; base_rgdpmp is rgdpmp@base; betap is betaP.
ALIAS_MAP = {
    "pf0": "pf@base",
    "xf0": "xf@base",
    "base_rgdpmp": "rgdpmp@base",
    "base_pabs": "pabs@base",
    "betap": "betaP",
    "betag": "betaG",
    "betas": "betaS",
    "kappaf_activity": "kappaf",
}


def resolve_gams_symbol(param_name: str) -> tuple[str, Optional[str]]:
    """Return (gams_symbol, period_override_or_None) for a Python Param name."""
    alias = ALIAS_MAP.get(param_name, param_name)
    if "@" in alias:
        sym, period = alias.split("@", 1)
        return sym, period
    return alias, None


def extract_params(model: Any) -> dict:
    """Return {param_name: {idx_tuple: float}} for all active Params."""
    out: dict[str, dict] = {}
    for p in model.component_objects(Param, active=True):
        cells = {}
        for idx in p:
            try:
                v = value(p[idx])
                if v is not None:
                    cells[_idx_key(idx)] = float(v)
            except Exception:
                pass
        if cells:
            out[p.local_name] = cells
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `export PATH="/Library/Frameworks/GAMS.framework/Versions/53/Resources:$PATH" && uv run pytest tests/parity/test_probe.py::test_alias_map_and_resolve tests/parity/test_probe.py::test_extract_params_with_real_model -v`
Expected: PASS (or SKIP if fixture absent)

- [ ] **Step 5: Commit**

```bash
git add scripts/parity/_probe_params.py tests/parity/test_probe.py
git commit -m "feat(probe): extract Params + GAMS symbol resolution for --params"
```

---

## Task 2: Diff Params vs GAMS GDX into 3 groups (`_probe_params.py`)

**Files:**
- Modify: `scripts/parity/_probe_params.py`
- Test: `tests/parity/test_probe.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/parity/test_probe.py
from _probe_params import diff_params_vs_gams


def test_diff_params_vs_gams_three_groups():
    import pytest
    pytest.importorskip("pyomo")
    from pathlib import Path as _P
    from _adapter_protocol import AdapterRegistry
    ref = ROOT / "output" / "gtap7_3x3_altertax_neos_bundle" / "out_local.gdx"
    if not ref.exists():
        pytest.skip("reference GDX absent")
    adapter = AdapterRegistry.get("gtap")()
    if ("gtap7_3x3", "altertax_check") not in adapter.enumerate_combinations():
        pytest.skip("gtap7_3x3 not available")
    m = adapter.build_warmstarted_model("gtap7_3x3", "altertax_check")
    result = diff_params_vs_gams(m, ref, "check", tol_rel=1e-3)
    assert "diverge" in result and "ok" in result and "no_match" in result
    # ok + diverge are verifiable vs GAMS; no_match are not
    assert isinstance(result["diverge"], list)
    assert isinstance(result["ok"], list)
    assert isinstance(result["no_match"], list)
    # kappaf was verified == GAMS this session → should be ok, not diverge
    assert "kappaf" not in {r["param"] for r in result["diverge"]}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/parity/test_probe.py::test_diff_params_vs_gams_three_groups -v`
Expected: FAIL with `ImportError: cannot import name 'diff_params_vs_gams'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/parity/_probe_params.py
import re
from pathlib import Path

_PREFIXES = ("a_", "c_", "f_", "r_")


def _strip(s) -> str:
    s = str(s)
    for p in _PREFIXES:
        if s.startswith(p):
            return s[len(p):]
    return s


def _gams_period_slice(gdx_path: Path, symbol: str, period: str) -> dict:
    """Read a GAMS symbol, keep entries for `period`, strip prefixes + period dim."""
    from _diff_core import gams_levels  # type: ignore[import-not-found]
    raw = gams_levels(Path(gdx_path), symbol)
    out = {}
    for k, v in raw.items():
        if not isinstance(k, tuple) or k[-1] != period:
            continue
        nk = tuple(_strip(x) for x in k[:-1])
        out[nk] = float(v)
    return out


def diff_params_vs_gams(model, gdx_path, period: str, tol_rel: float = 1e-3) -> dict:
    """Compare all model Params vs the GAMS GDX for `period`.

    Returns {'diverge': [...], 'ok': [...], 'no_match': [...]}. Each diverge entry:
    {param, gams_symbol, cells, match, diverge, max_rel, worst}.
    """
    params = extract_params(model)
    diverge, ok, no_match = [], [], []
    for name, cells in params.items():
        gsym, period_override = resolve_gams_symbol(name)
        gperiod = period_override or period
        gams = _gams_period_slice(Path(gdx_path), gsym, gperiod)
        if not gams:
            no_match.append({"param": name, "gams_symbol": gsym, "cells": len(cells)})
            continue
        n_match = n_div = 0
        max_rel = 0.0
        worst = None
        for idx, pv in cells.items():
            key = idx[0] if (isinstance(idx, tuple) and len(idx) == 1) else idx
            gv = gams.get(key)
            if gv is None:
                continue
            rel = abs(pv - gv) / max(abs(gv), 1e-12)
            if rel > tol_rel:
                n_div += 1
                if rel > max_rel:
                    max_rel = rel
                    worst = (key, pv, gv)
            else:
                n_match += 1
        rec = {"param": name, "gams_symbol": gsym, "cells": n_match + n_div,
               "match": n_match, "diverge": n_div, "max_rel": max_rel, "worst": worst}
        (diverge if n_div > 0 else ok).append(rec)
    diverge.sort(key=lambda r: r["max_rel"], reverse=True)
    return {"diverge": diverge, "ok": ok, "no_match": no_match}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `export PATH="/Library/Frameworks/GAMS.framework/Versions/53/Resources:$PATH" && uv run pytest tests/parity/test_probe.py::test_diff_params_vs_gams_three_groups -v`
Expected: PASS (or SKIP if GDX absent)

- [ ] **Step 5: Commit**

```bash
git add scripts/parity/_probe_params.py tests/parity/test_probe.py
git commit -m "feat(probe): diff_params_vs_gams (diverge/ok/no-match groups)"
```

---

## Task 3: compare-builds — Params that change with/without t0_snapshot

**Files:**
- Modify: `scripts/parity/_probe_params.py`
- Test: `tests/parity/test_probe.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/parity/test_probe.py
from _probe_params import diff_param_builds


def test_diff_param_builds_detects_t0_dependence():
    import pytest
    pytest.importorskip("pyomo")
    from _adapter_protocol import AdapterRegistry
    if ("gtap7_3x3", "altertax_check") not in AdapterRegistry.get("gtap")().enumerate_combinations():
        pytest.skip("gtap7_3x3 not available")
    changed = diff_param_builds("gtap7_3x3", tol_rel=1e-3)
    names = {r["param"] for r in changed}
    # the Fisher anchors / shares depend on t0_snapshot — the shock-bug signature
    assert "pf0" in names or "base_rgdpmp" in names, f"expected build-dependent params, got {names}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/parity/test_probe.py::test_diff_param_builds_detects_t0_dependence -v`
Expected: FAIL with `ImportError: cannot import name 'diff_param_builds'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/parity/_probe_params.py

def _build_gtap7_har_models(dataset: str):
    """Build the altertax model twice: WITHOUT and WITH t0_snapshot. Returns (m_no_t0, m_t0)."""
    _ROOT = Path(__file__).resolve().parents[2]
    import sys as _sys
    if str(_ROOT / "src") not in _sys.path:
        _sys.path.insert(0, str(_ROOT / "src"))
    from equilibria.templates.gtap import GTAPParameters, GTAPModelEquations
    from equilibria.templates.gtap.altertax import apply_altertax_elasticities
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    har = _ROOT / "datasets" / dataset
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=har / "basedata.har", sets_path=har / "sets.har",
        default_path=har / "default.prm", baserate_path=har / "baserate.har",
    )
    p_alt = apply_altertax_elasticities(p, in_place=False)
    res = list(p_alt.sets.r)[-1]
    base_cl = GTAPClosureConfig(name="base", closure_type="MCP",
                                capital_mobility="sluggish", fix_endowments=False,
                                fix_taxes=False, fix_technology=False, if_sub=False,
                                numeraire="pnum")
    alt_cl = GTAPClosureConfig(name="altertax", closure_type="MCP",
                               capital_mobility="mobile", fix_endowments=False,
                               fix_taxes=True, fix_technology=True, if_sub=False,
                               numeraire="pnum")
    m_no_t0 = GTAPModelEquations(p_alt.sets, p_alt, alt_cl, residual_region=res).build_model()
    m_b = GTAPModelEquations(p_alt.sets, p_alt, base_cl, residual_region=res).build_model()
    m_t0 = GTAPModelEquations(p_alt.sets, p_alt, alt_cl, residual_region=res,
                              t0_snapshot=m_b).build_model()
    return m_no_t0, m_t0


def diff_param_builds(dataset: str, tol_rel: float = 1e-3) -> list:
    """Return Params whose value changes between build-without-t0 and build-with-t0.

    Auto-detects the construction-dependent universe (the shock-bug signature)
    without needing GAMS. Each entry: {param, cells, changed, max_rel, worst}.
    """
    m_a, m_b = _build_gtap7_har_models(dataset)
    pa, pb = extract_params(m_a), extract_params(m_b)
    changed = []
    for name in sorted(set(pa) & set(pb)):
        n_chg = 0
        max_rel = 0.0
        worst = None
        for idx, va in pa[name].items():
            vb = pb[name].get(idx)
            if vb is None:
                continue
            rel = abs(va - vb) / max(abs(vb), 1e-12)
            if rel > tol_rel:
                n_chg += 1
                if rel > max_rel:
                    max_rel = rel
                    worst = (idx, va, vb)
        if n_chg > 0:
            changed.append({"param": name, "cells": len(pa[name]), "changed": n_chg,
                            "max_rel": max_rel, "worst": worst})
    changed.sort(key=lambda r: r["max_rel"], reverse=True)
    return changed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/parity/test_probe.py::test_diff_param_builds_detects_t0_dependence -v`
Expected: PASS (or SKIP if fixture absent)

- [ ] **Step 5: Commit**

```bash
git add scripts/parity/_probe_params.py tests/parity/test_probe.py
git commit -m "feat(probe): diff_param_builds detects t0_snapshot-dependent Params"
```

---

## Task 4: Wire --params / --params-compare-builds into probe.py CLI

**Files:**
- Modify: `scripts/parity/probe.py`
- Test: `tests/parity/test_probe.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/parity/test_probe.py
def test_cli_params_runs(tmp_path):
    import pytest
    ref = ROOT / "output" / "gtap7_3x3_altertax_neos_bundle" / "out_local.gdx"
    if not ref.exists():
        pytest.skip("reference GDX absent")
    cmd = [
        "uv", "run", "python", "scripts/parity/probe.py",
        "--dataset", "gtap7_3x3", "--scenario", "altertax_check",
        "--params", "--gdx-ref", str(ref), "--cache-dir", str(tmp_path / "c"),
    ]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    if "not available" in (out.stdout + out.stderr):
        pytest.skip("gtap7_3x3 not available")
    assert out.returncode == 0, out.stderr
    assert "coverage:" in out.stdout
    assert "param" in out.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/parity/test_probe.py::test_cli_params_runs -v`
Expected: FAIL (no `--params` flag → `coverage:` not in output / nonzero exit)

- [ ] **Step 3: Add flags + dispatch to probe.py**

Add these arguments near the other `ap.add_argument` calls in `main()` (after the
`--seed-gams` group):

```python
    ap.add_argument("--params", action="store_true",
                    help="diff all Pyomo Params vs the GAMS GDX (3 groups)")
    ap.add_argument("--params-compare-builds", dest="params_compare_builds",
                    action="store_true",
                    help="report Params that change with/without t0_snapshot (no GAMS needed)")
    ap.add_argument("--params-period", dest="params_period", default=None,
                    help="GAMS period for --params (default: derived from scenario)")
```

Add the imports at the top of `probe.py` (with the other `_probe_*` imports):

```python
from _probe_params import diff_params_vs_gams, diff_param_builds  # noqa: E402
```

Add a scenario→period map near `_CLOSURE_NAME` (top of probe.py):

```python
_PARAMS_PERIOD = {
    "altertax_check": "check",
    "altertax_shock": "shock",
    "baseline": "base",
    "shock_tm10": "shock",
}
```

In `main()`, immediately after the `enumerate_combinations` check and BEFORE
`_build_and_value` / `--compare-ref`, add:

```python
    if args.params_compare_builds:
        changed = diff_param_builds(args.dataset, tol_rel=args.tol_rel)
        print(f"=== Params that change with/without t0_snapshot ({len(changed)}) ===")
        for r in changed[:args.top]:
            w = r["worst"]
            wc = f"{w[0]} no_t0={w[1]:.5f} t0={w[2]:.5f}" if w else ""
            print(f"  {r['param']:<16s} {r['changed']}/{r['cells']} changed "
                  f"max_rel={r['max_rel']:.3e}  {wc}")
        return 0

    if args.params:
        if not args.gdx_ref:
            print("error: --params requires --gdx-ref", file=sys.stderr)
            return 2
        model = adapter.build_warmstarted_model(args.dataset, args.scenario)
        period = args.params_period or _PARAMS_PERIOD.get(args.scenario, "base")
        res = diff_params_vs_gams(model, Path(args.gdx_ref), period, tol_rel=args.tol_rel)
        print(f"=== Param diff vs GAMS (period={period}) ===")
        print(f"{'param':<18s} {'cells':>6s} {'match':>6s} {'diverge':>8s} "
              f"{'max_rel':>10s}  worst")
        for r in res["diverge"][:args.top]:
            w = r["worst"]
            wc = f"{w[0]} py={w[1]:.5f} gams={w[2]:.5f}" if w else ""
            print(f"  {r['param']:<16s} {r['cells']:>6d} {r['match']:>6d} "
                  f"{r['diverge']:>8d} {r['max_rel']:>10.3e}  {wc}")
        n_verifiable = len(res["diverge"]) + len(res["ok"])
        print(f"coverage: {n_verifiable} params verifiable vs GAMS, "
              f"{len(res['no_match'])} with no GAMS counterpart")
        return 0
```

Note: `args.tol_rel` must exist. If `probe.py` has no `--tol-rel` yet, add:

```python
    ap.add_argument("--tol-rel", type=float, default=1e-3)
```

(check first with `grep -n "tol_rel\|tol-rel" scripts/parity/probe.py`; only add if absent).

- [ ] **Step 4: Run test to verify it passes**

Run: `export PATH="/Library/Frameworks/GAMS.framework/Versions/53/Resources:$PATH" && uv run pytest tests/parity/test_probe.py::test_cli_params_runs -v`
Expected: PASS (or SKIP if fixture absent)

- [ ] **Step 5: Commit**

```bash
git add scripts/parity/probe.py tests/parity/test_probe.py
git commit -m "feat(probe): --params and --params-compare-builds CLI flags"
```

---

## Task 5: Register the 5th tool in CLAUDE.md + skill doc note

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the tool row to CLAUDE.md**

Find the debug-tools cascade table in `CLAUDE.md` (section "Herramientas de debug
parity (cascade de 4)"). Add this row after the `.nl diff` row (row `| 3 |`):

```markdown
| 4 | **Param/calibration diff** | `probe.py --params` | Constantes calibradas (pf0, base_rgdpmp, p_gf, betap) que dependen del modo de construcción (t0_snapshot) | El solver converge a un equilibrio válido pero los **niveles** difieren y las otras 4 dicen "todo bien" |
```

And add this note after the "Pitfall clave" line:

```markdown

**Punto ciego cubierto por la 5ª herramienta:** las 4 primeras ven ecuaciones (.nl),
punto-solución (residual), warm-start (variables sembradas) y closure (fija/libre),
pero NINGUNA ve las **constantes horneadas en build** según `t0_snapshot` (pf0,
base_rgdpmp, p_gf, betap). Precedente: el shock de gtap7_3x3 convergía a otro
equilibrio porque las anclas Fisher dependían de t0_snapshot. Usar `probe.py
--params` (vs GAMS) y `--params-compare-builds` (auto-detecta las construcción-
dependientes). Lección extra: el closure diff hay que correrlo comparando AMBOS lados
(Python vs GAMS), no solo Python.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(CLAUDE): register Param/calibration diff as 5th cascade tool"
```

---

## Task 6: Full test run + end-to-end smoke

**Files:** none new

- [ ] **Step 1: Run the full probe test module**

Run: `export PATH="/Library/Frameworks/GAMS.framework/Versions/53/Resources:$PATH" && uv run pytest tests/parity/test_probe.py -v`
Expected: all PASS or SKIP (skips only where fixture GDX absent)

- [ ] **Step 2: End-to-end smoke — the shock-bug signature**

Run:
```bash
export PATH="/Library/Frameworks/GAMS.framework/Versions/53/Resources:$PATH"
uv run python scripts/parity/probe.py --dataset gtap7_3x3 \
  --scenario altertax_check --params-compare-builds --top 15
uv run python scripts/parity/probe.py --dataset gtap7_3x3 \
  --scenario altertax_check --params \
  --gdx-ref output/gtap7_3x3_altertax_neos_bundle/out_local.gdx --top 15
```
Expected: `--params-compare-builds` lists `pf0`/`base_rgdpmp`/`p_gf`/`betap` as
changing with t0; `--params` prints the diverge/ok groups and a `coverage:` line.

- [ ] **Step 3: Regression gate (must stay green)**

Run: `uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py -q`
Expected: `5 passed`

- [ ] **Step 4: Commit (notes only, if any)**

```bash
git commit --allow-empty -m "chore(probe): --params end-to-end smoke verified (gtap7_3x3)"
```
