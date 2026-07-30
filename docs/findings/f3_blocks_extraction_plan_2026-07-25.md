# F3 — GTAP to Symbolic Blocks (framework repair) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Move GTAP onto the repaired symbolic Block framework, with the north-star gate **gtap7_3x3 solves via the framework and matches GAMS (NLP+MCP)**.

**Architecture:** Repair the Pyomo bridge so it never silently drops equations; migrate all 7 GTAP block-units to `Block` subclasses with true CES/CET equations; assemble via a composer the bridge emits to Pyomo; keep the monolith as the parity oracle throughout.

**Tech Stack:** Python 3, Pyomo, `equilibria.blocks` (Block/SymbolicEquation), `equilibria.backends.pyomo_backend`, PATH C-API (via path-capi-python) + IPOPT, pytest (`-m integration`).

## Global Constraints

- **Spec:** `docs/findings/f3_blocks_extraction_spec_2026-07-25.md`.
- **FIDELITY IS SUPREME.** Any change in the model's economics is a bug. The monolith `gtap_model_equations.py` is the parity oracle and stays intact until the north-star gate is green.
- **The bridge must RAISE, never silently drop.** Removing the equation-swallowing `except` (pyomo_backend.py:233-238), the `dummy_constraint` (:296-297), and the legacy `pass` (:292-295) is the first repair — everything downstream depends on it.
- **Runner:** `uv run python` / `uv run python -m pytest`. Solver: inject `/Users/marmol/proyectos/path-capi-python/src` on sys.path + `import path_capi_python` (registers `path_capi_bridge`), per the measure tools pattern.
- **gitignore allowlist:** new `scripts/gtap/*.py` need a `!scripts/gtap/<f>.py` line.
- **ty ratchet:** tests with sys.path dynamic imports go in the `.pre-commit-config.yaml` exclude list.
- **Parity gates stamp:** `scripts/gtap/*` + `tests/...` changes require the full `run_parity_gates.py` sweep + stamp before push (PreToolUse hook enforces it).
- **Order of work (user choice):** migrate all 7 block-units, THEN first solve — BUT stand up diagnostic tooling (Task 2) BEFORE the first solve so a failure is debugged directed, not blind.

---

### Task 1: Make the bridge raise instead of silently dropping equations

The single most important repair. Today `pyomo_backend._build_constraints` swallows any equation that fails to build and stubs `dummy_constraint = 1==1` if none survive — a dropped equation is invisible, fatal for parity.

**Files:**
- Modify: `src/equilibria/backends/pyomo_backend.py:220-297`
- Test: `tests/backends/test_pyomo_backend_strict.py`

**Interfaces:**
- Produces: `PyomoBackend._build_constraints` now raises `BridgeTranslationError` (new exception in the module) on any un-buildable equation and on zero-constraint models; no `dummy_constraint`, no legacy `pass`-drop.

- [ ] **Step 1: Write the failing test**

```python
# tests/backends/test_pyomo_backend_strict.py
"""The bridge must RAISE on an un-translatable equation, not swallow it."""
from __future__ import annotations
import pytest
from equilibria.backends.pyomo_backend import PyomoBackend, BridgeTranslationError

def test_bridge_raises_on_unbuildable_equation():
    # a block whose build_expression raises must surface, not be skipped
    backend = _backend_with_one_failing_equation()  # helper below
    with pytest.raises(BridgeTranslationError):
        backend.build()

def test_bridge_raises_when_zero_constraints():
    backend = _backend_with_no_equations()
    with pytest.raises(BridgeTranslationError):
        backend.build()  # must NOT inject dummy_constraint=1==1
```
(Write `_backend_with_one_failing_equation` / `_backend_with_no_equations` as minimal fixtures: a `Model` with a block whose `build_expression` raises `ValueError`, and an empty model. Inspect `model.py:add_block` + `PyomoBackend.__init__` for the exact construction — mirror `tests/backends/test_pyomo_backend.py`'s setup.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m pytest tests/backends/test_pyomo_backend_strict.py -v`
Expected: FAIL — `BridgeTranslationError` doesn't exist; the bridge currently swallows + stubs.

- [ ] **Step 3: Implement — raise, no silent drop**

In `pyomo_backend.py`: define `class BridgeTranslationError(RuntimeError): ...`. In `_build_constraints`:
- Replace the `except (ValueError, KeyError, AttributeError, TypeError): … continue` (233-238) with: log, then `raise BridgeTranslationError(f"equation {eq_name}{indices} failed to build") from e`.
- Replace the legacy `else: pass` (292-295) with `raise BridgeTranslationError(f"equation {eq_name} has no build_expression (legacy closure form not supported)")`.
- Replace `if constraint_count == 0: dummy_constraint = 1==1` (296-297) with `raise BridgeTranslationError("no constraints were built — model would be trivially feasible")`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run python -m pytest tests/backends/test_pyomo_backend_strict.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the existing backend tests — nothing else broke**

Run: `uv run python -m pytest tests/backends/test_pyomo_backend.py -v`
Expected: PASS (or the one `solve()`-raises test still passes; if a test relied on the dummy_constraint, that test encoded the bug — update it to expect the raise and note why).

- [ ] **Step 6: Commit**

```bash
git add src/equilibria/backends/pyomo_backend.py tests/backends/test_pyomo_backend_strict.py
git commit -m "fix(blocks): bridge raises on untranslatable/zero equations (no silent drop, no dummy_constraint)"
```

---

### Task 2: Diagnostic tooling for the first solve (stand up BEFORE migrating blocks)

Because the framework has never solved a model and we migrate all 7 units at once, a failed 3x3 solve must be debuggable directed. Three diagnostics, reusing the existing cascade tools where possible.

**Files:**
- Create: `scripts/gtap/blocks_diag.py`
- Modify: `.gitignore` (`!scripts/gtap/blocks_diag.py`)
- Test: `tests/templates/gtap/test_blocks_diag.py`

**Interfaces:**
- Produces:
  - `residual_report(pyomo_model, seed_gdx) -> list[(eq_name, index, residual)]` sorted by |residual| desc — which equation is violated at the seeded point.
  - `form_diff(block, monolith_model) -> list[(index, block_expr, monolith_expr)]` — per-cell expanded-Pyomo-expr diff of a B-block vs the monolith's constraint (wraps/reuses `scripts/gtap/diff_equation_form.py`, cascade tool 5).
  - `domain_bounds_diff(block_model, monolith_model) -> list[(var, index, b_domain, mono_domain, b_bounds, mono_bounds)]` — per-var domain/bounds mismatches.

- [ ] **Step 1: Write the failing test**

```python
# tests/templates/gtap/test_blocks_diag.py
"""Diagnostic tools must surface a planted residual / form / domain mismatch."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/gtap")); sys.path.insert(0, str(ROOT / "src"))

def test_residual_report_ranks_worst_first():
    from blocks_diag import residual_report
    # a tiny pyomo model with one satisfied + one violated constraint at a point
    m = _toy_model_one_violation()
    rep = residual_report(m, seed=None)
    assert rep[0][2] > rep[-1][2]          # sorted worst-first
    assert rep[0][0] == "eq_bad"           # the violated one is on top

def test_domain_bounds_diff_flags_mismatch():
    from blocks_diag import domain_bounds_diff
    a = _toy_model(domain="Reals")
    b = _toy_model(domain="NonNegativeReals")
    diffs = domain_bounds_diff(a, b)
    assert any(d[0] == "x" for d in diffs)
```
(Write `_toy_model*` as ~10-line Pyomo models. `form_diff` is exercised once a real block exists — Task 4 — so its test can be a `pytest.skip` placeholder here or a synthetic two-expression compare.)

- [ ] **Step 2: Run to verify fails**

Run: `uv run python -m pytest tests/templates/gtap/test_blocks_diag.py -v`
Expected: FAIL — `blocks_diag` doesn't exist.

- [ ] **Step 3: Implement the three diagnostics**

Write `scripts/gtap/blocks_diag.py`:
- `residual_report`: iterate active `Constraint`s, evaluate `body - lower`/`upper` at current values (seed from GDX if given via the existing seeding helper), return sorted list.
- `domain_bounds_diff`: iterate `Var`s in both models by name, compare `.domain` and `.bounds`.
- `form_diff`: import and wrap `diff_equation_form.py`'s expansion for a single named constraint.

- [ ] **Step 4: gitignore + run to pass**

Add `!scripts/gtap/blocks_diag.py` to `.gitignore`.
Run: `uv run python -m pytest tests/templates/gtap/test_blocks_diag.py -v`
Expected: PASS (form_diff test may skip until Task 4).

- [ ] **Step 5: Commit**

```bash
git add scripts/gtap/blocks_diag.py tests/templates/gtap/test_blocks_diag.py .gitignore
git commit -m "tools(F3): blocks_diag — residual/form/domain diagnostics for the first block solve"
```

---

### Task 3: Resolve the symbolic API (single build_expression contract)

The ABC and the real blocks disagree on `build_expression`'s signature; the `ResidualEquation` DSL is dead. Standardize on ONE contract before writing GTAP blocks against it.

**Files:**
- Modify: `src/equilibria/core/symbolic_equations.py`, `src/equilibria/blocks/base.py`
- Test: `tests/blocks/test_symbolic_contract.py`

**Interfaces:**
- Produces: a single `SymbolicEquation.build_expression(self, pyomo_model, indices) -> expr | None` contract (the form the real blocks + bridge already use). Dead `ResidualEquation`/`var/param/power` combinator code is removed (or, if kept, made to route through the same contract — decide by what GTAP needs; default: remove, YAGNI).

- [ ] **Step 1: Write the failing test**

```python
# tests/blocks/test_symbolic_contract.py
"""There is ONE build_expression contract, and the bridge consumes it."""
def test_single_build_expression_signature():
    import inspect
    from equilibria.core.symbolic_equations import SymbolicEquation
    sig = inspect.signature(SymbolicEquation.build_expression)
    assert list(sig.parameters)[1:] == ["pyomo_model", "indices"]

def test_dead_dsl_removed_or_routed():
    import equilibria.core.symbolic_equations as se
    # ResidualEquation, if present, must implement the same (pyomo_model, indices) contract
    if hasattr(se, "ResidualEquation"):
        import inspect
        sig = inspect.signature(se.ResidualEquation.build_expression)
        assert list(sig.parameters)[1:] == ["pyomo_model", "indices"]
```

- [ ] **Step 2: Run to verify fails** — `uv run python -m pytest tests/blocks/test_symbolic_contract.py -v` (FAIL: ABC signature is the 4-arg residual form).

- [ ] **Step 3: Implement** — change the ABC `build_expression` to `(self, pyomo_model, indices)`; remove the dead `ResidualEquation` + combinator DSL (or align it). Update `blocks/base.py` docstrings/`setup` return type accordingly. Keep `CalibrationMixin` untouched (it is the sound part).

- [ ] **Step 4: Run to pass** + run the existing block tests: `uv run python -m pytest tests/blocks/ -v` (nothing else broke).

- [ ] **Step 5: Commit** — `git commit -m "refactor(blocks): single build_expression contract, retire dead ResidualEquation DSL"`

---

### Task 4: Migrate the 7 GTAP block-units to Block subclasses (true CES/CET)

The bulk. Each of the 7 dependency units becomes a `Block` subclass in `blocks/gtap/`, with its variables (distributed to owner) and its equations as `build_expression` bodies carrying the TRUE CES/CET forms — moved from the monolith, not re-derived. Circular bundles stay merged.

**Files:**
- Create: `src/equilibria/blocks/gtap/{trade_cet,production_supply,factor,trade_armington_bilateral,demand_utility,income,closure}.py`
- Create: `src/equilibria/blocks/gtap/__init__.py`
- Modify: `.gitignore` if needed (these are under `src/`, tracked by default)
- Test: `tests/templates/gtap/test_gtap_blocks_form.py`

**Interfaces:**
- Consumes: the monolith's equation bodies (the parity oracle); `blocks_diag.form_diff` (Task 2); the single contract (Task 3).
- Produces: 7 `Block` subclasses; a `blocks/gtap/__init__.py` exposing them + a `GTAP_BLOCK_ORDER` list in dependency order (TRADE_CET, PRODUCTION_SUPPLY, FACTOR, ARMINGTON_BILATERAL, DEMAND_UTILITY, INCOME, CLOSURE).

For EACH unit, in dependency order, do the sub-cycle (Steps A–D). Written once here; repeat per unit.

- [ ] **Step A: Write the block** — port the unit's Var declarations (its owned vars, with EXACT domain/bounds from the monolith) + its constraints as `build_expression` bodies. Preserve every conditional/Skip/`param.get(key, default)`/`**omega` exactly (they are expressible — the callable runs arbitrary Python returning a Pyomo expr or `None` for skip). Merge circular bundles into one Block.

- [ ] **Step B: Form-diff gate (no solve)** — `blocks_diag.form_diff(block, monolith)` for every cell == the monolith's expanded expression. Must be clean before proceeding. This is the cheap catch for translation drift.

- [ ] **Step C: Domain+bounds gate (no solve)** — `blocks_diag.domain_bounds_diff` shows zero mismatches for the unit's vars.

- [ ] **Step D: Commit the unit** — `git commit -m "blocks(F3): <unit> as symbolic Block (form+domain diff clean vs monolith)"`

Do TRADE_CET first (leaf), CLOSURE last. After all 7:

```python
# tests/templates/gtap/test_gtap_blocks_form.py — the aggregate form gate
def test_all_gtap_blocks_form_match_monolith():
    # for each block in GTAP_BLOCK_ORDER, every cell's expanded expr == monolith's
    ...
```

- [ ] **Final step of Task 4:** `uv run python -m pytest tests/templates/gtap/test_gtap_blocks_form.py -v` PASS for all 7 units; commit.

---

### Task 5: Compose the blocks into a solvable model + first 3x3 solve (north-star gate)

Assemble the 7 blocks via the registry into a model the repaired bridge emits to Pyomo, and solve gtap7_3x3 — the north-star. This is where the accepted all-at-once bisect risk lands; Task 2's diagnostics are the safety net.

**Files:**
- Create: `src/equilibria/templates/gtap/gtap_block_model.py` (composer)
- Test: `tests/templates/gtap/test_gtap_blocks_solve.py`

**Interfaces:**
- Consumes: the 7 blocks + `GTAP_BLOCK_ORDER` (Task 4); the repaired bridge (Task 1); `blocks_diag` (Task 2); the seeding + solve path from `gtap_multiperiod_driver`.
- Produces: `build_block_model(dataset, ifsub) -> pyomo_model` and a solve returning code + values comparable to the monolith/GAMS.

- [ ] **Step 1: Write the failing test (north-star)**

```python
# tests/templates/gtap/test_gtap_blocks_solve.py
"""gtap7_3x3 solves via the symbolic block framework and matches GAMS."""
pytestmark = pytest.mark.integration
def test_3x3_via_blocks_matches_gams():
    from gtap_block_model import build_block_model, solve_block_model
    m = build_block_model("gtap7_3x3", ifsub=1)
    code = solve_block_model(m, seed_gdx=REF_3X3)
    assert code == 1
    within = _measure_vs_gams(m, REF_3X3)   # reuse the existing pp/rel measure
    assert within >= 0.99
```

- [ ] **Step 2: Run — expect it to fail (first solve of a never-solved framework)**

Run: `uv run python -m pytest tests/templates/gtap/test_gtap_blocks_solve.py -v -m integration`
Expected: FAIL initially. **Use `blocks_diag` to debug directed:** residual_report → which equation; form_diff → is its form wrong; domain_bounds_diff → is a var boxed wrong. Fix the offending block/bridge, re-run. Iterate until code=1 + match. (This is the accepted-risk debugging loop; the diagnostics make it directed.)

- [ ] **Step 3: Canary green, then the form/domain gates for the whole model** — confirm all four gate layers (form, domain, canary 3x3, then Task 6's full sweep).

- [ ] **Step 4: Commit** — `git commit -m "blocks(F3): compose 7 GTAP blocks; gtap7_3x3 solves via framework, matches GAMS (north-star green)"`

---

### Task 6: Full sweep across datasets + mandatory parity gates + PR

Extend from 3x3 to the rest, run the mandatory sweep, open the PR.

- [ ] **Step 1: Run 3x4…15x10 via blocks** — the same solve+measure for each dataset (NLP+MCP); each ≥ its matrix floor. Debug with `blocks_diag` as needed. (15x10 is slow; background it.)
- [ ] **Step 2: Mandatory parity gates** — `uv run python scripts/gtap/run_parity_gates.py` — full NLP+MCP sweep + measured-docs regen + stamp. MUST be green; if any cell regresses vs the monolith, STOP and fix (the monolith is the oracle).
- [ ] **Step 3: Finding + roadmap** — write `docs/findings/f3_blocks_done_2026-07-25.md` (what the framework repair took, the north-star result, any framework limitations found); update the roadmap F3 status + memory pointer.
- [ ] **Step 4: Push + PR** — `git push -u origin gtap/f3-blocks-extraction; gh pr create --base main --title "gtap(F3): GTAP on the repaired symbolic block framework — 3x3+ via blocks matches GAMS" --body "<repair summary, north-star result, gates>"`
- [ ] **Step 5: Finish the branch** — REQUIRED SUB-SKILL: superpowers:finishing-a-development-branch.

---

## Self-Review

**Spec coverage:** Bridge repair (spec item 1) → Task 1. Symbolic API (item 2) → Task 3. 7 GTAP blocks with true CES/CET (item 3) → Task 4. Real solve+parity test suite (item 4) → Tasks 5-6. Diagnostic tooling (mandatory mitigation) → Task 2, stood up BEFORE the first solve (Task 5). North-star gate (3x3 via B matches GAMS) → Task 5. 4-layer gate → Tasks 4 (form+domain) + 5 (canary) + 6 (sweep). Monolith-as-oracle → used in Tasks 4-6. All-at-once order → Task 4 migrates all 7 before Task 5's first solve.

**Placeholder scan:** The toy-model fixtures in Task 2/3 tests are described by shape, not spelled out (they're ~10-line Pyomo models the implementer writes against the cited existing test files) — acceptable as they're standard Pyomo scaffolding, not domain logic. The per-unit Step A-D sub-cycle in Task 4 is written once and repeated per unit (7×) rather than duplicated — the units and their line ranges come from the dependency map in the spec. No "add error handling"/"TBD".

**Type consistency:** `BridgeTranslationError` (Task 1) used in Task 5's debug loop. `blocks_diag`'s three functions (Task 2) consumed by Task 4 (form/domain) and Task 5 (residual). `GTAP_BLOCK_ORDER` defined in Task 4, consumed by Task 5's composer. `build_block_model`/`solve_block_model` defined in Task 5, used in its test. The single `build_expression(pyomo_model, indices)` contract (Task 3) is what Task 4's blocks implement.
