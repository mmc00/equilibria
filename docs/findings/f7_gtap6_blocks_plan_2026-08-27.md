# F7 — GTAP 6.2 Template on Symbolic Blocks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a GTAP version 6.2 (Hertel/Itakura/McDougall 2003) CGE
template in `equilibria`, built on the same symbolic `Block` framework F3
proved for GTAP7 (≥99% vs GAMS), reaching ≤1% gap vs GEMPACK on
`gtap6_3x3` → `gtap6_15x10`.

**Architecture:** Port `templates/gtap_v62/{sets,parameters,calibration,contract}.py`
verbatim (API-adapted) from the orphan branch `gtap/v62-multiperiod` into
`templates/gtap6/` — they already solve HAR v6.2a reading, SLUG-based factor
mobility, and SAM-consistent calibration and don't depend on block
architecture. Rewrite only the 2055-line monolithic
`gtap_v62_model_equations.py` as 5 new `Block` subclasses in
`blocks/gtap6/` (leaf→closure order: `TradeArmington` → `Production` →
`Factor` → `DemandUtility` → `IncomeClosure`), composed by a new
`gtap6_block_model.py` mirroring `templates/gtap/gtap_block_model.py`'s
composer pattern. A temporary copy of the monolith serves as the parity
oracle for the 4-layer gate (form-diff → domain/bounds → canary solve →
full GEMPACK sweep), one dataset at a time.

**Tech Stack:** Python, Pyomo (via `equilibria.blocks.base.Block` +
`equilibria.core.symbolic_equations.SymbolicEquation`), pytest, IPOPT (NLP)
+ PATH (MCP) solvers, GTAP HAR/GDX readers (`equilibria.babel.har`,
`equilibria.babel.gdx`).

**Spec:** [`f7_gtap6_blocks_spec_2026-08-27.md`](f7_gtap6_blocks_spec_2026-08-27.md)

## Global Constraints

- Gate ≤1% gap vs GEMPACK Gragg-multi (NLP+MCP), Walras < 1e-6, per dataset
  — replicates, does not relax, the 0.06–0.64% already measured by the
  orphan-branch prototype (Phase 3.38).
- Dataset order: `gtap6_3x3` → `gtap6_5x5` → `gtap6_10x7` → `gtap6_15x10`,
  green gate before advancing. `gtap6_20x41` is OUT OF SCOPE (documented
  MUMPS 32-bit stack limit, not a model defect) — no task in this plan
  targets it.
- `blocks/gtap6/` does NOT share instances with `blocks/gtap/` (GTAP7) —
  v6.2 has no make-matrix, no ND intermediate bundle, no output CET,
  `cgds` is a producing sector not an agent, single aggregate tax stream
  (6 components) not 10.
- Known bugs to avoid (from `docs/findings/gtap_v62_phase338_*.md`, ported
  as design constraints, not literal code):
  - `sav` (regional savings) MUST be a Pyomo `Var`, never a `Param` —
    budget identity `y = yp + yg + sav` must close under shock.
  - VIWS parity metric (if any new comparison script is written) is
    `qxs * pmcif` (CIF/world price), never `qxs * pms` (agent price).
- No MRIO, no NTM AVE, no dynamics, no shared class hierarchy with
  `blocks/gtap/` — see spec Non-goals.
- Monolith oracle (`scripts/gtap6/_v62_monolith_oracle.py`) is never
  imported by `templates/gtap6/` or `blocks/gtap6/` — test-only, deleted
  once all 4 datasets pass Task 12.

---

## File Structure

```
src/equilibria/templates/gtap6/
  __init__.py
  gtap6_sets.py           # Task 1 — ported from gtap_v62_sets.py
  gtap6_parameters.py     # Task 2 — ported from gtap_v62_parameters.py
  gtap6_calibration.py    # Task 3 — ported from gtap_v62_calibration.py
  gtap6_contract.py       # Task 4 — ported from gtap_v62_contract.py
  gtap6_block_model.py    # Task 10 — NEW composer
  gtap6_solver.py         # Task 11 — ported from gtap_v62_solver.py

src/equilibria/blocks/gtap6/
  __init__.py              # Task 10 — GTAP6_BLOCK_ORDER
  trade_armington.py        # Task 6 — leaf unit
  production.py              # Task 7
  factor.py                   # Task 8
  demand_utility.py            # Task 9a
  income_closure.py            # Task 9b — last (closure)

scripts/gtap6/
  _v62_monolith_oracle.py  # Task 5 — copied from orphan branch, test-only

tests/templates/gtap6/
  test_gtap6_sets.py           # Task 1
  test_gtap6_parameters.py     # Task 2
  test_gtap6_calibration.py    # Task 3
  test_gtap6_contract.py       # Task 4
  test_gtap6_blocks_solve.py   # Task 10 (canary)
  test_gtap6_gempack_parity.py # Task 12 (final gate)

tests/blocks/gtap6/
  test_gtap6_blocks_form.py    # Task 6-9 (per-block form/domain gate)
```

---

### Task 1: Port GTAP6 sets module

**Files:**
- Create: `src/equilibria/templates/gtap6/__init__.py`
- Create: `src/equilibria/templates/gtap6/gtap6_sets.py`
- Test: `tests/templates/gtap6/test_gtap6_sets.py`

**Interfaces:**
- Produces: `GTAP6Sets` dataclass with fields `r, i, cgds, f, marg: list[str]`,
  `mf, sf: list[str]` (mobility partition), `m, s: list[str]` (aliases),
  `aggregation_name: str`, `source_path: Path | None`. Methods:
  `load_from_har(sets_path: Path, default_path: Path | None = None) -> None`,
  properties `a` (alias of `i`), `prod_comm`, `demd_comm`, `nsav_comm`,
  `is_diagonal` (always `True`), `n_regions`, `n_commodities`, `n_factors`,
  `n_mobile_factors`, `n_sluggish_factors`, `validate() -> tuple[bool, list[str]]`,
  `get_info() -> dict[str, Any]`.
- Consumes: `equilibria.babel.har.read_har`.

Port `gtap_v62_sets.py` (342 lines, orphan branch `gtap/v62-multiperiod`)
to `gtap6_sets.py`, renaming `GTAPv62Sets` → `GTAP6Sets` and updating the
module docstring's references from `templates.gtap_v62` to
`templates.gtap6`. The logic (header-candidate lookup for
`H1/H2/H6/H9`, the `SLUG` heuristic for mobile/sluggish partition, the
`prod_comm`/`demd_comm`/`nsav_comm` derived sets) is already correct —
change only names/imports, not behavior.

- [ ] **Step 1: Fetch the source file from the orphan branch**

```bash
git show gtap/v62-multiperiod:src/equilibria/templates/gtap_v62/gtap_v62_sets.py > /tmp/gtap_v62_sets.py
```

- [ ] **Step 2: Create the package `__init__.py`**

```python
"""GTAP 6.2 CGE template (Hertel/Itakura/McDougall 2003).

Built on the symbolic `equilibria.blocks` framework (see
`equilibria.blocks.gtap6`), reusing the pattern F3 proved for GTAP7 but
with 6.2's own block units — v6.2 has no make-matrix, no ND intermediate
bundle, no output CET, and `cgds` is a producing sector, not an agent.
"""

from __future__ import annotations

__all__: list[str] = []
```

- [ ] **Step 3: Write `gtap6_sets.py`**

Take `/tmp/gtap_v62_sets.py` verbatim and apply these renames only:
- Module docstring: `GTAP v6.2 Sets` stays the same content but update the
  class name references.
- `class GTAPv62Sets` → `class GTAP6Sets`.
- Docstring cross-references `templates.gtap.gtap_sets.GTAPSets` (unchanged
  — still a correct comparison target).
- `__repr__` returns `f"GTAP6Sets({self.aggregation_name}: ...)"` (was
  `GTAPv62Sets(...)`).

No other line changes — the header-candidate tuples (`_REG_HEADERS`,
`_TRAD_COMM_HEADERS`, `_ENDW_COMM_HEADERS`, `_CGDS_COMM_HEADERS`,
`_MARG_HEADERS`), the `_load_slug` heuristic, and all properties/validation
are correct as-is (verified against `datasets/gtap6_3x3/sets.har` directly
during spec research: `H1=REG`, `H2=TRAD_COMM`, `H6=ENDW_COMM`,
`H9=CGDS_COMM`, `COMM==ACTS` confirming no make-matrix).

- [ ] **Step 4: Write the failing test**

```python
"""GTAP6Sets loads datasets/gtap6_3x3 correctly."""
from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.templates.gtap6.gtap6_sets import GTAP6Sets

DATASET = Path(__file__).resolve().parents[3] / "datasets" / "gtap6_3x3"


def test_load_from_har_gtap6_3x3():
    sets = GTAP6Sets()
    sets.load_from_har(DATASET / "sets.har", default_path=DATASET / "default.prm")

    assert sets.r == ["USA", "EU", "ROW"] or len(sets.r) == 3
    assert len(sets.i) == 3
    assert sets.a == sets.i  # alias property, no ACT/COMM split
    assert sets.is_diagonal is True
    assert len(sets.f) >= 1
    is_valid, errors = sets.validate()
    assert is_valid, errors


def test_mobile_sluggish_partition_covers_all_factors():
    sets = GTAP6Sets()
    sets.load_from_har(DATASET / "sets.har", default_path=DATASET / "default.prm")

    assert set(sets.mf) | set(sets.sf) == set(sets.f)
    assert set(sets.mf) & set(sets.sf) == set()
```

- [ ] **Step 5: Run test to verify it fails**

Run: `uv run pytest tests/templates/gtap6/test_gtap6_sets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'equilibria.templates.gtap6'`

- [ ] **Step 6: Create `gtap6_sets.py` per Step 3, run test again**

Run: `uv run pytest tests/templates/gtap6/test_gtap6_sets.py -v`
Expected: PASS (2 tests). If region names in the dataset differ from
`["USA", "EU", "ROW"]`, relax the first assertion to just `len(sets.r) == 3`
before committing — read the actual `sets.har` content if unsure:
`uv run python -c "from equilibria.babel.har import read_har; from pathlib import Path; d = read_har(Path('datasets/gtap6_3x3/sets.har')); print(d['H1'].array if hasattr(d['H1'], 'array') else d['H1'])"`

- [ ] **Step 7: Commit**

```bash
git add src/equilibria/templates/gtap6/__init__.py src/equilibria/templates/gtap6/gtap6_sets.py tests/templates/gtap6/test_gtap6_sets.py
git commit -m "feat(gtap6): port GTAP6Sets from orphan branch gtap/v62-multiperiod"
```

---

### Task 2: Port GTAP6 parameters module

**Files:**
- Create: `src/equilibria/templates/gtap6/gtap6_parameters.py`
- Test: `tests/templates/gtap6/test_gtap6_parameters.py`

**Interfaces:**
- Consumes: `GTAP6Sets` (Task 1) — `sets.i`, `sets.r`, `sets.f`, `sets.prod_comm`.
- Produces: `GTAP6Elasticities` dataclass (`esubd, esubm, esubt, esubva,
  etrae, rorflex, slug: dict[str, float]`, `incpar, subpar: dict[tuple[str,str], float]`,
  method `load_from_har(default_path: Path, sets: GTAP6Sets) -> None`) and
  a benchmark-values dataclass (name it `GTAP6BenchmarkValues`, ported from
  whatever class holds `VDFA/VDFM/VIFA/...` in the orphan file — inspect
  Step 1's output to confirm the exact class name before renaming) with
  method `load_from_har(basedata_path: Path, sets: GTAP6Sets) -> None`.
  A top-level `GTAP6Parameters` dataclass bundling both plus `validate()
  -> tuple[bool, list[str]]`, matching the shape of
  `equilibria.templates.gtap.gtap_parameters.GTAPParameters` (v7) closely
  enough that downstream code (`gtap6_calibration.py`) can consume it the
  same way `gtap_v62_calibration.py` already does.

- [ ] **Step 1: Fetch and inspect the source file**

```bash
git show gtap/v62-multiperiod:src/equilibria/templates/gtap_v62/gtap_v62_parameters.py > /tmp/gtap_v62_parameters.py
wc -l /tmp/gtap_v62_parameters.py
grep -n "^class \|^@dataclass" /tmp/gtap_v62_parameters.py
```

Read the full class list this prints (expect `GTAPv62Elasticities` plus at
least one benchmark-values class and a top-level `GTAPv62Parameters`
bundling class) — the exact benchmark-values class name was not
transcribed in the design spec, so confirm it here before the rename pass.

- [ ] **Step 2: Write `gtap6_parameters.py`**

Copy `/tmp/gtap_v62_parameters.py` to
`src/equilibria/templates/gtap6/gtap6_parameters.py` and apply a
mechanical rename pass:
- `GTAPv62Elasticities` → `GTAP6Elasticities`
- `GTAPv62Parameters` → `GTAP6Parameters` (or whatever the top-level
  bundling class is actually named per Step 1)
- Any benchmark-values class found in Step 1 → prefix `GTAP6` instead of
  `GTAPv62`
- `from equilibria.templates.gtap_v62.gtap_v62_sets import GTAPv62Sets` →
  `from equilibria.templates.gtap6.gtap6_sets import GTAP6Sets`, and every
  `GTAPv62Sets` type annotation in this file → `GTAP6Sets`.

Do not change the HAR array names being read (`VDFA`, `VDFM`, `VIFA`,
`VIFM`, `VXMD`, `VXWD`, `VIWS`, `VIMS`, `EVFA`, `EVOA`, `VKB`, `VST`,
`VTWR`, `FBEP`, `FTRV`, `MFRV`, `TFRV`, `XTRV`, `ADRV`, `PTAX`, `PURV`,
`CSEP`, `ISEP`, `DPSM`, `SAVE`, `POP`, `VDEP`, `ESBD`, `ESBM`, `ESBT`,
`ESBV`, `ETRE`, `RFLX`, `SLUG`, `INCP`, `SUBP`) — these are the v6.2a HAR
schema and must stay exactly as read from `datasets/gtap6_*/basedata.har`.

- [ ] **Step 3: Write the failing test**

```python
"""GTAP6Parameters loads datasets/gtap6_3x3 correctly."""
from __future__ import annotations

from pathlib import Path

from equilibria.templates.gtap6.gtap6_parameters import GTAP6Parameters
from equilibria.templates.gtap6.gtap6_sets import GTAP6Sets

DATASET = Path(__file__).resolve().parents[3] / "datasets" / "gtap6_3x3"


def _load_sets() -> GTAP6Sets:
    sets = GTAP6Sets()
    sets.load_from_har(DATASET / "sets.har", default_path=DATASET / "default.prm")
    return sets


def test_load_from_har_gtap6_3x3():
    sets = _load_sets()
    params = GTAP6Parameters()
    params.load_from_har(DATASET / "basedata.har", DATASET / "default.prm", sets)

    is_valid, errors = params.validate()
    assert is_valid, errors
    # Benchmark output value for at least one (region, commodity) must be positive.
    assert any(v > 0 for v in params.benchmark.vdfm.values())
```

Adjust the `params.load_from_har(...)` call signature and
`params.benchmark.vdfm` attribute path once Step 1/2 confirm the actual
method signature and nested-object structure — this is a starting point,
not a byte-exact prediction of the ported file's API.

- [ ] **Step 4: Run test to verify it fails, then implement, then pass**

Run: `uv run pytest tests/templates/gtap6/test_gtap6_parameters.py -v`
Iterate Step 2/3 together until green — the exact attribute paths depend
on what Step 1 discovers, so treat Step 3's test as a live document to
correct against the real ported API, not a fixed target.

- [ ] **Step 5: Commit**

```bash
git add src/equilibria/templates/gtap6/gtap6_parameters.py tests/templates/gtap6/test_gtap6_parameters.py
git commit -m "feat(gtap6): port GTAP6Parameters (v6.2a HAR schema) from orphan branch"
```

---

### Task 3: Port GTAP6 calibration module

**Files:**
- Create: `src/equilibria/templates/gtap6/gtap6_calibration.py`
- Test: `tests/templates/gtap6/test_gtap6_calibration.py`

**Interfaces:**
- Consumes: `GTAP6Sets` (Task 1), `GTAP6Parameters` (Task 2).
- Produces: a `derive_calibration(sets: GTAP6Sets, params: GTAP6Parameters)
  -> DerivedV62Calibration`-shaped function/dataclass (rename
  `DerivedV62Calibration` → `DerivedGTAP6Calibration`), exposing calibrated
  tax rates, benchmark aggregates, and CDE calibration parameters that
  `blocks/gtap6/*.py` (Tasks 6-9) will read the same way
  `gtap_v62_model_equations.py` already does in the orphan branch.

- [ ] **Step 1: Fetch the source file**

```bash
git show gtap/v62-multiperiod:src/equilibria/templates/gtap_v62/gtap_v62_calibration.py > /tmp/gtap_v62_calibration.py
grep -n "^class \|^def " /tmp/gtap_v62_calibration.py
```

- [ ] **Step 2: Write `gtap6_calibration.py`**

Copy verbatim, then rename-pass:
- `DerivedV62Calibration` → `DerivedGTAP6Calibration`
- `derive_calibration` stays (already generic)
- Every `from equilibria.templates.gtap_v62...` import → `from
  equilibria.templates.gtap6...`
- Every `GTAPv62Sets`/`GTAPv62Parameters` type annotation → `GTAP6Sets`/`GTAP6Parameters`

Do not change the economic logic (tax-rate derivation, CDE `incpar`/`subpar`
handling, SAM-consistency adjustments) — this is the code that reached
0.06–0.64% gap vs GEMPACK in the orphan branch; only names change.

- [ ] **Step 3: Write the failing test**

```python
"""derive_calibration produces a consistent GTAP6 calibration for gtap6_3x3."""
from __future__ import annotations

from pathlib import Path

from equilibria.templates.gtap6.gtap6_calibration import derive_calibration
from equilibria.templates.gtap6.gtap6_parameters import GTAP6Parameters
from equilibria.templates.gtap6.gtap6_sets import GTAP6Sets

DATASET = Path(__file__).resolve().parents[3] / "datasets" / "gtap6_3x3"


def test_derive_calibration_gtap6_3x3():
    sets = GTAP6Sets()
    sets.load_from_har(DATASET / "sets.har", default_path=DATASET / "default.prm")
    params = GTAP6Parameters()
    params.load_from_har(DATASET / "basedata.har", DATASET / "default.prm", sets)

    derived = derive_calibration(sets, params)

    assert derived is not None
    # Regional income must be positive for every region once calibrated.
    for r in sets.r:
        assert derived.y0[r] > 0.0
```

Adjust `derived.y0` to whatever the ported module actually names regional
benchmark income (confirm via Step 1's `grep` output on field names before
finalizing this assertion).

- [ ] **Step 4: Run test, iterate to green**

Run: `uv run pytest tests/templates/gtap6/test_gtap6_calibration.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/equilibria/templates/gtap6/gtap6_calibration.py tests/templates/gtap6/test_gtap6_calibration.py
git commit -m "feat(gtap6): port GTAP6 calibration (SAM-consistent tax/CDE derivation)"
```

---

### Task 4: Port GTAP6 contract module

**Files:**
- Create: `src/equilibria/templates/gtap6/gtap6_contract.py`
- Test: `tests/templates/gtap6/test_gtap6_contract.py`

**Interfaces:**
- Produces: `GTAP6ClosureConfig(ModelClosureConfig)` (numeraire
  `"pgdpwld"`, `rordelta: bool = True`, `if_sub: bool = False`, `fixed:
  tuple[str, ...]`, `endogenous: tuple[str, ...]`), `GTAP6EquationConfig
  (ModelEquationConfig)`, `GTAP6BoundsConfig(ModelBoundsConfig)`,
  `GTAP6Contract(ModelContract)`, `build_gtap6_contract(closure_name: str =
  "gtap6_standard") -> GTAP6Contract`, `default_gtap6_contract() ->
  GTAP6Contract`. Equation-ID tuples `_GTAP6_PRODUCTION`,
  `_GTAP6_FINAL_DEMAND`, `_GTAP6_TRADE`, `_GTAP6_MARGINS`,
  `_GTAP6_FACTOR_MARKETS`, `_GTAP6_INCOME_AND_CLOSURE` — these are the
  authoritative list of equation names Tasks 6-9's blocks must produce.

- [ ] **Step 1: Fetch the source file**

```bash
git show gtap/v62-multiperiod:src/equilibria/templates/gtap_v62/gtap_v62_contract.py > /tmp/gtap_v62_contract.py
```

- [ ] **Step 2: Write `gtap6_contract.py`**

Copy verbatim, rename pass:
- `_V62_PRODUCTION` → `_GTAP6_PRODUCTION` (and the 5 sibling tuples,
  `_V62_FINAL_DEMAND`, `_V62_TRADE`, `_V62_MARGINS`,
  `_V62_FACTOR_MARKETS`, `_V62_INCOME_AND_CLOSURE` similarly)
- `_full_gtap_v62_equation_ids` → `_full_gtap6_equation_ids`
- `GTAPv62ClosureConfig` → `GTAP6ClosureConfig`
- `GTAPv62EquationConfig` → `GTAP6EquationConfig`
- `GTAPv62BoundsConfig` → `GTAP6BoundsConfig`
- `GTAPv62Contract` → `GTAP6Contract`
- `name: str = "gtap_v62_standard"` → `name: str = "gtap6_standard"`
  (also update the 3 string-literal comparisons in `_closure_for` and the
  docstring's `"gtap_v62_standard"` mentions)
- `build_gtap_v62_contract` → `build_gtap6_contract`
- `default_gtap_v62_contract` → `default_gtap6_contract`
- `__all__` list entries renamed to match

Equation IDs themselves (`e_qo`, `e_ps`, `e_qf`, ... all ~50 entries across
the 6 tuples) do NOT change — Tasks 6-9's blocks are named to produce
exactly these IDs.

- [ ] **Step 3: Write the failing test**

```python
"""GTAP6Contract builds a valid standard closure."""
from __future__ import annotations

from equilibria.templates.gtap6.gtap6_contract import (
    build_gtap6_contract,
    default_gtap6_contract,
)


def test_default_contract_is_standard_closure():
    contract = default_gtap6_contract()
    assert contract.closure.name == "gtap6_standard"
    assert contract.closure.numeraire == "pgdpwld"
    assert contract.closure.if_sub is False


def test_full_equation_ids_include_production_and_closure():
    contract = build_gtap6_contract("gtap6_standard")
    ids = set(contract.equations.include)
    assert "e_qo" in ids
    assert "e_walras" in ids
    assert "e_pgdpwld" in ids


def test_trade_policy_closure_frees_tariffs():
    contract = build_gtap6_contract("trade_policy")
    assert "tm" not in contract.closure.fixed
    assert "tms" not in contract.closure.fixed
```

- [ ] **Step 4: Run test to verify it fails, then implement Step 2, run again**

Run: `uv run pytest tests/templates/gtap6/test_gtap6_contract.py -v`
Expected: PASS (3 tests) after Step 2.

- [ ] **Step 5: Commit**

```bash
git add src/equilibria/templates/gtap6/gtap6_contract.py tests/templates/gtap6/test_gtap6_contract.py
git commit -m "feat(gtap6): port GTAP6Contract (standard/altertax/trade_policy closures)"
```

---

### Task 5: Bring in the monolith form-diff oracle

**Files:**
- Create: `scripts/gtap6/__init__.py`
- Create: `scripts/gtap6/_v62_monolith_oracle.py`

**Interfaces:**
- Produces: a callable `build_monolith_model(dataset_dir: Path, *, mode:
  str = "nlp") -> ConcreteModel` wrapping the ported
  `GTAPv62ModelEquations`/`GTAP6ModelEquationsOracle` class's
  `build_model()`, used ONLY by `tests/blocks/gtap6/test_gtap6_blocks_form.py`
  (Tasks 6-9) as the parity oracle. This module is never imported from
  `templates/gtap6/` or `blocks/gtap6/`.

- [ ] **Step 1: Fetch the monolith source file**

```bash
git show gtap/v62-multiperiod:src/equilibria/templates/gtap_v62/gtap_v62_model_equations.py > /tmp/gtap_v62_model_equations.py
git show gtap/v62-multiperiod:src/equilibria/templates/gtap_v62/gtap_v62_solver.py > /tmp/gtap_v62_solver.py
wc -l /tmp/gtap_v62_model_equations.py /tmp/gtap_v62_solver.py
```

- [ ] **Step 2: Create `scripts/gtap6/__init__.py`** (empty file, makes the
  directory a package for pytest imports)

```python
"""Oracle-only scripts supporting the F7 GTAP6 blocks migration.

Not part of the public equilibria API — used exclusively by the form-diff
gate in tests/blocks/gtap6/test_gtap6_blocks_form.py.
"""
```

- [ ] **Step 3: Write `scripts/gtap6/_v62_monolith_oracle.py`**

Copy `/tmp/gtap_v62_model_equations.py` content into this file, applying
the same import renames as Tasks 1-4 (`gtap_v62_sets` →
`equilibria.templates.gtap6.gtap6_sets`, etc. — point every import at the
Task 1-4 ported modules, NOT at new orphan-branch copies, so the oracle
and the real blocks share the same sets/parameters/calibration/contract
layer and only the equation-construction style differs). Rename the class
itself `GTAPv62ModelEquations` → `GTAP6MonolithOracle` throughout this
file (including its own internal self-references and the `build_model`
docstring). Do NOT fix the "Phase 2a placeholder" docstring at the top —
delete that stale paragraph since `_add_equations` is fully implemented in
this file (confirmed via `grep -c "def eq_\|Constraint(" /tmp/gtap_v62_model_equations.py`
showing dozens of hits, not zero).

Add a module-level helper at the bottom:

```python
def build_monolith_model(dataset_dir, *, mode: str = "nlp"):
    """Build the GTAP6 monolith oracle model for form-diff comparison.

    Args:
        dataset_dir: Path to a datasets/gtap6_* directory.
        mode: "nlp" or "mcp", forwarded to GTAP6MonolithOracle.

    Returns:
        A built Pyomo ConcreteModel (GTAP6MonolithOracle.build_model()).
    """
    from equilibria.templates.gtap6.gtap6_calibration import derive_calibration
    from equilibria.templates.gtap6.gtap6_parameters import GTAP6Parameters
    from equilibria.templates.gtap6.gtap6_sets import GTAP6Sets

    sets = GTAP6Sets()
    sets.load_from_har(dataset_dir / "sets.har", default_path=dataset_dir / "default.prm")
    params = GTAP6Parameters()
    params.load_from_har(dataset_dir / "basedata.har", dataset_dir / "default.prm", sets)
    derived = derive_calibration(sets, params)

    oracle = GTAP6MonolithOracle(sets, params, derived=derived, mode=mode)
    return oracle.build_model()
```

Adjust `GTAP6Parameters.load_from_har`'s actual argument order/count to
match what Task 2 finalized (this plan's Task 2 leaves that signature
open pending Step 1's inspection — keep this helper's call in sync).

- [ ] **Step 4: Smoke-test the oracle builds without error**

```bash
uv run python -c "
from pathlib import Path
from scripts.gtap6._v62_monolith_oracle import build_monolith_model
m = build_monolith_model(Path('datasets/gtap6_3x3'))
print('built', len(list(m.component_objects())), 'components')
"
```
Expected: prints a component count > 0, no traceback. If it raises,
iterate Step 3's import renames until this smoke test passes — do not
proceed to Task 6 with a broken oracle, since every subsequent form-diff
gate depends on it.

- [ ] **Step 5: Commit**

```bash
git add scripts/gtap6/__init__.py scripts/gtap6/_v62_monolith_oracle.py
git commit -m "test(gtap6): bring in v6.2 monolith as form-diff oracle (test-only, not public API)"
```

---

### Task 6: TradeArmington block (leaf unit)

**Files:**
- Create: `src/equilibria/blocks/gtap6/__init__.py` (partial — extended in Task 9b)
- Create: `src/equilibria/blocks/gtap6/trade_armington.py`
- Test: `tests/blocks/gtap6/test_gtap6_blocks_form.py` (new file, extended per block)

**Interfaces:**
- Consumes: `GTAP6Sets`, `GTAP6Parameters`, `DerivedGTAP6Calibration` (as
  `self.sets`/`self.params`/`self.derived` instance attributes, matching
  the `TradeCETBlock` pattern of `self.sets`/`self.params: Any = None`
  fields set post-construction).
- Produces: a `TradeArmingtonBlock(Block)` class registering variables
  `qfd, qfm, qfa, pfa, qxs, pms, pmcif, pe, pim, qds` (owned) and
  equations with `name` matching the contract's `_GTAB6_TRADE` +
  `_GTAP6_MARGINS` IDs: `e_qfd_arm, e_qfm_arm, e_qfa, e_pfa, e_qxs, e_pms,
  e_pmcif, e_pe, e_pim, e_qds, e_qst, e_pst, e_qtm, e_ptmg, e_pwmg,
  e_qtmfsd`. Returned as `list[SymbolicEquation]` from `setup(set_manager,
  parameters, variables)`.

Read `docs/findings/gtap_v62_phase315_diagonal_trade.md` and
`docs/findings/gtap_v62_phase316_diagonal_calibration.md` (fetch via `git
show gtap/v62-multiperiod:docs/findings/gtap_v62_phase315_diagonal_trade.md`
if not already checked out) before writing the equation bodies — these
document the diagonal-trade calibration fix that mattered for this exact
block family.

- [ ] **Step 1: Read the monolith's trade/margin equation bodies**

```bash
grep -n "def eq_qfd_arm\|def eq_qfm_arm\|def eq_qfa\|def eq_pfa\|def eq_qxs\|def eq_pms\|def eq_pmcif\|def eq_pe\b\|def eq_pim\|def eq_qds\|def eq_qst\|def eq_pst\|def eq_qtm\|def eq_ptmg\|def eq_pwmg\|def eq_qtmfsd" /tmp/gtap_v62_model_equations.py
```

Read each matched function body in `/tmp/gtap_v62_model_equations.py`
(use `sed -n '<start>,<start+30>p'` around each match) to transcribe the
exact algebraic form into the block's `SymbolicEquation.build_expression`
— this plan does not repeat all ~16 equation bodies inline because they
must be transcribed from the actual monolith source (available only after
Task 5 Step 1's fetch), not invented; the fidelity requirement is a
byte-faithful port, same as `TradeCETBlock` was ported from
`gtap_model_equations.py` for GTAP7 (see that file's docstring: "Ports the
monolith's ... equations VERBATIM").

- [ ] **Step 2: Write `trade_armington.py`**

Follow the exact structural pattern of
`src/equilibria/blocks/gtap/trade_cet.py` (Block subclass, `model_post_init`
sets `required_sets`, `setup()` builds `Variable`/`Parameter` objects then
nested `SymbolicEquation` subclasses per equation, closures capture
`self.params`/`self.sets` derived values before the nested class
definitions). Class skeleton:

```python
"""GTAP6 TRADE_ARMINGTON block (leaf unit).

Ports the v6.2 monolith's Armington + bilateral trade + margin equations
VERBATIM from scripts/gtap6/_v62_monolith_oracle.py (GTAP6MonolithOracle),
following the same fidelity discipline blocks/gtap/trade_cet.py used for
GTAP7's CET block.

v6.2 differences from GTAP7's ArmingtonBilateralBlock: no MRIO, no
region-indexed esubd/esubm (commodity-only), margins are Cobb-Douglas
(v6.2 has no ESUBS), no ifSUB macro substitution.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from equilibria.blocks.base import Block
from equilibria.core.parameters import Parameter
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable


class TradeArmingtonBlock(Block):
    """GTAP6 Armington + bilateral trade + Cobb-Douglas margins."""

    name: str = "GTAP6_TRADE_ARMINGTON"
    description: str = "GTAP6 Armington demand, bilateral trade, CD margins"
    sets: Any = None
    params: Any = None
    derived: Any = None

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["r", "i"]

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        regions = list(set_manager.get("r"))
        comms = list(set_manager.get("i"))
        # ... transcribe variable declarations + equation classes here,
        # one SymbolicEquation subclass per equation ID from Step 1.
        ...
```

Transcribe each of the ~16 equations from Step 1 as a nested
`SymbolicEquation` subclass, e.g.:

```python
        class EqQxs(SymbolicEquation):
            name: str = "e_qxs"
            domains: tuple = ("i", "s", "r")

            def build_expression(self, pyomo_model, indices):
                i, s, r = indices
                # <transcribed from the monolith's eq_qxs body, Step 1>
                ...
```

- [ ] **Step 3: Write the form-diff gate test**

```python
"""GTAP6 block units vs the v6.2 monolith oracle — form + domain gate."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATASET = ROOT / "datasets" / "gtap6_3x3"

_MIGRATED: list[str] = ["TradeArmingtonBlock"]


def _build_oracle():
    from gtap6._v62_monolith_oracle import build_monolith_model

    return build_monolith_model(DATASET)


def _build_block_model():
    from equilibria.blocks.gtap6.trade_armington import TradeArmingtonBlock
    from equilibria.templates.gtap6.gtap6_calibration import derive_calibration
    from equilibria.templates.gtap6.gtap6_parameters import GTAP6Parameters
    from equilibria.templates.gtap6.gtap6_sets import GTAP6Sets

    sets = GTAP6Sets()
    sets.load_from_har(DATASET / "sets.har", default_path=DATASET / "default.prm")
    params = GTAP6Parameters()
    params.load_from_har(DATASET / "basedata.har", DATASET / "default.prm", sets)
    derived = derive_calibration(sets, params)

    block = TradeArmingtonBlock(sets=sets, params=params, derived=derived)
    return block, sets, params, derived


def test_trade_armington_block_setup_returns_all_contract_equations():
    from equilibria.templates.gtap6.gtap6_contract import _GTAP6_TRADE, _GTAP6_MARGINS

    block, sets, params, derived = _build_block_model()
    from equilibria.core.sets import SetManager

    set_manager = SetManager()
    set_manager.add("r", sets.r)
    set_manager.add("i", sets.i)
    set_manager.add("s", sets.r)

    equations = block.setup(set_manager, {}, {})
    eq_names = {eq.name for eq in equations}

    expected = set(_GTAP6_TRADE) | set(_GTAP6_MARGINS)
    missing = expected - eq_names
    assert not missing, f"TradeArmingtonBlock did not produce: {missing}"
```

Adjust `SetManager`'s actual `add`/constructor API to match
`equilibria.core.sets.SetManager` (inspect `src/equilibria/core/sets.py`
before finalizing this test if the API shown here doesn't match — this is
scaffolding, confirm against the real class signature).

- [ ] **Step 4: Run the test, iterate block implementation to green**

Run: `uv run pytest tests/blocks/gtap6/test_gtap6_blocks_form.py -v`
Iterate Step 2 (equation bodies) until every equation ID in
`_GTAP6_TRADE | _GTAP6_MARGINS` is produced without a Python exception.

- [ ] **Step 5: Add a numeric form-diff assertion against the oracle**

Extend the test to actually build both models on the same seed and compare
one representative equation's residual body value cell-by-cell (mirror the
approach in `tests/templates/gtap/test_gtap_blocks_form.py`'s
`_exprs_equal` helper — import
`scripts.gtap.blocks_diag._exprs_equal` if it is generic enough to reuse
across gtap/gtap6, or write a local `_residual_diff` helper that evaluates
both Pyomo expressions at the benchmark point and asserts `abs(a - b) <
1e-9` per cell). This is the load-bearing check — Step 3 only proves the
block returns SOMETHING, not that it's correct.

- [ ] **Step 6: Commit**

```bash
git add src/equilibria/blocks/gtap6/__init__.py src/equilibria/blocks/gtap6/trade_armington.py tests/blocks/gtap6/test_gtap6_blocks_form.py
git commit -m "feat(gtap6): TradeArmingtonBlock — Armington + bilateral trade + CD margins"
```

---

### Task 7: Production block

**Files:**
- Create: `src/equilibria/blocks/gtap6/production.py`
- Modify: `tests/blocks/gtap6/test_gtap6_blocks_form.py` (extend `_MIGRATED`)

**Interfaces:**
- Consumes: `pd, ps, pfa` (stubs from TradeArmingtonBlock, dedup by name).
- Produces: `ProductionBlock(Block)` owning `qo, ps, qf, pf, qva, pva, qfe,
  pfe` and equations `e_qo, e_ps, e_qf, e_pf, e_qva, e_pva, e_qfe, e_pfe`
  (from `_GTAP6_PRODUCTION`).

- [ ] **Step 1: Read the monolith's production equation bodies**

```bash
grep -n "def eq_qo\|def eq_ps\b\|def eq_qf\b\|def eq_pf\b\|def eq_qva\|def eq_pva\|def eq_qfe\|def eq_pfe" /tmp/gtap_v62_model_equations.py
```

Transcribe each body (Leontief top nest per the spec's note "No
intermediate bundle (Leontief implicit)" — v6.2's `e_qf`/`e_pf` are
Leontief fixed-coefficient, not CES, so expect simpler algebra than
GTAP7's `ProductionSupplyBlock`).

- [ ] **Step 2: Write `production.py`**

Same structural pattern as Task 6 Step 2 (Block subclass +
`model_post_init` + `setup()` + nested `SymbolicEquation` classes),
transcribing the Step 1 bodies. Class name `ProductionBlock`, `name:
str = "GTAP6_PRODUCTION"`.

- [ ] **Step 3: Extend the form-diff test's `_MIGRATED` list and equation-set assertion**

Add `"ProductionBlock"` to `_MIGRATED` in
`tests/blocks/gtap6/test_gtap6_blocks_form.py`, and a second
`test_production_block_setup_returns_all_contract_equations` mirroring
Task 6 Step 3's test but importing `_GTAP6_PRODUCTION` and
`ProductionBlock`.

- [ ] **Step 4: Run and iterate to green**

Run: `uv run pytest tests/blocks/gtap6/test_gtap6_blocks_form.py -v`

- [ ] **Step 5: Add numeric form-diff (same approach as Task 6 Step 5)**

- [ ] **Step 6: Commit**

```bash
git add src/equilibria/blocks/gtap6/production.py tests/blocks/gtap6/test_gtap6_blocks_form.py
git commit -m "feat(gtap6): ProductionBlock — Leontief top nest + CES value-added"
```

---

### Task 8: Factor block

**Files:**
- Create: `src/equilibria/blocks/gtap6/factor.py`
- Modify: `tests/blocks/gtap6/test_gtap6_blocks_form.py` (extend `_MIGRATED`)

**Interfaces:**
- Consumes: `pd, ps, pfe` stubs (dedup).
- Produces: `FactorBlock(Block)` owning `qoes, pmes, qe` and equations
  `e_qoes, e_pmes, e_pm_endw, e_qe, e_pe_endw` (from
  `_GTAP6_FACTOR_MARKETS`). Mobile/sluggish split driven by `sets.mf`/
  `sets.sf` (Task 1's `SLUG`-derived partition), NOT a per-activity
  `tinc(e,a,r)` tax as in GTAP7 — v6.2 has no activity-level factor income
  tax, per the spec.

- [ ] **Step 1: Read the monolith's factor-market equation bodies**

```bash
grep -n "def eq_qoes\|def eq_pmes\|def eq_pm_endw\|def eq_qe\b\|def eq_pe_endw" /tmp/gtap_v62_model_equations.py
```

Read `docs/findings/gtap_v62_phase318_closure_audit.md` (fetch via `git
show gtap/v62-multiperiod:docs/findings/gtap_v62_phase318_closure_audit.md`)
before transcribing — it documents a closure-consistency issue at the
factor-market/income boundary in the original monolith.

- [ ] **Step 2: Write `factor.py`**

Same pattern as Tasks 6-7. Class `FactorBlock`, `name: str =
"GTAP6_FACTOR"`. The CET-based sluggish-factor allocation
(`e_qoes`/`e_pmes`) applies only to `sets.sf`; mobile factors (`sets.mf`)
use the simpler `e_qe`/`e_pe_endw` law-of-one-price form — mirror
whichever conditional structure the monolith body uses (Step 1) rather
than inventing a different branch condition.

- [ ] **Step 3: Extend the form-diff test**

Add `"FactorBlock"` to `_MIGRATED` and a
`test_factor_block_setup_returns_all_contract_equations` test importing
`_GTAP6_FACTOR_MARKETS`.

- [ ] **Step 4: Run and iterate to green**

Run: `uv run pytest tests/blocks/gtap6/test_gtap6_blocks_form.py -v`

- [ ] **Step 5: Add numeric form-diff**

- [ ] **Step 6: Commit**

```bash
git add src/equilibria/blocks/gtap6/factor.py tests/blocks/gtap6/test_gtap6_blocks_form.py
git commit -m "feat(gtap6): FactorBlock — commodity-level mobile/sluggish factor markets"
```

---

### Task 9a: DemandUtility block

**Files:**
- Create: `src/equilibria/blocks/gtap6/demand_utility.py`
- Modify: `tests/blocks/gtap6/test_gtap6_blocks_form.py` (extend `_MIGRATED`)

**Interfaces:**
- Consumes: `pfa` stub (dedup).
- Produces: `DemandUtilityBlock(Block)` owning `qpd, qpm, qp, pp, pq, up,
  qgd, qgm, qg, pg, pgov, ug, qcgds, pcgds` and equations `e_qpd, e_qpm,
  e_qp, e_pp, e_pq, e_up, e_qgd, e_qgm, e_qg, e_pg, e_pgov, e_ug, e_qcgds,
  e_pcgds, e_qfd_cgds, e_qfm_cgds` (from `_GTAP6_FINAL_DEMAND`, excluding
  `e_yp`/`e_yg` which belong to Task 9b's income block per the contract's
  own placement — confirm this split against `_GTAP6_FINAL_DEMAND`'s
  actual tuple contents from Task 4 before finalizing which IDs land in
  9a vs 9b, since the household/gov utility equations and their income
  identities may be more entangled than this split assumes).

Read `docs/findings/gtap_v62_phase319_cde_preferences.md`,
`docs/findings/gtap_v62_phase320_levels_cde.md`, and
`docs/findings/gtap_v62_phase321_cde_income_split.md` (fetch via `git
show gtap/v62-multiperiod:docs/findings/<name>.md`) before transcribing —
these are the 3 phases that got CDE preferences right (true
levels-CDE Hanoch-Hertel expenditure function, not the earlier
Cobb-Douglas approximation) and document exactly which bugs to avoid.

- [ ] **Step 1: Read the monolith's demand equation bodies**

```bash
grep -n "def eq_qpd\|def eq_qpm\|def eq_qp\b\|def eq_pp\b\|def eq_pq\b\|def eq_up\b\|def eq_qgd\|def eq_qgm\|def eq_qg\b\|def eq_pg\b\|def eq_pgov\|def eq_ug\b\|def eq_qcgds\|def eq_pcgds\|def eq_qfd_cgds\|def eq_qfm_cgds" /tmp/gtap_v62_model_equations.py
```

- [ ] **Step 2: Write `demand_utility.py`**

Same pattern as prior tasks. Class `DemandUtilityBlock`, `name: str =
"GTAP6_DEMAND_UTILITY"`. The CDE utility equations (`e_up`) must use the
true Hanoch-Hertel levels form documented in Phase 3.20's finding, not a
Cobb-Douglas shortcut.

- [ ] **Step 3: Extend the form-diff test**

Add `"DemandUtilityBlock"` to `_MIGRATED`.

- [ ] **Step 4: Run and iterate to green**

Run: `uv run pytest tests/blocks/gtap6/test_gtap6_blocks_form.py -v`

- [ ] **Step 5: Add numeric form-diff**

- [ ] **Step 6: Commit**

```bash
git add src/equilibria/blocks/gtap6/demand_utility.py tests/blocks/gtap6/test_gtap6_blocks_form.py
git commit -m "feat(gtap6): DemandUtilityBlock — CDE household, CD gov, cgds sector"
```

---

### Task 9b: IncomeClosure block (last — closure)

**Files:**
- Create: `src/equilibria/blocks/gtap6/income_closure.py`
- Create: `src/equilibria/blocks/gtap6/__init__.py` finalized with `GTAP6_BLOCK_ORDER`
- Modify: `tests/blocks/gtap6/test_gtap6_blocks_form.py` (extend `_MIGRATED`, final block)

**Interfaces:**
- Consumes: `yc, yg` or equivalent income stubs surfaced by Task 9a
  (dedup), `qo, pfe` stubs from Tasks 7-8.
- Produces: `IncomeClosureBlock(Block)` owning `sav` (as a Pyomo `Var`,
  per the Global Constraints — this is the Phase 3.38 fix, not a Param),
  `y, ysav, psave, rorg, kb, ke, walras, pgdpwld, taxrev, gdpmp, rgdpmp,
  pgdpmp` and equations `e_y, e_ysav, e_psave, e_rorg, e_kb, e_ke,
  e_walras, e_pgdpwld, e_taxrev, e_gdpmp, e_rgdpmp, e_pgdpmp` (from
  `_GTAP6_INCOME_AND_CLOSURE`), plus `e_yp, e_yg` if Task 9a's split left
  them here (resolve during Task 9a per its own note).

Read `docs/findings/gtap_v62_phase338_sav_var_budget_identity.md` in full
before writing `e_walras`/`sav` — this is the single most important
finding for this block (already summarized in the spec's "Bugs conocidos"
section, but read the full doc for the exact diagnostic pattern in case
another region-level leak needs the same treatment).

- [ ] **Step 1: Read the monolith's income/closure equation bodies**

```bash
grep -n "def eq_y\b\|def eq_ysav\|def eq_psave\|def eq_rorg\|def eq_kb\b\|def eq_ke\b\|def eq_walras\|def eq_pgdpwld\|def eq_taxrev\|def eq_gdpmp\|def eq_rgdpmp\|def eq_pgdpmp" /tmp/gtap_v62_model_equations.py
```

Confirm the monolith's `eq_walras` body already reflects the Phase 3.38
fix (i.e. references a `sav` `Var`, not a `save_0` `Param`) — if
`/tmp/gtap_v62_model_equations.py` predates that fix (check with `git log
--oneline gtap/v62-multiperiod -- src/equilibria/templates/gtap_v62/gtap_v62_model_equations.py`
and confirm the file checked out in Task 5 Step 1 is at or after commit
`83cdf8b`), it is already correct; if for any reason an earlier commit was
fetched, re-fetch at `gtap/v62-multiperiod` HEAD (the branch tip) to get
the fixed version before transcribing.

- [ ] **Step 2: Write `income_closure.py`**

Same pattern as prior tasks. Class `IncomeClosureBlock`, `name: str =
"GTAP6_INCOME_CLOSURE"`. Declare `sav` explicitly as:

```python
        variables["sav"] = Variable(
            name="sav",
            value=sav_init,  # from derived.y0 - derived.yp0 - derived.yg0 per region, or the monolith's init
            domains=("r",),
            domain="Reals",
            lower=float("-inf"),
            upper=float("inf"),
        )
```

with `e_walras` and `e_ysav` referencing `pyomo_model.sav[r]` as a live
variable, matching the transcribed monolith body from Step 1 (which, per
Step 1's verification, already uses `sav` as a Var post-Phase-3.38).

- [ ] **Step 3: Write `blocks/gtap6/__init__.py`**

```python
"""GTAP6 symbolic Block units.

5 units (leaf -> closure), fewer than GTAP7's 7 because v6.2 has no
make-matrix / MRIO / output-CET split: TradeArmington, Production, Factor,
DemandUtility, IncomeClosure.
"""

from equilibria.blocks.gtap6.demand_utility import DemandUtilityBlock
from equilibria.blocks.gtap6.factor import FactorBlock
from equilibria.blocks.gtap6.income_closure import IncomeClosureBlock
from equilibria.blocks.gtap6.production import ProductionBlock
from equilibria.blocks.gtap6.trade_armington import TradeArmingtonBlock

GTAP6_BLOCK_ORDER = [
    TradeArmingtonBlock,
    ProductionBlock,
    FactorBlock,
    DemandUtilityBlock,
    IncomeClosureBlock,
]

__all__ = [
    "DemandUtilityBlock",
    "FactorBlock",
    "GTAP6_BLOCK_ORDER",
    "IncomeClosureBlock",
    "ProductionBlock",
    "TradeArmingtonBlock",
]
```

- [ ] **Step 4: Extend the form-diff test to cover all 5 blocks, assert full equation-ID coverage**

Add a final aggregate test:

```python
def test_all_5_blocks_together_cover_every_contract_equation():
    from equilibria.templates.gtap6.gtap6_contract import _full_gtap6_equation_ids
    from equilibria.core.sets import SetManager
    from equilibria.blocks.gtap6 import GTAP6_BLOCK_ORDER

    _, sets, params, derived = _build_block_model()  # from TradeArmington's helper
    set_manager = SetManager()
    set_manager.add("r", sets.r)
    set_manager.add("i", sets.i)
    set_manager.add("s", sets.r)

    all_names: set[str] = set()
    for cls in GTAP6_BLOCK_ORDER:
        block = cls(sets=sets, params=params, derived=derived)
        eqs = block.setup(set_manager, {}, {})
        all_names |= {eq.name for eq in eqs}

    expected = set(_full_gtap6_equation_ids())
    missing = expected - all_names
    assert not missing, f"No block produces: {missing}"
```

- [ ] **Step 5: Run and iterate to green**

Run: `uv run pytest tests/blocks/gtap6/test_gtap6_blocks_form.py -v`
Expected: all 5 per-block tests + the aggregate coverage test PASS.

- [ ] **Step 6: Add numeric form-diff for this block**

- [ ] **Step 7: Commit**

```bash
git add src/equilibria/blocks/gtap6/income_closure.py src/equilibria/blocks/gtap6/__init__.py tests/blocks/gtap6/test_gtap6_blocks_form.py
git commit -m "feat(gtap6): IncomeClosureBlock — sav as Var (Phase 3.38 fix), 5/5 blocks complete"
```

---

### Task 10: Composer + canary solve (gtap6_3x3)

**Files:**
- Create: `src/equilibria/templates/gtap6/gtap6_block_model.py`
- Create: `tests/templates/gtap6/test_gtap6_blocks_solve.py`

**Interfaces:**
- Consumes: `GTAP6_BLOCK_ORDER` (Task 9b), `GTAP6Sets`, `GTAP6Parameters`,
  `DerivedGTAP6Calibration`, `GTAP6ClosureConfig`.
- Produces: `build_block_single_period(sets: GTAP6Sets, params:
  GTAP6Parameters, derived, closure: GTAP6ClosureConfig, *, mode: str =
  "nlp") -> ConcreteModel`, mirroring
  `templates/gtap/gtap_block_model.py`'s `build_block_single_period`
  signature and composition steps (compose blocks → `PyomoBackend.build`
  → strip `_con` suffix).

- [ ] **Step 1: Read the GTAP7 composer in full**

```bash
cat src/equilibria/templates/gtap/gtap_block_model.py
```

(Already partially read during design research — read the FULL file now,
including the parts after line 130, to transcribe the
`_strip_con_suffix`/`PyomoBackend.build`/scaling-application sequence
faithfully.)

- [ ] **Step 2: Write `gtap6_block_model.py`**

Adapt the GTAP7 composer's structure — `_block_classes()`,
`_set_elems(sets)`, `_mk_unit(cls, sets, params, ...)`,
`_strip_con_suffix(pm)`, and the top-level `build_block_single_period`
— to GTAP6's simpler set structure (no `aa`/`fd`/`gy` aggregate-agent
sets since v6.2 has no separate Armington-agent split — `cgds` is in
`sets.prod_comm`, not a final-demand agent list) and to
`GTAP6_BLOCK_ORDER`'s 5 units instead of GTAP7's 7. Drop any
scaling/snapshot step from the GTAP7 composer's checklist
(`blocks/gtap/__init__.py`'s docstring items 1-7) that references
v7-only constructs (`ifSUB` macros, make-matrix `xscale`, `pmuv`
Tornqvist switch) — v6.2 has none of these per the contract
(`if_sub: bool = False` is a fixed constant, not a runtime switch).

```python
"""Compose the 5 GTAP6 symbolic blocks into a solvable model.

Mirrors templates/gtap/gtap_block_model.py's composer pattern but for
GTAP6's simpler set structure (no aa/gy aggregate sets, no ifSUB switch,
no make-matrix scaling).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from equilibria.backends.pyomo_backend import PyomoBackend
from equilibria.core.sets import Set as ESet
from equilibria.model import Model

if TYPE_CHECKING:
    from pyomo.environ import ConcreteModel


def _set_elems(sets: Any) -> dict[str, list[str]]:
    return {
        "r": list(sets.r),
        "s": list(sets.r),  # bilateral alias
        "i": list(sets.i),
        "f": list(sets.f),
        "mf": list(sets.mf),
        "sf": list(sets.sf),
        "m": list(sets.m),
        "marg": list(sets.marg),
        "cgds": list(sets.cgds),
    }


def _strip_con_suffix(pm: "ConcreteModel") -> None:
    from pyomo.environ import Constraint

    for c in list(pm.component_objects(Constraint, active=True)):
        nm = c.name
        if nm.endswith("_con"):
            base = nm[:-4]
            pm.del_component(c)
            pm.add_component(base, c)


def build_block_single_period(sets, params, derived, closure, *, mode: str = "nlp"):
    from equilibria.blocks.gtap6 import GTAP6_BLOCK_ORDER
    from equilibria.core.sets import SetManager

    set_manager = SetManager()
    for name, elems in _set_elems(sets).items():
        set_manager.add(name, elems)

    model = Model(name="gtap6_block_model")
    parameters: dict[str, Any] = {}
    variables: dict[str, Any] = {}
    all_equations = []

    for cls in GTAP6_BLOCK_ORDER:
        block = cls(sets=sets, params=params, derived=derived)
        equations = block.setup(set_manager, parameters, variables)
        all_equations.extend(equations)

    backend = PyomoBackend()
    pm = backend.build(set_manager, parameters, variables, all_equations)
    _strip_con_suffix(pm)
    return pm
```

Adjust `Model`/`PyomoBackend.build`/`SetManager.add` call signatures to
match the actual APIs in `src/equilibria/model.py`,
`src/equilibria/backends/pyomo_backend.py`, and `src/equilibria/core/sets.py`
— this skeleton mirrors the GTAP7 composer's intent but Step 1's full read
of the real file is what fixes the exact argument order/names before this
is finalized.

- [ ] **Step 3: Write the canary solve test**

```python
"""GTAP6 block-composed model solves gtap6_3x3 (canary — F7 Task 10 gate)."""
from __future__ import annotations

from pathlib import Path

import pytest

DATASET = Path(__file__).resolve().parents[3] / "datasets" / "gtap6_3x3"


def _build():
    from equilibria.templates.gtap6.gtap6_block_model import build_block_single_period
    from equilibria.templates.gtap6.gtap6_calibration import derive_calibration
    from equilibria.templates.gtap6.gtap6_contract import default_gtap6_contract
    from equilibria.templates.gtap6.gtap6_parameters import GTAP6Parameters
    from equilibria.templates.gtap6.gtap6_sets import GTAP6Sets

    sets = GTAP6Sets()
    sets.load_from_har(DATASET / "sets.har", default_path=DATASET / "default.prm")
    params = GTAP6Parameters()
    params.load_from_har(DATASET / "basedata.har", DATASET / "default.prm", sets)
    derived = derive_calibration(sets, params)
    closure = default_gtap6_contract().closure

    return build_block_single_period(sets, params, derived, closure, mode="nlp")


@pytest.mark.local
def test_gtap6_3x3_block_model_solves_nlp():
    from pyomo.environ import SolverFactory, TerminationCondition, value

    model = _build()
    solver = SolverFactory("ipopt")
    result = solver.solve(model, tee=False)

    ok_status = result.solver.termination_condition in (
        TerminationCondition.optimal,
        TerminationCondition.locallyOptimal,
    )
    assert ok_status, result.solver.termination_condition
    assert abs(value(model.walras)) < 1e-6
```

Mark with `@pytest.mark.local` (matching the repo's convention for tests
needing a local IPOPT/PATH install, per the coverage-matrix docs'
"local gates need PATH/IPOPT" note) if that marker exists in
`pyproject.toml`/`conftest.py` — check `grep -n "markers" pyproject.toml`
first and reuse the existing marker name if different.

- [ ] **Step 4: Run test, debug via the diagnostic tooling pattern**

Run: `uv run pytest tests/templates/gtap6/test_gtap6_blocks_solve.py -v -m local`

If it fails to solve, do NOT guess — build a per-equation residual report
at the seeded point first (same discipline F3 mandated before its first
solve): evaluate every constraint body at initial variable values and
print the largest residuals, narrowing to which block's equations are
violated, before changing any equation body. This mirrors
`scripts/gtap/blocks_diag.py`'s role for GTAP7 — write an analogous
`scripts/gtap6/blocks_diag.py` only if the first solve attempt fails and a
residual report is needed; do not write it speculatively.

- [ ] **Step 5: Commit once green**

```bash
git add src/equilibria/templates/gtap6/gtap6_block_model.py tests/templates/gtap6/test_gtap6_blocks_solve.py
git commit -m "feat(gtap6): composer + canary solve — gtap6_3x3 solves via 5 blocks (code=1, walras<1e-6)"
```

---

### Task 11: Port the GTAP6 solver module

**Files:**
- Create: `src/equilibria/templates/gtap6/gtap6_solver.py`

**Interfaces:**
- Produces: whatever solve-orchestration API `gtap_v62_solver.py` exposes
  (91 lines — small; likely a thin wrapper choosing IPOPT for `mode="nlp"`
  / PATH for `mode="mcp"` around `build_block_single_period`). Confirm the
  exact function name via Step 1 before writing downstream consumers.

- [ ] **Step 1: Fetch and read the source file**

```bash
git show gtap/v62-multiperiod:src/equilibria/templates/gtap_v62/gtap_v62_solver.py > /tmp/gtap_v62_solver.py
cat /tmp/gtap_v62_solver.py
```

- [ ] **Step 2: Write `gtap6_solver.py`**

Copy verbatim, rename `gtap_v62_*` imports/types to `gtap6_*` per the
pattern established in Tasks 1-4, and repoint any reference to
`GTAPv62ModelEquations.build_model()` to instead call Task 10's
`build_block_single_period` (this is the one substantive change — the
orphan branch's solver drove the monolith directly; ours drives the
composed block model).

- [ ] **Step 3: Write a smoke test reusing Task 10's canary**

```python
"""GTAP6 solver module smoke test."""
from __future__ import annotations

from pathlib import Path

import pytest

DATASET = Path(__file__).resolve().parents[3] / "datasets" / "gtap6_3x3"


@pytest.mark.local
def test_solve_gtap6_returns_converged_result():
    from equilibria.templates.gtap6.gtap6_calibration import derive_calibration
    from equilibria.templates.gtap6.gtap6_contract import default_gtap6_contract
    from equilibria.templates.gtap6.gtap6_parameters import GTAP6Parameters
    from equilibria.templates.gtap6.gtap6_sets import GTAP6Sets
    from equilibria.templates.gtap6.gtap6_solver import solve_gtap6  # confirm actual name in Step 1

    sets = GTAP6Sets()
    sets.load_from_har(DATASET / "sets.har", default_path=DATASET / "default.prm")
    params = GTAP6Parameters()
    params.load_from_har(DATASET / "basedata.har", DATASET / "default.prm", sets)
    derived = derive_calibration(sets, params)
    closure = default_gtap6_contract().closure

    result = solve_gtap6(sets, params, derived, closure, mode="nlp")
    assert result.status == "optimal" or getattr(result, "code", None) == 1
```

Adjust `solve_gtap6`'s actual name/signature and the result object's
status attribute once Step 1 confirms them.

- [ ] **Step 4: Run and iterate to green**

Run: `uv run pytest tests/templates/gtap6/test_gtap6_solver.py -v -m local`

- [ ] **Step 5: Commit**

```bash
git add src/equilibria/templates/gtap6/gtap6_solver.py tests/templates/gtap6/test_gtap6_solver.py
git commit -m "feat(gtap6): port solver orchestration, wired to the block composer"
```

---

### Task 12: Full NLP+MCP sweep vs GEMPACK, per dataset (final gate)

**Files:**
- Create: `tests/templates/gtap6/test_gtap6_gempack_parity.py`
- Modify: `scripts/gtap/coverage_matrix.py` (fill in the `gtap6` rows the
  investigation found already reserved as scaffolding — `MODELS =
  {"gtap6","gtap7"}` with 0 rows for gtap6)
- Modify: `docs/site/guide/gtap7_coverage_matrix.md` or create a sibling
  `docs/site/guide/gtap6_coverage_matrix.md` if the generator
  (`scripts/gtap/gen_coverage_doc.py`) treats each model as a separate
  page — check that script's output-path logic first.

**Interfaces:**
- Consumes: `build_block_single_period` (Task 10), GEMPACK reference GDX
  files for each `gtap6_*` dataset (locate via the same convention as
  `test_gtap7_gempack_parity.py` — check that test file for its reference
  GDX path pattern, e.g. `runs/*/gempack_refs/` or similar, before writing
  paths here).

- [ ] **Step 1: Read the GTAP7 GEMPACK parity test in full**

```bash
cat tests/templates/gtap/test_gtap7_gempack_parity.py
```

- [ ] **Step 2: Locate or generate GEMPACK references for gtap6_3x3**

```bash
find . -iname "*gempack*" -path "*gtap6*" 2>/dev/null
git show gtap/v62-multiperiod --stat -- 'runs/gtap_v62*' | head -30
```

If the orphan branch already produced GEMPACK reference outputs for
`gtap6_3x3` (Phase 3.38's own comparison presumably used some), fetch them
the same way as the source files (`git show
gtap/v62-multiperiod:<path> > <local_path>`). If none exist, this step
blocks on generating a fresh GEMPACK/RunGTAP run — flag this explicitly
rather than fabricating a reference; producing a new GEMPACK reference
GDX is out of scope for this plan's mechanical steps (it requires running
RunGTAP/GEMPACK interactively) and should be done by hand following
whatever process produced the existing `gtap7_*` references
(`docs/site/guide/gtap7_coverage_matrix_gempack.md` documents that
process — read it before running GEMPACK).

- [ ] **Step 3: Write `test_gtap6_gempack_parity.py` for gtap6_3x3**

Mirror `test_gtap7_gempack_parity.py`'s structure (per Step 1's read):
build the block model via Task 10's composer, solve NLP (IPOPT) and MCP
(PATH), load the GEMPACK reference from Step 2, compare cell-by-cell at
1% tolerance, assert `match_pct >= 99.0` (i.e. gap ≤ 1%) AND
`termination_condition` indicates convergence for both solves.

```python
"""GTAP6 vs GEMPACK parity gate — F7 Task 12. Local; needs IPOPT + PATH + refs."""
from __future__ import annotations

from pathlib import Path

import pytest

DATASETS = ["gtap6_3x3", "gtap6_5x5", "gtap6_10x7", "gtap6_15x10"]


@pytest.mark.local
@pytest.mark.parametrize("dataset_name", DATASETS)
def test_gtap6_matches_gempack_within_1pct(dataset_name):
    dataset = Path(__file__).resolve().parents[3] / "datasets" / dataset_name
    # ... build + solve (NLP and MCP) + load GEMPACK ref + compare, per Step 1's pattern
    ...
```

The `...` bodies are filled in directly from `test_gtap7_gempack_parity.py`'s
already-working comparison logic (import and reuse its cell-comparison
helper if it is dataset-agnostic; otherwise adapt it) — do not
reimplement cell-matching logic from scratch since GTAP7's version is
already correct and tested.

- [ ] **Step 4: Run for `gtap6_3x3` only first (gate before advancing)**

Run: `uv run pytest tests/templates/gtap6/test_gtap6_gempack_parity.py -v -m local -k gtap6_3x3`
Expected: PASS with match ≥ 99%. If it fails, debug via the 4-layer gate
in order (re-check Task 6-9's form-diff tests still pass, re-check Task
10's canary, then this sweep) — do not skip layers.

- [ ] **Step 5: Once 3x3 green, run 5x5, then 10x7, then 15x10 — one at a time**

Run: `uv run pytest tests/templates/gtap6/test_gtap6_gempack_parity.py -v -m local -k gtap6_5x5`
Run: `uv run pytest tests/templates/gtap6/test_gtap6_gempack_parity.py -v -m local -k gtap6_10x7`
Run: `uv run pytest tests/templates/gtap6/test_gtap6_gempack_parity.py -v -m local -k gtap6_15x10`

Each must independently pass ≥99% before moving to the next — if any
regresses relative to the prototype's measured 0.06–0.64% gap, that is a
fidelity regression in this reimplementation, not a new floor to accept.

- [ ] **Step 6: Delete the monolith oracle now that all 4 datasets are green**

```bash
git rm -r scripts/gtap6/_v62_monolith_oracle.py
```

Also remove the numeric form-diff assertions in
`tests/blocks/gtap6/test_gtap6_blocks_form.py` that depend on
`_build_oracle()` / `build_monolith_model` (keep the equation-ID coverage
tests — those don't need the oracle).

- [ ] **Step 7: Update the coverage matrix**

Edit `scripts/gtap/coverage_matrix.py` to populate the `gtap6` rows
(3x3/5x5/10x7/15x10 × NLP/MCP) with the floors measured in Step 4-5.
Regenerate the doc:

```bash
uv run python scripts/gtap/gen_coverage_doc.py
```

- [ ] **Step 8: Final commit**

```bash
git add tests/templates/gtap6/test_gtap6_gempack_parity.py tests/blocks/gtap6/test_gtap6_blocks_form.py scripts/gtap/coverage_matrix.py docs/site/guide/
git rm -r scripts/gtap6/_v62_monolith_oracle.py 2>/dev/null
git commit -m "feat(gtap6): F7 done — gtap6_3x3..15x10 solve via blocks, >=99% vs GEMPACK; 20x41 documented out of scope"
```

---

## Self-Review Notes

**Spec coverage:**
- ✅ Ported sets/parameters/calibration/contract (Tasks 1-4) per spec Decision 2.
- ✅ 5 new blocks in `blocks/gtap6/`, no shared instances with `blocks/gtap/` (Tasks 6-9b).
- ✅ Monolith oracle brought in test-only, deleted at the end (Tasks 5, 12 Step 6).
- ✅ 4-layer gate per dataset (Tasks 6-9 form/domain, 10 canary, 12 sweep).
- ✅ Known bugs from findings ported as design constraints (Global Constraints
  section + Task 9b's explicit `sav`-as-Var step).
- ✅ 20x41 excluded — no task targets it; Task 12 documents it as out of scope.
- ✅ Coverage matrix / roadmap update (Task 12 Step 7).

**Known open items requiring judgment during execution** (flagged inline in
their tasks, not hidden): Task 2's exact benchmark-values class name, Task
9a/9b's exact equation-ID split for `e_yp`/`e_yg`, Task 10's exact
`Model`/`PyomoBackend`/`SetManager` call signatures, Task 12's GEMPACK
reference availability. These are marked as "confirm during Step 1" rather
than guessed, per the domain — the orphan branch's exact internals were
sampled but not exhaustively transcribed during design; each task's Step 1
closes that gap with a real read before code is written.
