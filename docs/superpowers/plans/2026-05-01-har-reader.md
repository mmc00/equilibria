# equilibria.babel.har — HAR Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `equilibria.babel.har` — a thin harpy3 wrapper that reads GEMPACK `.har` files and feeds them into `GTAPParameters`, enabling NUS333 (3-sector × 2-region) datasets as a fast iteration target for GTAP parity work.

**Architecture:** `equilibria/babel/har/` mirrors the existing `equilibria/babel/gdx/` structure: `reader.py` wraps `harpy.HarFileObj`, `symbols.py` holds a `HeaderArray` dataclass, and `__init__.py` exposes the public API. A separate `load_from_har()` method on `GTAPParameters` (and its sub-objects) maps GEMPACK header names and dimension orderings to the Python internal dicts using the existing `GTAP_STD7_INDEX_REORDER` table.

**Tech Stack:** `harpy3==0.3.1` (GEMPACK HAR reader, installed at `/opt/homebrew/lib/python3.14/site-packages/harpy`), numpy, Pydantic, existing `equilibria.babel.gdx` patterns, `GTAP_STD7_INDEX_REORDER` from `gtap_std7_mapping.py`.

---

## File Structure

**Create:**
- `src/equilibria/babel/har/__init__.py` — public API: `read_har`, `HeaderArray`
- `src/equilibria/babel/har/reader.py` — `read_har()`, `read_header_array()`, `get_header_names()`
- `src/equilibria/babel/har/symbols.py` — `HeaderArray` dataclass
- `tests/babel/har/__init__.py` — empty
- `tests/babel/har/test_reader.py` — reader unit tests against NUS333 basedata.har

**Modify:**
- `pyproject.toml` — add `harpy3>=0.3` to `[project.optional-dependencies] har`
- `src/equilibria/templates/gtap/gtap_parameters.py` — add `load_from_har()` to `GTAPSets`, `GTAPBenchmark`, `GTAPElasticities`, `GTAPTaxes`, and top-level `GTAPParameters`

---

## NUS333 Dataset Reference

Located at `/Users/marmol/Downloads/10284/`:
- `sets.har` — REG (USA, ROW), COMM (AGR, MFG, SER), ACTS (AGR, MFG, SER), ENDW (LAND, LABOR, CAPITAL), MARG (SER)
- `basedata.har` — 37 benchmark headers (VDPP, VMPP, VMSB, VCIF, VTWR, SAVE, VKB, ...)
- `default.prm` — elasticities (ESBD→esubd, ESBM→esubm, ESBV→esubva, ...)
- `baserate.har` — tax rates (RTMS→imptx, RTPD→rtpd, RTXS→rtxs, ...)

## Dimension Reordering

GEMPACK stores arrays with dimensions `(COMM, REG)` = `(i, r)`. Python internal dicts use key order `(r, i)`. The existing `GTAP_STD7_INDEX_REORDER` table in `gtap_std7_mapping.py` already encodes the right permutations:

| Header | HAR dims | Python key | Reorder |
|--------|----------|------------|---------|
| VDPP, VMPP, etc. | (COMM, REG) | (r, i) | (1, 0) |
| VCIF, VMSB, VXSB, VFOB | (COMM, REG, REG) | (r, i, rp) | (1, 0, 2) |
| VTWR | (MARG, COMM, REG, REG) | (r, i, rp, m) | (2, 1, 3, 0) |
| SAVE, VKB, VDEP, POP | (REG,) | (r,) | none |
| RTMS, RTXS | (COMM, REG, REG) | (i, r, rp) | (1, 0, 2) |
| ESBD, ESBM, INCP, SUBP | (COMM, REG) | (r, i) | (1, 0) |
| ESBV, ESBT, ETRQ | (ACTS, REG) | (r, a) | (1, 0) |
| ETRE | (ENDW, REG) | (r, f) | (1, 0) |

---

## Task 1: Add harpy3 dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add optional dependency group**

Open `pyproject.toml`. In `[project.optional-dependencies]`, add:
```toml
# GEMPACK HAR format support
har = ["harpy3>=0.3"]
```

- [ ] **Step 2: Install in venv**

```bash
.venv/bin/pip install harpy3
```

Expected: `Successfully installed harpy3-0.3.1`

- [ ] **Step 3: Verify**

```bash
.venv/bin/python -c "import harpy; print('harpy OK', harpy.__file__)"
```

Expected: `harpy OK .venv/lib/python.../site-packages/harpy/__init__.py`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat(babel): add harpy3 optional dependency for HAR support"
```

---

## Task 2: `symbols.py` — HeaderArray dataclass

**Files:**
- Create: `src/equilibria/babel/har/symbols.py`

- [ ] **Step 1: Write the failing test**

Create `tests/babel/har/__init__.py` (empty) and `tests/babel/har/test_reader.py`:

```python
from equilibria.babel.har.symbols import HeaderArray
import numpy as np

def test_header_array_creation():
    arr = HeaderArray(
        name="VDPP",
        coeff_name="VDPP",
        long_name="domestic purchases by households",
        array=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        set_names=["COMM", "REG"],
        set_elements=[["AGR", "MFG", "SER"], ["USA", "ROW"]],
    )
    assert arr.name == "VDPP"
    assert arr.rank == 2
    assert arr.shape == (3, 2)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/babel/har/test_reader.py::test_header_array_creation -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'equilibria.babel.har'`

- [ ] **Step 3: Write `symbols.py`**

```python
# src/equilibria/babel/har/symbols.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class HeaderArray:
    """Represents a single header array from a GEMPACK HAR file."""
    name: str
    coeff_name: str
    long_name: str
    array: np.ndarray
    set_names: list[str]
    set_elements: list[list[str]]

    @property
    def rank(self) -> int:
        return self.array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape
```

Also create `src/equilibria/babel/har/__init__.py` (empty for now):
```python
# src/equilibria/babel/har/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/babel/har/test_reader.py::test_header_array_creation -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/equilibria/babel/har/symbols.py src/equilibria/babel/har/__init__.py tests/babel/har/__init__.py tests/babel/har/test_reader.py
git commit -m "feat(babel.har): add HeaderArray dataclass and test scaffold"
```

---

## Task 3: `reader.py` — thin harpy3 wrapper

**Files:**
- Create: `src/equilibria/babel/har/reader.py`
- Modify: `tests/babel/har/test_reader.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/babel/har/test_reader.py`:

```python
from pathlib import Path
from equilibria.babel.har.reader import read_har, get_header_names, read_header_array

NUS333_BASE = Path("/Users/marmol/Downloads/10284/basedata.har")
NUS333_SETS = Path("/Users/marmol/Downloads/10284/sets.har")

def test_get_header_names():
    names = get_header_names(NUS333_BASE)
    assert "VDPP" in names
    assert "VMSB" in names
    assert "VKB" in names

def test_read_har_returns_dict():
    data = read_har(NUS333_BASE)
    assert "VDPP" in data
    arr = data["VDPP"]
    assert arr.shape == (3, 2)   # (COMM, REG)
    assert arr.set_names == ["COMM", "REG"]

def test_read_har_3d():
    data = read_har(NUS333_BASE)
    vmsb = data["VMSB"]
    assert vmsb.shape == (3, 2, 2)   # (COMM, REG, REG)

def test_read_har_select():
    data = read_har(NUS333_BASE, select_headers=["VDPP", "VKB"])
    assert set(data.keys()) == {"VDPP", "VKB"}

def test_read_header_array_elements():
    data = read_har(NUS333_SETS)
    reg = data["REG"]
    assert list(reg.set_elements[0]) == ["USA", "ROW"]

def test_missing_file_raises():
    import pytest
    with pytest.raises(FileNotFoundError):
        read_har(Path("/does/not/exist.har"))
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/pytest tests/babel/har/test_reader.py -k "not test_header_array" -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Write `reader.py`**

```python
# src/equilibria/babel/har/reader.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from equilibria.babel.har.symbols import HeaderArray


def read_har(
    filepath: str | Path,
    select_headers: list[str] | None = None,
) -> dict[str, HeaderArray]:
    """Read a GEMPACK HAR file and return its header arrays.

    Args:
        filepath: Path to the .har or .prm file.
        select_headers: If provided, only these headers are loaded.

    Returns:
        Dict mapping header name → HeaderArray.

    Raises:
        FileNotFoundError: If filepath does not exist.
        ImportError: If harpy3 is not installed.
    """
    try:
        import harpy
    except ImportError as e:
        raise ImportError(
            "harpy3 is required to read HAR files. "
            "Install it with: pip install harpy3"
        ) from e

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"HAR file not found: {filepath}")

    hf = harpy.HarFileObj._loadFromDisk(
        str(filepath),
        ha_names=select_headers,
    )

    result: dict[str, HeaderArray] = {}
    for name in hf.getHeaderArrayNames():
        obj = hf[name]
        if obj.array is None:
            continue
        arr = np.array(obj.array)
        raw_set_elements = obj.sets  # harpy SetElement object
        set_elements: list[list[str]] = []
        for sn in obj.setNames:
            if sn is None:
                set_elements.append([])
                continue
            # harpy stores elements in the SetElement object keyed by name
            try:
                elems = [str(e).strip() for e in raw_set_elements[sn]]
            except (KeyError, TypeError):
                elems = []
            set_elements.append(elems)
        result[name] = HeaderArray(
            name=name,
            coeff_name=obj.coeff_name.strip() if obj.coeff_name else name,
            long_name=obj.long_name.strip() if obj.long_name else "",
            array=arr,
            set_names=[sn for sn in obj.setNames],
            set_elements=set_elements,
        )
    return result


def get_header_names(filepath: str | Path) -> list[str]:
    """Return all header names in a HAR file without loading array data."""
    try:
        import harpy
    except ImportError as e:
        raise ImportError("harpy3 is required. Install with: pip install harpy3") from e
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"HAR file not found: {filepath}")
    hf = harpy.HarFileObj._loadFromDisk(str(filepath))
    return hf.getHeaderArrayNames()


def read_header_array(
    filepath: str | Path,
    name: str,
) -> HeaderArray:
    """Read a single named header array from a HAR file."""
    data = read_har(filepath, select_headers=[name])
    if name not in data:
        raise KeyError(f"Header '{name}' not found in {filepath}")
    return data[name]
```

- [ ] **Step 4: Fix harpy SetElement access**

Verify how harpy exposes set elements (the `obj.sets` attribute). Run this probe first:

```bash
.venv/bin/python -c "
import harpy
hf = harpy.HarFileObj._loadFromDisk('/Users/marmol/Downloads/10284/sets.har')
arr = hf['REG']
print(type(arr.sets))
print(dir(arr.sets))
se = arr.sets
print(list(se))
"
```

If `arr.sets` is not subscriptable by name, use `arr.setElements` instead (a list of lists). Update `reader.py` accordingly:

```python
# Alternative if obj.sets is not a dict:
for i, sn in enumerate(obj.setNames):
    try:
        elems = [str(e).strip() for e in obj.setElements[i]]
    except (IndexError, TypeError, AttributeError):
        elems = []
    set_elements.append(elems)
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/pytest tests/babel/har/test_reader.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/equilibria/babel/har/reader.py tests/babel/har/test_reader.py
git commit -m "feat(babel.har): add read_har/get_header_names reader wrapping harpy3"
```

---

## Task 4: `__init__.py` — public API

**Files:**
- Modify: `src/equilibria/babel/har/__init__.py`

- [ ] **Step 1: Write failing test**

Add to `tests/babel/har/test_reader.py`:

```python
def test_public_api_import():
    from equilibria.babel.har import read_har, get_header_names, HeaderArray
    assert callable(read_har)
    assert callable(get_header_names)
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/bin/pytest tests/babel/har/test_reader.py::test_public_api_import -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Write `__init__.py`**

```python
# src/equilibria/babel/har/__init__.py
"""equilibria.babel.har — GEMPACK HAR file reader.

Reads .har and .prm files produced by GEMPACK (RunGTAP/GEMPACK software)
without requiring GEMPACK to be installed.

Requires: harpy3>=0.3  (install with: pip install harpy3)
"""

from equilibria.babel.har.reader import (
    get_header_names,
    read_har,
    read_header_array,
)
from equilibria.babel.har.symbols import HeaderArray

__all__ = [
    "HeaderArray",
    "get_header_names",
    "read_har",
    "read_header_array",
]
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/babel/har/ -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/equilibria/babel/har/__init__.py
git commit -m "feat(babel.har): expose public API from __init__.py"
```

---

## Task 5: `GTAPSets.load_from_har()`

**Files:**
- Modify: `src/equilibria/templates/gtap/gtap_parameters.py` (GTAPSets class, around line 96)

The `GTAPSets` class currently only has `load_from_gdx()`. Add `load_from_har()` that reads `sets.har`.

GEMPACK set names map to internal names:
- REG → `r` (regions)
- COMM → `i` (commodities, all traded)
- ACTS → `a` (activities/sectors)
- ENDW → `f` (endowments/factors)
- MARG → `m` (margin goods — subset of COMM)

- [ ] **Step 1: Write failing test**

Add to `tests/babel/har/test_reader.py`:

```python
from equilibria.templates.gtap.gtap_parameters import GTAPSets
from pathlib import Path

NUS333_SETS = Path("/Users/marmol/Downloads/10284/sets.har")

def test_gtap_sets_load_from_har():
    sets = GTAPSets()
    sets.load_from_har(NUS333_SETS)
    assert sets.r == ["USA", "ROW"]
    assert sets.i == ["AGR", "MFG", "SER"]
    assert sets.a == ["AGR", "MFG", "SER"]
    assert sets.f == ["LAND", "LABOR", "CAPITAL"]
    assert sets.m == ["SER"]
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/bin/pytest tests/babel/har/test_reader.py::test_gtap_sets_load_from_har -v
```

Expected: FAIL with `AttributeError: 'GTAPSets' object has no attribute 'load_from_har'`

- [ ] **Step 3: Add `load_from_har()` to GTAPSets**

In `src/equilibria/templates/gtap/gtap_parameters.py`, find the `GTAPSets` class and add after `load_from_gdx()`:

```python
def load_from_har(self, sets_path: Path) -> None:
    """Load set definitions from a GEMPACK sets.har file.

    Args:
        sets_path: Path to sets.har (contains REG, COMM, ACTS, ENDW, MARG arrays).
    """
    from equilibria.babel.har import read_har
    data = read_har(sets_path, select_headers=["REG", "COMM", "ACTS", "ENDW", "MARG"])
    def _elems(name: str) -> list[str]:
        if name not in data:
            return []
        return [str(e).strip() for e in data[name].array]
    self.r = _elems("REG")
    self.i = _elems("COMM")
    self.a = _elems("ACTS")
    self.f = _elems("ENDW")
    self.m = _elems("MARG")
    # rp = all trading partners = r (same set in GTAP)
    self.rp = list(self.r)
```

- [ ] **Step 4: Run test**

```bash
.venv/bin/pytest tests/babel/har/test_reader.py::test_gtap_sets_load_from_har -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/equilibria/templates/gtap/gtap_parameters.py tests/babel/har/test_reader.py
git commit -m "feat(gtap): add GTAPSets.load_from_har() for GEMPACK sets.har"
```

---

## Task 6: `GTAPBenchmark.load_from_har()`

**Files:**
- Modify: `src/equilibria/templates/gtap/gtap_parameters.py` (GTAPBenchmark class, around line 811)

`GTAPBenchmark` holds `~35` monetary dicts. All come from `basedata.har`. The mapping reuses `GTAP_STD7_INDEX_REORDER` from `gtap_std7_mapping.py`.

Key monetary headers in `basedata.har` and their internal dict names:

| HAR header | Internal attr | ndim | Reorder | Note |
|------------|---------------|------|---------|------|
| VDPP | vdpp | 2 | (1,0) | domestic private |
| VMPP | vmpp | 2 | (1,0) | imported private |
| VDPB | vdpb | 2 | (1,0) | |
| VMPB | vmpb | 2 | (1,0) | |
| VDGP | vdgp | 2 | (1,0) | domestic govt |
| VMGP | vmgp | 2 | (1,0) | imported govt |
| VDGB | vdgb | 2 | (1,0) | |
| VMGB | vmgb | 2 | (1,0) | |
| VDIP | vdip | 2 | (1,0) | domestic invest |
| VMIP | vmip | 2 | (1,0) | imported invest |
| VDIB | vdib | 2 | (1,0) | |
| VMIB | vmib | 2 | (1,0) | |
| VDFP | vdfp | 3 | (1,0,2)→(r,i,a) | domestic interm |
| VMFP | vmfp | 3 | (1,0,2) | imported interm |
| VDFB | vdfb | 3 | (1,0,2) | |
| VMFB | vmfb | 3 | (1,0,2) | |
| EVFP | vfm | 3 | (1,0,2)→(r,f,a) | but sets are ENDW,ACTS,REG |
| EVFB | evfb | 3 | (2,0,1)→(r,f,a) | ENDW,ACTS,REG |
| EVOS | evos | 3 | (2,0,1) | ENDW,ACTS,REG |
| VOSB | vom | 2 | (1,0) | ACTS,REG |
| MAKB | makb | 3 | (1,2,0)→(r,a,i) | REG,ACTS,COMM? check |
| MAKS | maks | 3 | same as MAKB | |
| VXSB | vxsb | 3 | (1,0,2) | COMM,REG,REG |
| VFOB | vfob | 3 | (1,0,2) | COMM,REG,REG |
| VCIF | vcif | 3 | (1,0,2) | COMM,REG,REG |
| VMSB | vmsb | 3 | (1,0,2) | COMM,REG,REG |
| VTWR | vtwr | 4 | (2,1,3,0) | MARG,COMM,REG,REG |
| VST | vst | 2 | (1,0) | COMM,REG |
| SAVE | save | 1 | none | REG |
| VDEP | vdep | 1 | none | REG |
| VKB | vkb | 1 | none | REG |

`inScale = 1e-6` is applied to all monetary values (same as `load_from_gdx`).

- [ ] **Step 1: Write failing test**

Add to `tests/babel/har/test_reader.py`:

```python
from equilibria.templates.gtap.gtap_parameters import GTAPSets, GTAPBenchmarkData

NUS333_BASE = Path("/Users/marmol/Downloads/10284/basedata.har")

def test_gtap_benchmark_load_from_har():
    sets = GTAPSets()
    sets.load_from_har(Path("/Users/marmol/Downloads/10284/sets.har"))
    bench = GTAPBenchmarkData()
    bench.load_from_har(NUS333_BASE, sets)
    # VDPP[AGR, USA] should be non-zero; key order is (r, i)
    assert bench.vdpp.get(("USA", "AGR"), 0.0) > 0
    # VMSB shape is (COMM, REG, REG) → (r, i, rp)
    assert len(bench.vmsb) > 0
    # scaling: values should be in trillions (÷1e6), so around 0.05 for USA AGR private
    val = bench.vdpp.get(("USA", "AGR"), 0.0)
    assert 0.0 < val < 1.0  # ~0.055 trillion USD for AGR USA
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/bin/pytest tests/babel/har/test_reader.py::test_gtap_benchmark_load_from_har -v
```

Expected: FAIL with `AttributeError: 'GTAPBenchmarkData' object has no attribute 'load_from_har'`

- [ ] **Step 3: Add helper `_har_to_dict()` and `load_from_har()` to GTAPBenchmark**

In `src/equilibria/templates/gtap/gtap_parameters.py`, find `GTAPBenchmarkData` and add:

```python
@staticmethod
def _har_to_dict(
    har_data: dict,
    header: str,
    sets: "GTAPSets",
    set_order: list[str],
    reorder: tuple[int, ...] | None,
    scale: float = 1.0,
) -> dict:
    """Convert a HAR header array to a sparse Python dict.

    Args:
        har_data: Output of read_har().
        header: Header name (e.g. 'VDPP').
        sets: GTAPSets (for element lists).
        set_order: GEMPACK dimension names in HAR order, e.g. ['COMM','REG'].
        reorder: Index permutation from HAR order to internal key order, or None.
        scale: Scalar multiplier (use 1e-6 for inScale).
    """
    if header not in har_data:
        return {}
    ha = har_data[header]
    arr = ha.array
    # Map set names to element lists
    _SET_MAP = {
        "REG": sets.r, "COMM": sets.i, "ACTS": sets.a,
        "ENDW": sets.f, "MARG": sets.m,
    }
    dim_elements = [_SET_MAP.get(sn, []) for sn in set_order]
    result: dict = {}
    import itertools
    for indices in itertools.product(*[range(len(e)) for e in dim_elements]):
        val = float(arr[indices]) * scale
        if val == 0.0:
            continue
        raw_key = tuple(dim_elements[d][i] for d, i in enumerate(indices))
        if reorder is not None:
            key = tuple(raw_key[i] for i in reorder)
        else:
            key = raw_key if len(raw_key) > 1 else raw_key[0]
        result[key] = val
    return result

def load_from_har(self, basedata_path: Path, sets: "GTAPSets") -> None:
    """Load benchmark values from a GEMPACK basedata.har file."""
    from equilibria.babel.har import read_har
    har = read_har(basedata_path)
    S = 1e-6  # inScale

    def _h(header, set_order, reorder, scale=S):
        return self._har_to_dict(har, header, sets, set_order, reorder, scale)

    # 2D (COMM, REG) → (r, i): reorder (1, 0)
    r10 = (1, 0)
    self.vdpp.update(_h("VDPP", ["COMM","REG"], r10))
    self.vmpp.update(_h("VMPP", ["COMM","REG"], r10))
    self.vdpb.update(_h("VDPB", ["COMM","REG"], r10))
    self.vmpb.update(_h("VMPB", ["COMM","REG"], r10))
    self.vdgp.update(_h("VDGP", ["COMM","REG"], r10))
    self.vmgp.update(_h("VMGP", ["COMM","REG"], r10))
    self.vdgb.update(_h("VDGB", ["COMM","REG"], r10))
    self.vmgb.update(_h("VMGB", ["COMM","REG"], r10))
    self.vdip.update(_h("VDIP", ["COMM","REG"], r10))
    self.vmip.update(_h("VMIP", ["COMM","REG"], r10))
    self.vdib.update(_h("VDIB", ["COMM","REG"], r10))
    self.vmib.update(_h("VMIB", ["COMM","REG"], r10))
    self.vst.update(_h("VST",   ["COMM","REG"], r10))

    # 2D (ACTS, REG) → (r, a): reorder (1, 0)
    self.vom.update(_h("VOSB", ["ACTS","REG"], r10))

    # 3D intermediate (COMM, ACTS, REG) → (r, i, a): reorder (2, 0, 1)
    r201 = (2, 0, 1)
    self.vdfp.update(_h("VDFP", ["COMM","ACTS","REG"], r201))
    self.vmfp.update(_h("VMFP", ["COMM","ACTS","REG"], r201))
    self.vdfb.update(_h("VDFB", ["COMM","ACTS","REG"], r201))
    self.vmfb.update(_h("VMFB", ["COMM","ACTS","REG"], r201))

    # 3D factors (ENDW, ACTS, REG) → (r, f, a): reorder (2, 0, 1)
    self.vfm.update(_h("EVFP",  ["ENDW","ACTS","REG"], r201))
    self.evfb.update(_h("EVFB", ["ENDW","ACTS","REG"], r201))
    self.evos.update(_h("EVOS", ["ENDW","ACTS","REG"], r201))

    # 3D make matrix (REG, ACTS, COMM) → (r, a, i): reorder (0, 1, 2) = identity
    self.makb.update(_h("MAKB", ["REG","ACTS","COMM"], None))
    self.maks.update(_h("MAKS", ["REG","ACTS","COMM"], None))

    # 3D trade (COMM, REG, REG) → (r, i, rp): reorder (1, 0, 2)
    r102 = (1, 0, 2)
    self.vxsb.update(_h("VXSB", ["COMM","REG","REG"], r102))
    self.vfob.update(_h("VFOB", ["COMM","REG","REG"], r102))
    self.vcif.update(_h("VCIF", ["COMM","REG","REG"], r102))
    self.vmsb.update(_h("VMSB", ["COMM","REG","REG"], r102))

    # 4D transport (MARG, COMM, REG, REG) → (r, i, rp, m): reorder (2, 1, 3, 0)
    self.vtwr.update(_h("VTWR", ["MARG","COMM","REG","REG"], (2,1,3,0)))

    # 1D (REG,): no reorder, scalar key
    self.save.update(_h("SAVE", ["REG"], None, scale=S))
    self.vdep.update(_h("VDEP", ["REG"], None, scale=S))
    self.vkb.update(_h("VKB",   ["REG"], None, scale=S))
```

- [ ] **Step 4: Fix scalar key handling in `_har_to_dict`**

For 1D arrays, the key should be a plain string, not a 1-tuple. The current code `raw_key[0]` handles this. Verify with:

```python
assert isinstance(next(iter(bench.save.keys())), str)   # "USA" not ("USA",)
```

- [ ] **Step 5: Run test**

```bash
.venv/bin/pytest tests/babel/har/test_reader.py::test_gtap_benchmark_load_from_har -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/equilibria/templates/gtap/gtap_parameters.py tests/babel/har/test_reader.py
git commit -m "feat(gtap): add GTAPBenchmarkData.load_from_har() mapping GEMPACK headers"
```

---

## Task 7: `GTAPElasticities.load_from_har()`

**Files:**
- Modify: `src/equilibria/templates/gtap/gtap_parameters.py` (GTAPElasticities class, around line 96)

Elasticities come from `default.prm`. The GEMPACK coeff names differ from header names:

| HAR header | coeff_name | Internal attr | set_order | reorder |
|------------|------------|---------------|-----------|---------|
| ESBD | ESUBD | esubd | COMM,REG | (1,0) |
| ESBM | ESUBM | esubm | COMM,REG | (1,0) |
| ESBV | ESUBVA | esubva | ACTS,REG | (1,0) |
| ESBT | ESUBT | esubt | ACTS,REG | (1,0) |
| ETRE | ETRAE | etrae | ENDW,REG | (1,0) → (r,f) |
| ETRQ | ETRAQ | etraq | ACTS,REG | (1,0) |
| ESBQ | ESUBQ | esubq | COMM,REG | (1,0) |
| ESBG | ESUBG | esubg | REG | none |
| ESBI | ESUBI | esubi | REG | none |
| INCP | INCPAR | incpar | COMM,REG | (1,0) |
| SUBP | SUBPAR | subpar | COMM,REG | (1,0) |

- [ ] **Step 1: Write failing test**

Add to `tests/babel/har/test_reader.py`:

```python
from equilibria.templates.gtap.gtap_parameters import GTAPSets, GTAPElasticities

NUS333_DEFAULT = Path("/Users/marmol/Downloads/10284/default.prm")

def test_gtap_elasticities_load_from_har():
    sets = GTAPSets()
    sets.load_from_har(Path("/Users/marmol/Downloads/10284/sets.har"))
    elast = GTAPElasticities()
    elast.load_from_har(NUS333_DEFAULT, sets)
    assert ("USA", "AGR") in elast.esubd
    assert elast.esubd[("USA", "AGR")] > 0
    assert ("USA", "AGR") in elast.esubm
    assert ("USA", "AGR") in elast.esubva
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/bin/pytest tests/babel/har/test_reader.py::test_gtap_elasticities_load_from_har -v
```

Expected: FAIL

- [ ] **Step 3: Add `load_from_har()` to GTAPElasticities**

In the `GTAPElasticities` class, add (reusing the same `_har_to_dict` pattern; note `GTAPElasticities` doesn't have `_har_to_dict` — use the one on `GTAPBenchmarkData` as a module-level helper, or duplicate the logic):

```python
def load_from_har(self, default_path: Path, sets: "GTAPSets") -> None:
    """Load elasticities from a GEMPACK default.prm file."""
    from equilibria.babel.har import read_har
    from equilibria.templates.gtap.gtap_parameters import GTAPBenchmarkData
    har = read_har(default_path)

    def _h(header, set_order, reorder):
        return GTAPBenchmarkData._har_to_dict(har, header, sets, set_order, reorder, scale=1.0)

    r10 = (1, 0)
    self.esubd.update(_h("ESBD", ["COMM","REG"], r10))
    self.esubm.update(_h("ESBM", ["COMM","REG"], r10))
    self.esubva.update(_h("ESBV", ["ACTS","REG"], r10))
    self.esubt.update(_h("ESBT", ["ACTS","REG"], r10))
    self.etraq.update(_h("ETRQ", ["ACTS","REG"], r10))
    self.esubq.update(_h("ESBQ", ["COMM","REG"], r10))
    self.incpar.update(_h("INCP", ["COMM","REG"], r10))
    self.subpar.update(_h("SUBP", ["COMM","REG"], r10))
    self.esubg.update(_h("ESBG", ["REG"], None))
    self.esubi.update(_h("ESBI", ["REG"], None))
    # ETRAE: (ENDW,REG)→(r,f)
    self.etrae.update(_h("ETRE", ["ENDW","REG"], r10))
    self.initialize_nested_elasticities(sets)
```

- [ ] **Step 4: Run test**

```bash
.venv/bin/pytest tests/babel/har/test_reader.py::test_gtap_elasticities_load_from_har -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/equilibria/templates/gtap/gtap_parameters.py tests/babel/har/test_reader.py
git commit -m "feat(gtap): add GTAPElasticities.load_from_har() for default.prm"
```

---

## Task 8: `GTAPTaxes.load_from_har()`

**Files:**
- Modify: `src/equilibria/templates/gtap/gtap_parameters.py` (GTAPTaxes class, around line 1104)

Tax rates come from `baserate.har`. GEMPACK headers and dimension orders:

| HAR header | Internal attr | set_order | reorder | Notes |
|------------|---------------|-----------|---------|-------|
| RTO | rto | COMM,ACTS,REG | (2,0,1)→(r,a,?) | output tax |
| RTPD | rtpd | COMM,REG | (1,0) | private dom |
| RTPM | rtpi | COMM,REG | (1,0) | private imp |
| RTGD | rtgd | COMM,REG | (1,0) | govt dom |
| RTGM | rtgi | COMM,REG | (1,0) | govt imp |
| RTID | — | COMM,REG | (1,0) | invest dom (skip if not in GTAPTaxes) |
| RTIM | — | COMM,REG | (1,0) | invest imp (skip if not in GTAPTaxes) |
| RTFD | rtfd | COMM,ACTS,REG | (2,0,1) | interm dom |
| RTFM | rtfi | COMM,ACTS,REG | (2,0,1) | interm imp |
| RTIN | rtf | ENDW,ACTS,REG | (2,0,1) | factor tax |
| RTXS | rtxs | COMM,REG,REG | (1,0,2) | export subsidy |
| RTMS | rtms | COMM,REG,REG | (1,0,2) | import tariff → imptx |

- [ ] **Step 1: Write failing test**

Add to `tests/babel/har/test_reader.py`:

```python
from equilibria.templates.gtap.gtap_parameters import GTAPSets, GTAPTaxes, GTAPBenchmarkData

NUS333_RATE = Path("/Users/marmol/Downloads/10284/baserate.har")

def test_gtap_taxes_load_from_har():
    sets = GTAPSets()
    sets.load_from_har(Path("/Users/marmol/Downloads/10284/sets.har"))
    bench = GTAPBenchmarkData()
    bench.load_from_har(Path("/Users/marmol/Downloads/10284/basedata.har"), sets)
    taxes = GTAPTaxes()
    taxes.load_from_har(NUS333_RATE, sets, bench)
    # imptx should be derived from rtms
    assert len(taxes.imptx) > 0
    # rtxs should have some entries
    assert len(taxes.rtxs) > 0
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/bin/pytest tests/babel/har/test_reader.py::test_gtap_taxes_load_from_har -v
```

Expected: FAIL

- [ ] **Step 3: Add `load_from_har()` to GTAPTaxes**

```python
def load_from_har(self, baserate_path: Path, sets: "GTAPSets", benchmark: "GTAPBenchmarkData") -> None:
    """Load tax rates from a GEMPACK baserate.har file."""
    from equilibria.babel.har import read_har
    from equilibria.templates.gtap.gtap_parameters import GTAPBenchmarkData
    har = read_har(baserate_path)

    def _h(header, set_order, reorder, scale=1.0):
        return GTAPBenchmarkData._har_to_dict(har, header, sets, set_order, reorder, scale)

    r10 = (1, 0)
    r201 = (2, 0, 1)
    r102 = (1, 0, 2)

    self.rtpd.update(_h("RTPD", ["COMM","REG"], r10))
    self.rtpi.update(_h("RTPM", ["COMM","REG"], r10))
    self.rtgd.update(_h("RTGD", ["COMM","REG"], r10))
    self.rtgi.update(_h("RTGM", ["COMM","REG"], r10))
    self.rtfd.update(_h("RTFD", ["COMM","ACTS","REG"], r201))
    self.rtfi.update(_h("RTFM", ["COMM","ACTS","REG"], r201))
    self.rtf.update(_h("RTIN",  ["ENDW","ACTS","REG"], r201))
    self.rto.update(_h("RTO",   ["COMM","ACTS","REG"], r201))
    self.rtxs.update(_h("RTXS", ["COMM","REG","REG"], r102))
    self.rtms.update(_h("RTMS", ["COMM","REG","REG"], r102))

    # Derive imptx from rtms (ad-valorem import tariff rate)
    if self.rtms:
        self.imptx = {k: float(v) / 1000.0 for k, v in self.rtms.items()}

    self.derive_from_benchmark(benchmark, sets)
```

- [ ] **Step 4: Run test**

```bash
.venv/bin/pytest tests/babel/har/test_reader.py::test_gtap_taxes_load_from_har -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/equilibria/templates/gtap/gtap_parameters.py tests/babel/har/test_reader.py
git commit -m "feat(gtap): add GTAPTaxes.load_from_har() for baserate.har"
```

---

## Task 9: `GTAPParameters.load_from_har()` — top-level entry point

**Files:**
- Modify: `src/equilibria/templates/gtap/gtap_parameters.py` (GTAPParameters class, around line 2015)

This wires the four sub-loaders into one call, mirroring `load_from_gdx()`.

- [ ] **Step 1: Write failing integration test**

Add to `tests/babel/har/test_reader.py`:

```python
from equilibria.templates.gtap.gtap_parameters import GTAPParameters

def test_gtap_parameters_load_from_har_roundtrip():
    params = GTAPParameters()
    params.load_from_har(
        basedata_path=Path("/Users/marmol/Downloads/10284/basedata.har"),
        sets_path=Path("/Users/marmol/Downloads/10284/sets.har"),
        default_path=Path("/Users/marmol/Downloads/10284/default.prm"),
        baserate_path=Path("/Users/marmol/Downloads/10284/baserate.har"),
    )
    # Sets loaded
    assert params.sets.r == ["USA", "ROW"]
    assert params.sets.i == ["AGR", "MFG", "SER"]
    # Benchmark loaded and scaled
    assert params.benchmark.vdpp.get(("USA", "AGR"), 0.0) > 0
    # Elasticities loaded
    assert params.elasticities.esubd.get(("USA", "AGR"), 0.0) > 0
    # Taxes loaded
    assert len(params.taxes.imptx) > 0
    # Shares calibrated
    assert len(params.shares.alphac) > 0 or len(params.calibrated.and_) > 0
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/bin/pytest tests/babel/har/test_reader.py::test_gtap_parameters_load_from_har_roundtrip -v
```

Expected: FAIL

- [ ] **Step 3: Add `load_from_har()` to GTAPParameters**

In the `GTAPParameters` class, after the existing `load_from_gdx()` method:

```python
def load_from_har(
    self,
    basedata_path: Path,
    sets_path: Path,
    default_path: Path,
    baserate_path: Path,
) -> None:
    """Load all parameters from GEMPACK HAR/PRM files.

    Args:
        basedata_path: Path to basedata.har (benchmark monetary flows).
        sets_path: Path to sets.har (REG, COMM, ACTS, ENDW, MARG).
        default_path: Path to default.prm (elasticities).
        baserate_path: Path to baserate.har (tax rates).
    """
    self.sets.load_from_har(sets_path)
    self.elasticities.load_from_har(default_path, self.sets)
    self.benchmark.load_from_har(basedata_path, self.sets)
    self.taxes.load_from_har(baserate_path, self.sets, self.benchmark)
    self.taxes.derive_from_benchmark(self.benchmark, self.sets)
    self.shares.calibrate(self.benchmark, self.elasticities, self.sets)
    self.calibrated.calibrate_from_benchmark(
        self.benchmark, self.elasticities, self.sets, self.taxes
    )
```

- [ ] **Step 4: Run integration test**

```bash
.venv/bin/pytest tests/babel/har/test_reader.py::test_gtap_parameters_load_from_har_roundtrip -v
```

Expected: PASS

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
.venv/bin/pytest tests/babel/har/ tests/babel/gdx/ -v --tb=short
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/equilibria/templates/gtap/gtap_parameters.py tests/babel/har/test_reader.py
git commit -m "feat(gtap): add GTAPParameters.load_from_har() — full HAR pipeline entry point"
```

---

## Task 10: Smoke test — build the NUS333 model

**Files:**
- Create: `scripts/gtap/validate_nus333_parity.py`

This is a quick smoke test that loads NUS333 into `GTAPParameters`, builds the Pyomo model, and runs the nonlinear PATH baseline — confirming the full pipeline works end-to-end.

- [ ] **Step 1: Write the smoke script**

Create `scripts/gtap/validate_nus333_parity.py`:

```python
"""Smoke test: load NUS333 via HAR, build Pyomo model, solve baseline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from equilibria.templates.gtap import GTAPParameters
from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
from equilibria.templates.gtap.gtap_contract import build_gtap_contract
from run_gtap import _run_path_capi_nonlinear_full, _build_gtap_contract_with_calibration

NUS333 = Path("/Users/marmol/Downloads/10284")

params = GTAPParameters()
params.load_from_har(
    basedata_path=NUS333 / "basedata.har",
    sets_path=NUS333 / "sets.har",
    default_path=NUS333 / "default.prm",
    baserate_path=NUS333 / "baserate.har",
)
print(f"Sets: r={params.sets.r}, i={params.sets.i}")
print(f"Benchmark vdpp entries: {len(params.benchmark.vdpp)}")

contract = _build_gtap_contract_with_calibration("gtap_standard7_9x10")
eq = GTAPModelEquations(params.sets, params, contract.closure)
model = eq.build_model()

result = _run_path_capi_nonlinear_full(
    model, params,
    enforce_post_checks=False,
    strict_path_capi=False,
    closure_config=contract.closure,
    equation_scaling=True,
)
print(f"Baseline: residual={result['residual']:.2e}, code={result['termination_code']}")
```

- [ ] **Step 2: Run the smoke test**

```bash
.venv/bin/python scripts/gtap/validate_nus333_parity.py 2>&1 | tail -5
```

Expected output (approximate):
```
Sets: r=['USA', 'ROW'], i=['AGR', 'MFG', 'SER']
Benchmark vdpp entries: 6
Baseline: residual=X.XXe-XX, code=1
```

- [ ] **Step 3: Commit**

```bash
git add scripts/gtap/validate_nus333_parity.py
git commit -m "feat(gtap): add NUS333 HAR smoke test — baseline solve via HAR pipeline"
```

---

## Self-Review

**Spec coverage:** All requirements covered — HAR reader module, GTAPParameters integration, NUS333 smoke test.

**Placeholder scan:** No TBDs. Every code block is complete.

**Type consistency:**
- `_har_to_dict` is defined as `@staticmethod` on `GTAPBenchmarkData` and referenced by `GTAPElasticities.load_from_har()` and `GTAPTaxes.load_from_har()` — both import it from `GTAPBenchmarkData`. Consistent.
- `read_har()` returns `dict[str, HeaderArray]` — used consistently in all `load_from_har()` methods.
- `GTAPSets.rp` is set to `list(self.r)` — consistent with how the 9×10 model uses it.

**Known risk — Task 3, Step 4:** harpy's `.sets` attribute returns a `SetElement` object that may not be subscriptable by name. The plan includes an explicit probe step and alternative code path. Must be resolved empirically before proceeding to Task 5.

**Known risk — MAKB dimension order:** The HAR `MAKB` header may have dimension order `(REG, ACTS, COMM)` or `(ACTS, COMM, REG)` — needs verification. Task 6, Step 3 assumes `(REG, ACTS, COMM)` = identity reorder `(r, a, i)`. If wrong, update reorder to `(2, 0, 1)` or similar.
