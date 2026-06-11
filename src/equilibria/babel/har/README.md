# equilibria.babel.har

Pure-Python **reader** and **writer** for GEMPACK **HAR** and **`.prm`**
files. No third-party HAR library required — `harpy3` / `harpy` is **NOT**
a runtime dependency.

## Origin and license

Both the reader and the writer are **clean-room reimplementations** developed
by inspecting the on-disk byte layout of HAR files produced by GEMPACK /
RunGTAP and by reading publicly available descriptions of the format. They
were written without viewing, copying, translating, or otherwise deriving
from:

- `harpy3` / `harpy` (GPLv3),
- GEMPACK source distributions (proprietary), or
- any other HAR implementation.

`harpy3` is used only as a black-box oracle in a sandboxed dev workflow
(see `scripts/har/oracle_check.py`) — it never runs in CI and is not
imported by any `equilibria` source.

The interoperability constants used here (Fortran unformatted record framing,
the `1CFULL` / `REFULL` / `RESPSE` / `2IFULL` type tokens, etc.) are the wire
format itself — not copyrightable expression.

Distributed under the MIT License, same as the rest of `equilibria`. See the
top-level `NOTICE` file for the full statement.

## Supported types

| Token    | Meaning                        | Notes                                    |
|----------|--------------------------------|------------------------------------------|
| `1CFULL` | 1-D character set              | Multi-record element lists are handled   |
| `REFULL` | Real dense N-D array           | Repeated set names (e.g. REG×REG) OK     |
| `RESPSE` | Real sparse N-D array          | 1-based Fortran flat indices             |
| `2IFULL` | 2-D integer dense array        | Hard `TypeError` for non-`int32` dtype   |

Reader validated header-by-header against `harpy3` on the full GTAP NUS333
and 9×10 datasets (182 headers total, exact match).

Writer validated by:
- **L3 semantic round-trip** on the 6 GTAP fixtures (`read_har` →
  `write_har` → `read_har` returns identical `HeaderArray`s);
- **L5 oracle goldens** (writer output re-read by `harpy3` in sandbox
  matches the same per-header dump as the original GEMPACK file);
- **L7 integration**: GTAP loader round-trip, alter-tax tariff-shock
  round-trip, pandas DataFrame export, and `harpy3` interop probe.

## Usage

### Reading

```python
from equilibria.babel.har import read_har

data = read_har("basedata.har")          # dict[name -> HeaderArray]
ha = data["VDFB"]
print(ha.long_name, ha.array.shape, ha.set_names)
```

### Writing — round-trip mutation (e.g. alter-tax tariff shock)

```python
from equilibria.babel.har import read_har, write_har

data = read_har("baserate.har")
data["rTMS"].array[...] *= 1.10        # 10% bilateral import-tariff shock
write_har("baserate_shocked.har", data)
```

`write_har` is atomic: it writes to a temp file in the target directory
and `os.replace`s into place. On any emission error the destination
path is left untouched.

### Writing — building a HAR from scratch

```python
import numpy as np
import pandas as pd
from equilibria.babel.har import HarWriter

with HarWriter("synthetic.har") as w:
    # Sets first (auto-registered for any later add_array reference)
    w.add_set("REG",  ["USA", "ROW"])
    w.add_set("COMM", ["AGR", "MFG"])

    # numpy array
    w.add_array(
        "VDPP",
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        set_names=["COMM", "REG"],
        long_name="domestic private purchases",
    )

    # pandas DataFrame — set names taken from index / columns names
    df = pd.DataFrame(
        [[100.0, 200.0], [300.0, 400.0]],
        index=pd.Index(["USA", "ROW"], name="REG"),
        columns=pd.Index(["AGR", "MFG"], name="COMM"),
    )
    w.add_dataframe("REGY", df, long_name="regional output")
```

The set registry detects conflicts (re-adding a set with different
elements raises) and verifies every `add_array`/`add_dataframe` reference
points to a known set.

### Public API

```python
from equilibria.babel.har import (
    read_har,         # path  -> dict[name, HeaderArray]
    write_har,        # path, dict[name, HeaderArray] -> None
    HarWriter,        # builder with add_set / add_array / add_dataframe
    HeaderArray,      # the value type
)
```

## Validation & known limitations

The writer hits **L3 semantic round-trip** exact on all 6 GTAP fixtures
(reading the output produces identical `HeaderArray` objects). It does
**not** reach byte-for-byte equality with the GEMPACK-emitted source,
because two pieces of GEMPACK structural information are discarded on
read:

1. **1CFULL per-header element width** — GEMPACK pads set elements to a
   per-header width (sometimes 12, sometimes 44+); `read_har` strips
   trailing spaces, so the original width can't be recovered.
2. **Sparse vs dense storage choice** — GEMPACK picks `RESPSE` vs
   `REFULL` via internal heuristics; `read_har` densifies on load, so
   the signal is gone by the time `write_har` sees the data.

These divergences are documented as `xfail`s in
`tests/babel/har/test_byte_exact.py` (one per fixture) with sibling
`.diff` sidecars listing the exact offsets. They are **structural, not
semantic** — any conforming HAR reader (ours, harpy3, RunGTAP, ViewHAR)
processes the writer output identically to the original.

## Validation tooling

`scripts/har/oracle_check.py` is a sandboxed harness that uses `harpy3`
as a black-box oracle. Not imported by any `equilibria` source, never
runs in CI. Workflow:

```bash
python3 -m venv .oracle_venv
.oracle_venv/bin/pip install harpy3 numpy
.oracle_venv/bin/python scripts/har/oracle_check.py refresh   # writes
.oracle_venv/bin/python scripts/har/oracle_check.py compare   # checks
```

`refresh` writes per-fixture JSON dumps (shape, set names, set elements,
statistics) to `tests/babel/har/golden/`. `compare` round-trips each
fixture through our writer and asserts harpy3 sees the result identical
to the original. The committed JSON dumps are consumed by
`tests/babel/har/test_oracle_golden.py` in CI without harpy3 installed.
