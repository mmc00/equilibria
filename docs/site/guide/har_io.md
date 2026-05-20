# HAR file I/O

`equilibria.babel.har` provides a pure-Python reader and writer for
GEMPACK **HAR** and **`.prm`** files. No GEMPACK installation, no
`harpy3` runtime dependency — the entire round trip stays in Python.

The module supports the four header types that the GTAP and PEP
fixtures use:

| Token    | Meaning                  | Notes                                    |
|----------|--------------------------|------------------------------------------|
| `1CFULL` | 1-D character set        | Multi-record element lists handled       |
| `REFULL` | Real dense N-D array     | Repeated set names (e.g. `REG × REG`) OK |
| `RESPSE` | Real sparse N-D array    | 1-based Fortran flat indices             |
| `2IFULL` | 2-D integer dense array  | Hard `TypeError` for non-`int32` dtype   |

## Reading a HAR

```python
from equilibria.babel.har import read_har

data = read_har("basedata.har")          # dict[name -> HeaderArray]
vdpp = data["VDPP"]
print(vdpp.long_name, vdpp.array.shape, vdpp.set_names)
print(vdpp.set_elements)                 # [[...COMM...], [...REG...]]
```

Every value is a `HeaderArray` carrying:

- `array` — the dense NumPy array;
- `long_name` — the GEMPACK long name (≤ 70 chars);
- `set_names`, `set_elements` — the set descriptor;
- `coeff_name` — the GEMPACK coefficient name (often equal to the
  header name).

## Round-trip mutation (alter-tax tariff shock)

A common workflow in GTAP work is to load a baseline HAR, mutate one
or more headers, and write the result back for the solver to consume.

```python
from equilibria.babel.har import read_har, write_har

data = read_har("baserate.har")

# 10% uniform bilateral import-tariff shock
data["rTMS"].array[...] *= 1.10

write_har("baserate_shocked.har", data)
```

`write_har` writes **atomically**: it emits to a temporary file in the
target directory and `os.replace`s it into place. On any emission error
the destination is left untouched.

## Building a HAR from scratch

For new datasets, use the `HarWriter` builder. It registers sets up
front, detects conflicts (re-adding a set with different elements
raises), and verifies that every array reference points to a known set.

```python
import numpy as np
import pandas as pd
from equilibria.babel.har import HarWriter

with HarWriter("synthetic.har") as w:
    # Sets first
    w.add_set("REG",  ["USA", "ROW"])
    w.add_set("COMM", ["AGR", "MFG"])

    # numpy array
    w.add_array(
        "VDPP",
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        set_names=["COMM", "REG"],
        long_name="domestic private purchases",
    )

    # pandas DataFrame — set names taken from index / column names
    df = pd.DataFrame(
        [[100.0, 200.0], [300.0, 400.0]],
        index=pd.Index(["USA", "ROW"], name="REG"),
        columns=pd.Index(["AGR", "MFG"], name="COMM"),
    )
    w.add_dataframe("REGY", df, long_name="regional output")
```

The context manager flushes to disk on `__exit__`; if you prefer
explicit lifetime, call `w.close()` yourself.

## Validation

The writer is exercised by four layers of tests:

- **L3 — Semantic round-trip.** For every fixture, `read_har` →
  `write_har` → `read_har` returns identical `HeaderArray` objects
  (same array values, same shape, same sets). Exact on all six GTAP
  fixtures shipped with `equilibria`.
- **L5 — Oracle goldens.** A sandboxed dev workflow runs `harpy3` over
  every fixture and dumps per-header statistics to JSON
  (`tests/babel/har/golden/`). CI re-reads those JSONs (without
  installing `harpy3`) and asserts the writer output matches.
- **L7 — Integration.** End-to-end tests cover (a) loading writer
  output with `GTAPBenchmarkValues`, (b) an alter-tax `rTMS × 1.10`
  round trip, (c) a pandas-DataFrame export, and (d) reading the
  writer output back with `harpy3` when it happens to be installed
  locally.
- **L4 — Byte-exact (documented xfails).** Comparing SHA-256 of the
  writer output to the GEMPACK source. See the next section.

## Byte-exact vs semantic equality

`write_har(read_har(p))` does **not** match `p` byte-for-byte for the
six GTAP reference fixtures. The divergence is structural, not
semantic: two pieces of GEMPACK metadata are discarded on read and
cannot be reconstructed.

1. **1CFULL per-header element width.** GEMPACK pads set elements to a
   per-header width (12 chars for short codes, 44+ for version
   strings). `read_har` strips trailing spaces, so the original width
   is gone by the time the writer sees the data.
2. **Sparse vs dense storage.** GEMPACK picks `RESPSE` vs `REFULL`
   via internal heuristics; `read_har` densifies on load, so the
   sparse-vs-dense signal isn't available.

These divergences are documented as `xfail`s in
`tests/babel/har/test_byte_exact.py` with sibling `.diff` sidecars
listing the exact offsets. They have **no functional impact** — any
conforming HAR reader (ours, `harpy3`, RunGTAP, ViewHAR) decodes the
writer output identically to the original.

## Clean-room provenance

Both the reader and the writer are **clean-room reimplementations**.
They were developed by inspecting the on-disk byte layout of HAR files
produced by GEMPACK / RunGTAP and by reading public format
documentation. No code was viewed, copied, translated, or otherwise
derived from `harpy3` / `harpy` (GPLv3) or from GEMPACK source.

`harpy3` is used **only** as a black-box oracle in a sandboxed dev
harness (`scripts/har/oracle_check.py`). It is never imported by any
`equilibria` source and never runs in CI.

See the top-level `NOTICE` file for the full clean-room statement.
