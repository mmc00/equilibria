# HAR Writer for `equilibria.babel.har` — Design Spec

**Date:** 2026-05-20
**Branch:** `harwriter` worktree
**Status:** draft, awaiting user approval

## 1. Background

`equilibria.babel.har` ships a pure-Python, clean-room **reader** for GEMPACK
HAR (`.har`) and `.prm` files (`src/equilibria/babel/har/reader.py`). The
reader supports four type tokens — `1CFULL`, `REFULL`, `RESPSE`, `2IFULL` —
covering every header in the GTAP NUS333 and 9x10 datasets shipped with the
project. There is currently no counterpart writer; the only direction
implemented is HAR → Python (and HAR → GDX via `har_to_gdx.py`).

This spec defines a clean-room HAR **writer** to close that gap.

## 2. Goals

The writer must enable three workflows, in priority order for v1:

1. **Export numpy arrays and pandas DataFrames to `.har`** — emit Python-side
   data (DataFrames, numpy arrays) so RunGTAP, ViewHAR, and other GEMPACK
   tools can consume it.
2. **Round-trip mutate an existing HAR** — read a HAR (e.g., `baserate.har`),
   modify a coefficient (e.g., apply a tariff shock for alter-tax), write back
   to a new file that RunGTAP accepts.
3. **Construct new HAR datasets from scratch** — build aggregated test
   fixtures or synthetic datasets from numpy arrays without a template HAR.

## 3. License and provenance

The existing reader is a clean-room reimplementation: it was written without
viewing, copying, translating, or otherwise deriving from `harpy3` /
`harpy` (GPLv3), GEMPACK source distributions, or any other HAR
implementation. equilibria is MIT-licensed.

**This writer must preserve that posture.** Specifically:

- No `harpy3` / `harpy` source code may be read at any point during design or
  implementation.
- `harpy3` may be used **only as a black-box oracle** in a sandboxed
  environment: install it in a separate venv, feed it bytes, observe its
  outputs, and capture results as golden fixtures.
- All wire-format knowledge for the writer must come from inspecting bytes of
  GEMPACK-produced HAR files (the same approach used for the reader) and from
  publicly documented format descriptions.
- The wire-format constants used by the writer (record framing, type tokens,
  padding bytes, fixed field widths) are the wire format itself — not
  copyrightable expression.

## 4. Scope

### Type tokens (all four)

The v1 writer supports the same four type tokens as the reader:

| Token    | Meaning                       | Notes                              |
|----------|-------------------------------|------------------------------------|
| `1CFULL` | 1-D character set             | Multi-record element lists when needed |
| `REFULL` | Real dense N-D array          | Repeated set names (e.g. REG×REG) OK |
| `RESPSE` | Real sparse N-D array         | 1-based Fortran flat indices       |
| `2IFULL` | 2-D integer dense array       |                                    |

### Fidelity bar

The writer must satisfy, in order:

- **L1 — Semantic round-trip.** `read_har(write_har(read_har(p)))` equals
  `read_har(p)` HeaderArray-by-HeaderArray (name, long_name, set_names,
  set_elements, array values).
- **L2 — Consumer acceptance.** RunGTAP / ViewHAR / `equilibria`'s own GTAP
  loader read the written file without error and produce identical results.
- **L3 — Oracle equivalence.** `harpy3`, in a sandbox env, reads the written
  file and produces the same arrays/metadata that it produces reading the
  GEMPACK-emitted original. (Captured as committed golden fixtures.)
- **L4 — Byte-for-byte identity.** `sha256(write_har(read_har(p)))` equals
  `sha256(p)` for the six GTAP fixtures shipped under
  `src/equilibria/templates/reference/gtap/data/`: NUS333 `basedata.har`,
  `sets.har`, `baserate.har`, `default.prm`; 9x10 `sets.har`, `basedata.har`.

**Escape hatch for L4.** If a specific fixture cannot reach byte equality
because GEMPACK's emitter introduces non-deterministic content
(uninitialized Fortran memory, timestamps, version-dependent padding), the
corresponding test is marked `xfail` with a sibling `.diff` file documenting
the hex-level divergence and a written note explaining why. Such fixtures
remain locked to L3 (oracle equivalence). Whether to invoke this escape
hatch is decided per-fixture during implementation; the goal is to attempt
L4 first.

## 5. Module structure

```
src/equilibria/babel/har/
├── __init__.py          # exports
├── wire.py    [NEW]     # shared Fortran-record codec + HAR-format primitives
├── reader.py  [SLIM]    # parsers per type token; imports from wire.py
├── writer.py  [NEW]     # emitters per type token; write_har(); HarWriter builder
├── symbols.py           # HeaderArray dataclass (unchanged)
└── README.md  [UPDATED] # writer documented with clean-room provenance
```

### 5.1 `wire.py` — shared wire-format primitives

Single source of truth for the on-disk byte layout. Both reader and writer
import from here.

- Constants: `_PAD = b"    "`, `_INT = struct.Struct("<i")`, type-token
  strings, fixed field widths (set-name field = 12 bytes; long-name field =
  70 bytes; header-name field = 4 bytes).
- `iter_records(buf)` — moved from `reader.py`; yields raw record bytes from
  a Fortran unformatted stream.
- `write_record(out_buf, payload)` — emits a Fortran-framed record: 4-byte
  little-endian length, payload bytes, 4-byte little-endian length suffix.
- `decode_str_block(blk, width)` / `encode_str_block(strings, width)` —
  fixed-width padded string blocks.
- `parse_set_descriptor(rec)` / `build_set_descriptor(coeff_name, set_names)`
  — codec for the per-coefficient set descriptor (read+write symmetric).
- `read_set_element_record(rec)` / `write_set_element_record(elements, width, flag)`
  — codec for the set-elements records, including the multi-record flag.

### 5.2 `reader.py` — slimmed

Behavioral changes: **none**. All wire-format helpers currently inlined
(`_iter_records`, `_decode_str_block`, `_read_set_element_record`,
`_parse_set_descriptor`, `_PAD`, `_INT`) are replaced with imports from
`wire.py`. The existing tests in `tests/babel/har/test_reader.py` are the
regression gate for this refactor.

### 5.3 `writer.py` — emitters and builder

Functions parallel to the reader's `_read_<token>` private functions:

- `_write_1cfull(out, header: HeaderArray) -> None`
- `_write_refull(out, header: HeaderArray) -> None`
- `_write_respse(out, header: HeaderArray) -> None`
- `_write_2ifull(out, header: HeaderArray) -> None`

Each writes the name record + meta record + type-specific data records
via `wire.write_record`.

Low-level entry:

```python
def write_har(
    path: str | Path,
    headers: dict[str, HeaderArray],
) -> None:
    """Write a HAR file from a dict of HeaderArrays.

    Sorts headers so all 1CFULL set headers are emitted first (GEMPACK
    convention), then non-set headers in dict insertion order. Writes
    atomically via a temp file + os.replace.
    """
```

Builder class:

```python
class HarWriter:
    def __init__(self, path: str | Path) -> None: ...
    def add_set(self, name: str, elements: list[str]) -> None: ...
    def add_array(
        self,
        name: str,
        array: np.ndarray,
        set_names: list[str],
        set_elements: list[list[str]] | None = None,
        long_name: str = "",
        *,
        sparse: bool = False,
    ) -> None: ...
    def add_dataframe(
        self,
        name: str,
        df: pd.DataFrame,
        set_names: list[str],
        long_name: str = "",
    ) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> "HarWriter": ...
    def __exit__(self, *exc) -> None: ...  # calls close()
```

## 6. Data flow

### 6.1 Export path (numpy / DataFrame → `.har`)

```
user code
  → HarWriter.add_dataframe(name, df, set_names, long_name)
      ├── df.values → np_array, cast to float32
      ├── set_elements derived from df.index, df.columns
      └── delegates to add_array(...)
  → HarWriter.add_array(name, np_array, set_names, set_elements, long_name)
      ├── for each (set_name, elements): registry.register_or_conflict_check(...)
      ├── classifies type token:
      │     float32/float64 dense   → REFULL
      │     float32/float64 + sparse=True → RESPSE
      │     int32 2-D               → 2IFULL
      └── appends HeaderArray to pending-arrays queue
  → HarWriter.close()
      ├── builds a 1CFULL HeaderArray for each registered set
      ├── concatenates: [set-headers in first-registration order]
      │                 + [pending arrays in add-call order]
      └── delegates to write_har(path, ordered_dict)
  → write_har(path, headers)
      └── for each header → dispatch to _write_<token>
                          → wire.write_record(...) per Fortran record
```

### 6.2 Round-trip path (alter-tax)

```
d = read_har("baserate.har")        # dict[name -> HeaderArray]
d["TM"].array[...] = new_values     # in-place mutation
write_har("baserate_mod.har", d)    # same low-level entry
```

`write_har` emits all `1CFULL` headers first (in dict order), then all
non-`1CFULL` headers (in dict order). The reader returns headers in file
order, and GEMPACK-produced files already place sets before arrays, so the
sort is a no-op for round-tripping a freshly-read HAR: order is preserved
end-to-end. Repeated set names in a descriptor (e.g. `VMSB` on
`COMM × REG × REG`) are preserved on the `HeaderArray.set_names` list; the
writer's set-descriptor encoder emits `ndim = len(set_names)` and the full
list but only emits one element record per unique set, symmetric with the
reader.

## 7. Error handling

### 7.1 `write_har` — strict validation, fail fast

| Condition                                         | Exception              |
|---------------------------------------------------|------------------------|
| `headers` empty                                   | `ValueError`           |
| Header name not exactly 1–4 bytes                 | `ValueError`           |
| `long_name` length > 70                           | `ValueError`           |
| Any set-element name length > 12                  | `ValueError`           |
| `array.ndim != len(set_names)`                    | `ValueError`           |
| `array.shape[k] != len(set_elements[k])`          | `ValueError`           |
| Unsupported dtype (not float32/float64/int32)     | `TypeError`            |
| `2IFULL` requested with non-int32 dtype           | `TypeError`            |
| Path not writable                                 | propagates `OSError`   |
| Header name not uppercase ASCII                   | `UserWarning` (still written) |

### 7.2 `HarWriter` — registry conflicts surface early

- `add_set("REG", A)` then `add_set("REG", B)` with `A != B` → `ValueError`,
  message lists both element lists.
- `add_array(..., set_names=["COMM"], set_elements=[A])` after a prior
  registration of `"COMM"` with `B != A` (via `add_set` or another
  `add_array`) → same `ValueError`.
- `add_dataframe` where `df.ndim != 2` → `ValueError` pointing the user at
  `add_array` for higher ranks.

### 7.3 Atomicity

`write_har` writes to `path + ".tmp"` and `os.replace(tmp, path)` on
success. On any exception during emission, the temp file is removed in
`finally` and `path` is left untouched. This is required so a half-written
`baserate_modified.har` cannot be silently consumed by RunGTAP.

## 8. Testing

Seven layers; all required to pass for v1.

| Layer | File                                          | Purpose                                                                 |
|-------|-----------------------------------------------|-------------------------------------------------------------------------|
| L1    | `tests/babel/har/test_wire.py` (new)          | `wire.py` encode↔decode primitives                                      |
| L2    | `tests/babel/har/test_writer.py` (new)        | per-type-token unit tests with golden hex blobs + builder validation    |
| L3    | `tests/babel/har/test_roundtrip.py` (new)     | reader↔writer semantic round-trip on 6 GTAP fixtures                    |
| L4    | `tests/babel/har/test_byte_exact.py` (new)    | sha256(write(read(p))) == sha256(p) on 6 GTAP fixtures; xfail+`.diff` escape hatch documented per failing fixture |
| L5    | `tests/babel/har/golden/*` + `scripts/har/oracle_check.py` (new) | committed harpy3 outputs as JSON; manual sandbox harness to regenerate |
| L6    | `tests/babel/har/test_reader.py` (existing)   | regression gate; must stay green after `wire.py` extraction             |
| L7    | `tests/babel/har/test_integration.py` (new)   | end-to-end: GTAP loader round-trip; alter-tax `tm*1.10` round-trip; DataFrame export end-to-end; optional `@pytest.mark.skipif(no harpy3)` interop check |

### L5 — oracle workflow

`scripts/har/oracle_check.py` is a manual script the user runs in a venv
with `pip install harpy3`. It:

1. Calls our writer on a fixture.
2. Loads the output with `harpy3` in the same venv.
3. Loads the GEMPACK-emitted original with `harpy3`.
4. Prints a structured diff and writes/updates JSON files under
   `tests/babel/har/golden/` capturing per-header shape, set_names,
   set_elements, sum, and first/last values from harpy3.

The committed golden JSONs are what CI tests check against. `harpy3` is
never installed in CI; the oracle script is run on demand by the developer
adding or debugging a fixture.

## 9. Out of scope for v1

The following are deliberately deferred:

- **Mutation context manager.** `HarFile("p", mode="r+")` with shape/dtype
  validation on `__setitem__`. The pure-functional pattern (`read_har` →
  mutate `HeaderArray.array` in place → `write_har`) is sufficient for
  alter-tax v1.
- **Extra type tokens** beyond the four reader-supported ones (`RECFUL`,
  `1CSET`, `CHFULL`, etc.). Added on demand when a real fixture in
  equilibria needs them.
- **Streaming / incremental writes** for HARs larger than RAM. Current GTAP
  HARs are well under 100 MB; full in-memory emission is fine.
- **CLI front-end** for writing HARs (analogous to `har_to_gdx.py`).
  DataFrame export from Python is the entry point.
- **Compression or non-standard variants.** GEMPACK's standard little-endian
  binary format only.

## 10. Public API summary

After v1, `equilibria.babel.har` exposes:

```python
# Reading (unchanged behavior; internals refactored to use wire.py)
from equilibria.babel.har import (
    HeaderArray,
    read_har,
    read_header_array,
    get_header_names,
)

# Writing (new)
from equilibria.babel.har import (
    write_har,        # low-level: dict[name, HeaderArray] -> file
    HarWriter,        # builder: add_set / add_array / add_dataframe / close
)
```

## 11. Acceptance criteria

v1 ships when all of the following hold:

1. All seven test layers (L1–L7) pass.
2. L4 either passes for all six GTAP fixtures, or each failing fixture has a
   committed `.diff` file and an `xfail` marker explaining the GEMPACK
   non-determinism.
3. `tests/babel/har/test_reader.py` remains green (no behavioral regression
   from the `wire.py` extraction).
4. README in `src/equilibria/babel/har/` documents the writer and reiterates
   the clean-room provenance.
5. NOTICE file updated if needed to reflect the writer's clean-room status.
6. No `harpy3` / `harpy` source has been read during implementation; the
   only contact with harpy3 is via `scripts/har/oracle_check.py` in a
   sandbox venv.
