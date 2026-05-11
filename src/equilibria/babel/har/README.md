# equilibria.babel.har

Pure-Python reader for GEMPACK **HAR** and **`.prm`** files. No third-party
HAR library required — `harpy3` / `harpy` is **NOT** a dependency.

## Origin and license

This reader is a **clean-room reimplementation** developed by inspecting the
on-disk byte layout of HAR files produced by GEMPACK / RunGTAP and by reading
publicly available descriptions of the format. It was written without viewing,
copying, translating, or otherwise deriving from:

- `harpy3` / `harpy` (GPLv3),
- GEMPACK source distributions (proprietary), or
- any other HAR implementation.

The interoperability constants used here (Fortran unformatted record framing,
the `1CFULL` / `REFULL` / `RESPSE` / `2IFULL` type tokens, etc.) are the wire
format itself — not copyrightable expression.

Distributed under the MIT License, same as the rest of `equilibria`. See the
top-level `NOTICE` file for the full statement.

## Supported types

| Token    | Meaning                        | Notes                                     |
|----------|--------------------------------|-------------------------------------------|
| `1CFULL` | 1-D character set              | Multi-record element lists are handled   |
| `REFULL` | Real dense N-D array           | Repeated set names (e.g. REG×REG) OK     |
| `RESPSE` | Real sparse N-D array          | 1-based Fortran flat indices             |
| `2IFULL` | 2-D integer dense array        |                                           |

Validated header-by-header against `harpy3` on the full GTAP NUS333 and 9×10
datasets (182 headers total, exact match).

## Usage

```python
from equilibria.babel.har import read_har

data = read_har("basedata.har")          # dict[name -> HeaderArray]
ha = data["VDFB"]
print(ha.long_name, ha.array.shape, ha.set_names)
```
