"""HAR → GDX bridge.

Reads one or more GEMPACK .har files and emits one or three GDX files using
the native writer.

- `write_har_to_gdx(hars, out)` — single GDX with all headers as sets/params.
- nus333's three-file bundle for getData.gms lives with the GAMS model:
  data/gtap7/gams/tools/nus333_bundle.py.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np

from equilibria.babel.gdx.symbols import Parameter, Set, SymbolBase
from equilibria.babel.gdx.writer import write_gdx
from equilibria.babel.har.reader import read_har
from equilibria.babel.har.symbols import HeaderArray

# ── single-file API (kept stable for prior callers) ──────────────────────────


def har_to_symbols(har_paths: list[str | Path]) -> list[SymbolBase]:
    headers: dict[str, HeaderArray] = {}
    for p in har_paths:
        for name, h in read_har(str(p)).items():
            headers[name] = h

    sets: list[Set] = []
    params: list[Parameter] = []

    for name, h in headers.items():
        if _is_set_header(h):
            sets.append(_set_from_header(h))
        elif h.array.ndim >= 1 and not h.set_names:
            continue
        else:
            params.append(_parameter_from_header(h))

    return [*sets, *params]


def write_har_to_gdx(har_paths: list[str | Path], out_path: str | Path) -> None:
    write_gdx(out_path, har_to_symbols(har_paths))


# ── shared helpers ───────────────────────────────────────────────────────────


def _is_set_header(h: HeaderArray) -> bool:
    return (
        h.array.ndim == 1 and h.array.dtype.kind in ("U", "S", "O") and not h.set_names
    )


def _set_from_header(h: HeaderArray) -> Set:
    elements = [[str(e).strip()] for e in h.array.tolist()]
    return Set(
        name=h.name,
        dimensions=1,
        description=h.long_name.strip(),
        domain=["*"],
        elements=elements,
    )


def _parameter_from_header(h: HeaderArray) -> Parameter:
    if h.array.ndim == 0:
        return Parameter(
            name=h.name,
            dimensions=0,
            description=h.long_name.strip(),
            domain=[],
            records=[([], float(h.array.item()))],
        )
    return Parameter(
        name=h.name,
        dimensions=h.array.ndim,
        description=h.long_name.strip(),
        domain=list(h.set_names),
        records=_records_from_array(h),
    )


def _records_from_array(
    h: HeaderArray,
    dim_prefixes: list[str] | None = None,
) -> list[tuple[list[str], float]]:
    """Materialize all non-zero records. Sparse: GAMS treats absent keys as 0.

    If dim_prefixes is given, prefix the label at each dimension. Used so that
    acts/comm labels don't collide in shared UEL space (a_AGR vs c_AGR).
    """
    arr = np.asarray(h.array)
    records: list[tuple[list[str], float]] = []
    pref = dim_prefixes or [""] * arr.ndim
    for idx in product(*(range(s) for s in arr.shape)):
        val = float(arr[idx])
        if val == 0.0:
            continue
        keys = [
            f"{pref[d]}{str(h.set_elements[d][i]).strip()}" for d, i in enumerate(idx)
        ]
        records.append((keys, val))
    return records
