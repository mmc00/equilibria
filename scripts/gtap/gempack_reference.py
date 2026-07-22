"""Read a RunGTAP updated.har (post-shock levels) and expose Var cells shaped
like gams_levels, so the parity measurement loop is reference-agnostic.

VAR_TO_HEADER is the cell-by-cell mapping — the OPEN-RISK piece from the spec.
It starts minimal (only vars with a clean levels correspondence) and grows as a
real updated.har's header inventory is inspected. Vars absent here are
aggregate-only and raise KeyError (never a fabricated match).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np  # noqa: E402

from equilibria.babel.har.reader import read_har  # noqa: E402

# Pyomo Var basename -> RunGTAP HAR header. Seeded minimally; extend against a
# real updated.har header list (spec §4 step 3).
VAR_TO_HEADER: dict[str, str] = {
    "qfd": "VDFB",  # firm domestic purchases
}


def gempack_levels(har_path: str, var_basename: str) -> dict[tuple[str, ...], float]:
    """Return {(set_elem, ...): value} cells for a Var, read from updated.har.

    Raises KeyError if the Var has no cell-by-cell GEMPACK header (aggregate-only
    or not yet mapped) — never a fabricated match.
    """
    header = VAR_TO_HEADER.get(var_basename)
    if header is None:
        raise KeyError(
            f"{var_basename!r} has no cell-by-cell GEMPACK header "
            f"(aggregate-only or not yet mapped in VAR_TO_HEADER)"
        )
    ha = read_har(har_path)[header]
    dims = [list(e) for e in ha.set_elements]
    out: dict[tuple[str, ...], float] = {}
    for idx in np.ndindex(*ha.array.shape):
        key = tuple(dims[d][i] for d, i in enumerate(idx))
        out[key] = float(ha.array[idx])
    return out
