"""Read a RunGTAP updated.har (post-shock levels) and expose Var cells shaped
like gams_levels, so the parity measurement loop is reference-agnostic.

VAR_TO_HEADER is the cell-by-cell mapping — the OPEN-RISK piece from the spec.
It starts minimal (only vars with a clean levels correspondence) and grows as a
real updated.har's header inventory is inspected. Vars absent here are
aggregate-only and raise KeyError (never a fabricated match).

FINDINGS (2026-07-23, empirical, on gtap7_3x3 — see the guide §8):
  * The updated.har headers are all VALUE flows ($ SAM), not quantities. There is
    NO qty header; `qfd` below is a mechanism placeholder for the unit tests
    against the synthetic fixture, NOT a verified economic mapping.
  * The correct VALUE reconstruction is  VDFB[c,a,r] = pd[r,c] * xd[r,c,a]
    (GTAP identity xd = vdfb/pd, pd=1 at benchmark). Compared as a post/base ratio
    (Python normalizes benchmark→1; GEMPACK updated.har is absolute $millions, so
    divide by basedata.har VDFB), this matches GEMPACK 66.67% @ 1% tol on 3x3.
  * That 33% gap is STRUCTURAL (Gragg-linearized vs levels), NOT a Python defect:
    GAMS-levels reconstructed identically gives the SAME 66.67% on the SAME cells.
  * A cleaner comparison is QUANTITY-vs-QUANTITY (GEMPACK `qfd` from the SL4
    solution vs Python `xd`) — folds the gap in once, not twice. That needs the
    SL4→HAR export (guide §8) plus a q-name→Var map, both TODO.

Wiring the real value/quantity comparison into the gate is the next-session work;
today this module only carries the verified mechanism + the synthetic-fixture map.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np  # noqa: E402

from equilibria.babel.har.reader import read_har  # noqa: E402

# Pyomo Var basename -> RunGTAP HAR header. `qfd`→`VDFB` is the SYNTHETIC-FIXTURE
# mechanism entry (the unit test exercises read→cells with it); it is NOT the real
# economic mapping (VDFB is a value = pd*xd, not the qfd quantity — see the module
# docstring). Real value/quantity entries land next session, against a real header
# inventory, never guessed.
VAR_TO_HEADER: dict[str, str] = {
    "qfd": "VDFB",  # SYNTHETIC-FIXTURE mechanism entry only — see docstring
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
