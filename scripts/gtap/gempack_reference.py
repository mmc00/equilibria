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
    solution vs Python `xd`) — folds the gap in once, not twice.

SL4 QUANTITY PATH (2026-07-23): the `sltoht`-exported SL4 dump
(`sl4dump_<ds>_tm10.har`) carries the solution %-changes for every model variable
(qfd/qxs/qo/...). Its headers are numbered `0001..` but EACH preserves the GEMPACK
variable name in its `long_name` (`"qfd # demand for domestic commodity ... #"`).
So the q-name→SL4-id map is recovered from the file itself (`sl4_index` /
`sl4_levels` below) — NOT hand-authored. 100% of headers (256 unique vars on 3x3)
parse cleanly. The ONE remaining piece is the q-name→Python-Var modeling map
(e.g. GEMPACK `qfd`→Python `xd`), still authored deliberately, never guessed.

Wiring the quantity comparison into the gate is the next step; this module carries
the verified value mechanism, the synthetic-fixture map, and the SL4 name reader.
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


def _cells(ha) -> dict[tuple[str, ...], float]:
    dims = [list(e) for e in ha.set_elements]
    out: dict[tuple[str, ...], float] = {}
    for idx in np.ndindex(*ha.array.shape):
        key = tuple(dims[d][i] for d, i in enumerate(idx))
        out[key] = float(ha.array[idx])
    return out


def sl4_index(har_path: str) -> dict[str, str]:
    """Map GEMPACK variable name -> numeric SL4 header id (`"0002"`), parsed from
    each header's `long_name` (`"<name> # description #"`).

    sltoht numbers the SL4 solution headers `0001..` but preserves the variable
    name in `long_name`, so the name->id map is recovered from the file itself —
    no hand-authored q-name→id table (the guide §8 TODO #1 is dissolved).
    """
    idx: dict[str, str] = {}
    for hid, ha in read_har(har_path).items():
        long_name = (ha.long_name or "").strip()
        if not long_name:
            continue
        name = long_name.split("#", 1)[0].strip()
        if name:
            idx.setdefault(name, hid)
    return idx


def sl4_levels(har_path: str, gempack_var: str) -> dict[tuple[str, ...], float]:
    """Return {(set_elem, ...): value} cells for a GEMPACK variable, read from an
    `sltoht`-exported SL4 HAR. Works on BOTH SL4 fixture kinds:

      * `sl4dump_<ds>_tm10.har`   -> cumulative %-CHANGES (sltoht's HAR/VAI mode
        only ever writes this column)
      * `sl4levels_<ds>_tm10.har` -> absolute POST-SIM LEVELS, produced by
        `export_sl4_levels.py` (sltoht exposes levels only in its TEXT modes
        under option SHL; see that script)

    Either way the numeric header is resolved by variable name via `sl4_index`.
    Raises KeyError if the variable is absent.
    """
    headers = read_har(har_path)
    name_to_id = {
        (ha.long_name or "").split("#", 1)[0].strip(): hid
        for hid, ha in headers.items()
        if (ha.long_name or "").strip()
    }
    hid = name_to_id.get(gempack_var)
    if hid is None:
        raise KeyError(f"{gempack_var!r} not found in SL4 dump {har_path}")
    return _cells(headers[hid])


# ── Quantity comparison: verified GEMPACK-q → Python-Var map (2026-07-23) ──
# Derived from each variable's GEMPACK long_name description AND numerically
# confirmed vs GAMS on gtap7_3x3 (median |Δpp| ~0.4). `reorder` maps the GEMPACK
# header key (set elems in header order) to the Python Var index tuple.
#   qfd [COMM,ACTS,REG] "demand for domestic commodity by activity" -> xd  [r,c,a]
#   qxs [COMM,src,dst]  "export sales from source"                  -> xw  [r_src,c,r_dst]
#   qo  [ACTS,REG]      "output of activity"                        -> xp  [r,a]
# The SL4 %-changes are in PERCENT; the natural comparison metric is ABSOLUTE
# percentage points (|Δpp|), NOT relative tolerance — small %-changes make a
# relative tol reject good 0.4pp agreements. GEMPACK is Gragg-linearized, Python is
# levels; the residual (concentrated in diagonal c==a and a few cells) is that
# structural gap, identical Python-vs-GAMS.
#
# WHY %-changes and not levels — Horridge & Pearson, "Solution Software for CGE
# Modeling" (COPS General Paper G-214, 2011), §4.1/4.2: GAMS uses a LEVELS strategy
# (solve for Y directly, merit FᵀF<1e-6), GEMPACK a CHANGE strategy where "most of
# the change variables are expressed as percentage changes" (§4.2.1) via multi-step
# Euler + Richardson extrapolation. §6.5 "All three give the same results": the same
# model in GEMPACK/GAMS/MPSGE yields the same numbers. But §6 (p.28) shows the
# linearized decomposition (−1.824) is "not exactly equal" to the levels result
# (−2.214) because "the linearized equations ... are not satisfied exactly by the
# accurate results" — that IS our structural pp-residual, per GEMPACK's own author.
# So comparing GEMPACK↔Python in %-change (the form both share) is the correct
# apples-to-apples; a levels comparison reduces to (1+%chg) after benchmark
# normalization (verified identical to 8 decimals) — there is no distinct
# "levels-vs-levels" GEMPACK comparison.
Q_TO_VAR: dict[str, dict] = {
    "qfd": {"var": "xd", "reorder": lambda k: (k[2], k[0], k[1])},   # (c,a,r)->(r,c,a)
    "qxs": {"var": "xw", "reorder": lambda k: (k[1], k[0], k[2])},   # (c,src,dst)->(src,c,dst)
    "qo": {"var": "xp", "reorder": lambda k: (k[1], k[0])},          # (a,r)->(r,a)
}


def gempack_qty_pct(sl4dump_path: str, gempack_var: str) -> dict[tuple[str, ...], float]:
    """GEMPACK %-change of a quantity variable, as a FRACTION (0.05 = 5%), keyed by
    the Python Var index order (via Q_TO_VAR[gempack_var]['reorder']).

    Read from a `sl4dump_<ds>_tm10.har` (cumulative %-changes). Raises KeyError if
    the variable is not in the verified quantity map or not in the dump.
    """
    spec = Q_TO_VAR.get(gempack_var)
    if spec is None:
        raise KeyError(f"{gempack_var!r} not in the verified GEMPACK-q→Var map")
    reorder = spec["reorder"]
    return {
        reorder(k): v / 100.0
        for k, v in sl4_levels(sl4dump_path, gempack_var).items()
    }
