"""What the old GTAP drivers gave, for the tests that compared them to the core.

``tests/golden/pr4_old_route_main.json.gz`` was recorded on main (2c0d882a) by the
PR 4 recorder (``scripts/dev/record_pr4_old_route.py``, committed in 6a6fd1bf and
deleted with the old drivers by PR 4 of gtap-on-core); two runs were identical. The
seven ``on_core_*`` cases (test_gtap_on_core_solve, commit f50dad93) were recorded
later by the same recorder on a checkout of 2c0d882a, two identical runs again. Each case is
the old side of one old-vs-core equality test. The core is compared against it with
a tolerance (``tests/_golden.py``): another CPU moves the last bits.
"""

from __future__ import annotations

import gzip
import json
import math
from functools import cache
from pathlib import Path
from typing import Any

from tests._golden import mismatches

GOLDEN = Path(__file__).parent / "golden" / "pr4_old_route_main.json.gz"
# a full solve: ~1e-11 relative on another CPU, a level near zero ~1e-11 absolute
REL, ABS = 1e-9, 1e-12
# walras is the solve's residual, zero in exact arithmetic: the Linux CI gives
# 3e-11 where the Mac recording has 3e-12 (2e-9 vs 7e-10 after a Johansen step)
WALRAS_ABS = 1e-8
_WALRAS = ("('walras',", "walrasbase", "walrascheck", "walrasshock")


@cache
def _all() -> dict:
    with gzip.open(GOLDEN, "rt", encoding="utf-8") as f:
        return json.load(f)


def case(name: str) -> dict:
    return _all()[name]


def pyomo_levels(m: Any) -> dict[str, float | None]:
    """Every Var level of a Pyomo model, keyed as the recorder keyed them."""
    from pyomo.environ import Var

    out: dict[str, float | None] = {}
    for v in m.component_objects(Var):
        for k in v:
            x = v[k].value
            out[repr((v.name, k))] = None if x is None else float(x)
    return out


def _num(x: Any) -> float:
    return math.nan if x is None else float(x)


def _is_walras(key: Any) -> bool:
    """A walras cell: a Pyomo level key (``"('walras', 'shock')"`` or
    ``"walrasshock"``) or an array cell ``("walras", field, position)``."""
    if isinstance(key, tuple):
        return bool(key) and key[0] == "walras"
    return str(key).startswith(_WALRAS)


def level_mismatches(got: dict, want: dict, *, rel: float = REL) -> list[str]:
    """Cells off the golden; NaN (no value) equals NaN, a key on one side fails.
    ``rel`` widens the band for a caller whose solve amplifies the CPU's noise (the
    Johansen step); walras is held to an absolute floor (``WALRAS_ABS``)."""
    if got.keys() != want.keys():
        return [f"keys: {sorted(got.keys() ^ want.keys())[:5]}"]
    g = {k: _num(v) for k, v in got.items() if not math.isnan(_num(v))}
    w = {k: _num(v) for k, v in want.items() if not math.isnan(_num(v))}
    nan_side = [k for k in got if math.isnan(_num(got[k])) != math.isnan(_num(want[k]))]
    walras = {k for k in g if _is_walras(k)}
    levels = mismatches(
        {k: v for k, v in g.items() if k not in walras},
        {k: v for k, v in w.items() if k not in walras},
        rel=rel,
        abs_tol=ABS,
    )
    residual = [
        f"{k}: {g[k]} != {w.get(k)}"
        for k in sorted(walras)
        if k not in w or abs(g[k] - w[k]) > WALRAS_ABS
    ]
    return [f"{k}: value on one side only" for k in nan_side] + levels + residual


def mp_mismatches(mp: Any, want: dict) -> list[str]:
    """``MPArrays`` against a recorded one: keys, fixed and active exactly; value,
    lb and ub within the tolerance (NaN equal to NaN)."""
    out: list[str] = []
    if sorted(mp.vars) != sorted(want["vars"]) or sorted(mp.rows) != sorted(
        want["rows"]
    ):
        return ["var or row names differ"]
    for name, v in mp.vars.items():
        w = want["vars"][name]
        if [repr(k) for k in v.keys] != w["keys"]:
            out.append(f"{name}: keys differ")
            continue
        if [bool(x) for x in v.fixed.tolist()] != w["fixed"]:
            out.append(f"{name}: fixed differs")
        for f in ("value", "lb", "ub"):
            got = dict(enumerate(getattr(v, f).tolist()))
            out += level_mismatches(
                {(name, f, i): x for i, x in got.items()},
                {(name, f, i): x for i, x in enumerate(w[f])},
            )
    for name, r in mp.rows.items():
        w = want["rows"][name]
        if [repr(k) for k in r.keys] != w["keys"] or [bool(x) for x in r.active] != w[
            "active"
        ]:
            out.append(f"{name}: rows differ")
    return out
