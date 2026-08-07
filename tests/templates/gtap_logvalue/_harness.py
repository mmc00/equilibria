"""Shared helpers for the log-value block gates.

`build_one_block` composes a single log-value block through the neutral backend
(Model.add_block → PyomoBackend) against the gtap7_3x3 Julia calibrated point, so a
per-block test can assert its equations built. `port_eq_rows` compares a block's
equation ROW/COL structure against the same equations in the port monolith.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from equilibria.backends.pyomo_backend import PyomoBackend
from equilibria.core.sets import Set as ESet
from equilibria.model import Model
from equilibria.templates.gtap_julia.calibration import load_calibrated

ROOT = Path(__file__).resolve().parents[3]
FIX = ROOT / "tests" / "fixtures" / "gtap_logvalue" / "julia_3x3_calibrated.csv"

# port set key -> block-model set name
_SETMAP = {"reg": "r", "comm": "i", "acts": "a", "endw": "f", "marg": "marg"}


def load_sol(dataset: str = "gtap7_3x3") -> dict[str, Any]:
    """The Julia calibrated point as {name: {idx: val}} plus a 'sets' sub-dict."""
    sol = load_calibrated(FIX)
    # SET_* rows carry string members (name,ordinal,member) — load_calibrated drops
    # them (member isn't a float), so read them straight from the CSV.
    sets: dict[str, list[tuple[int, str]]] = {}
    for line in FIX.read_text().splitlines():
        if not line.startswith("SET_"):
            continue
        parts = line.split(",")
        key = parts[0][4:]
        sets.setdefault(key, []).append((int(parts[1]), parts[2]))
    sol["sets"] = {k: [m for _, m in sorted(v)] for k, v in sets.items()}
    return sol


def _block_sets(sol: dict[str, Any]) -> dict[str, list[str]]:
    s = sol["sets"]
    return {
        "r": list(s["reg"]),
        "i": list(s["comm"]),
        "a": list(s["acts"]),
        "f": list(s["endw"]),
        "fm": list(s.get("endwm", [])),
        "fs": list(s.get("endws", [])),
        "ff": list(s.get("endwf", [])),
        "fms": list(s.get("endwms", [])),
        "marg": list(s.get("marg", [])),
        "rp": list(s["reg"]),
    }


def build_one_block(block_cls, dataset: str = "gtap7_3x3", deps=()):
    """Compose a block (plus any dependency blocks that own vars it consumes) through
    PyomoBackend; return the Pyomo ConcreteModel. A block that references a var owned
    by an earlier block in GTAP_LOGVALUE_BLOCK_ORDER must pass those as `deps`."""
    sol = load_sol(dataset)
    setmap = _block_sets(sol)
    model = Model(name="gtap_logvalue_one")
    for name, elems in setmap.items():
        model.add_set(ESet(name=name, elements=tuple(elems)))
    for dep_cls in deps:
        model.add_block(dep_cls(sol=sol))
    model.add_block(block_cls(sol=sol))
    backend = PyomoBackend()
    backend.build(model)
    return backend.pyomo_model
