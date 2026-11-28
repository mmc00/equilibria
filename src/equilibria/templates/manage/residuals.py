"""The block side of the residual harness: rows of a built Pyomo model, no solve.

A block-group PR checks its equations at the GAMS points before anything is
solved (design, "Parity test"): the levels of a GAMS point are put into the
Pyomo model, and every row of the group must be near zero there. GAMS's side of
the same check is ``convert.ConvertRows``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pyomo.environ as pyo

Rows = dict[tuple[Any, ...], float]


def block_residuals(pyomo_model: Any, families: Iterable[str]) -> dict[str, Rows]:
    """``{family: {index: body - bound}}`` of the named equation families at the
    values the model holds. A family is the equation's name (``eq_va``); the
    backend builds it as the Constraint ``<name>_con``."""
    out: dict[str, Rows] = {}
    for name in families:
        con = pyomo_model.find_component(f"{name}_con")
        if con is None:
            raise KeyError(f"no equation family {name!r} in the model")
        rows: Rows = {}
        for index, c in con.items():
            key = index if isinstance(index, tuple) else (index,)
            bound = c.upper if c.upper is not None else c.lower
            rows[key] = float(pyo.value(c.body) - pyo.value(bound))
        out[name] = rows
    return out
