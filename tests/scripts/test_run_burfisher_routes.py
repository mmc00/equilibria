"""``run_burfisher.solve_exercise(route=...)``: a Burfisher exercise solves to the
same point through the old driver and through the core (``Exercise``)."""

from __future__ import annotations

import importlib
import math
import sys
from typing import Any

import pytest
from tests.templates.gtap._nus333 import ROOT, nus333_params

pytestmark = pytest.mark.integration


def _rb() -> Any:
    nus333_params()  # SKIP without the dataset
    sys.path[:0] = [str(ROOT / "scripts" / "gtap")]
    from equilibria._local_refs import path_capi_src

    sys.path.insert(0, str(path_capi_src()))
    return importlib.import_module("run_burfisher")


def _levels(m: Any) -> dict:
    from pyomo.environ import Var

    out = {}
    for v in m.component_objects(Var):
        for k in v:
            x = v[k].value
            out[v.name, k] = math.nan if x is None else float(x)
    return out


@pytest.mark.parametrize("shock_method", ["nonlinear", "johansen"])
def test_solve_exercise_core_route_matches_old(shock_method: str) -> None:
    from equilibria._local_refs import nus333_dir

    rb = _rb()
    exp = "TBL45A" if shock_method == "johansen" else "TBL46A"
    m_old, _, c_old = rb.solve_exercise(exp, nus333_dir(), shock_method=shock_method)
    m_new, _, c_new = rb.solve_exercise(
        exp, nus333_dir(), shock_method=shock_method, route="core"
    )
    assert c_old == c_new == {"base": 1, "check": 1, "shock": 1}
    a, b = _levels(m_old), _levels(m_new)
    assert a.keys() == b.keys()
    bad = [
        (k, a[k], b[k])
        for k in a
        if not (a[k] == b[k] or (math.isnan(a[k]) and math.isnan(b[k])))
    ]
    assert not bad, f"{len(bad)} cells differ, first: {bad[:10]}"


def test_solve_exercise_refuses_an_unknown_route() -> None:
    from equilibria._local_refs import nus333_dir

    with pytest.raises(ValueError, match="route"):
        _rb().solve_exercise("TBL46A", nus333_dir(), route="new")
