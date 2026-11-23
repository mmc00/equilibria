"""``run_burfisher.solve_exercise``: a Burfisher exercise solves on the core
(``Exercise``) to the point the old Pyomo driver reached, recorded on main before
PR 4 of gtap-on-core deleted it (``tests/_old_route_golden.py``)."""

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
def test_solve_exercise_gives_what_the_old_driver_gave(shock_method: str) -> None:
    from tests._old_route_golden import case, level_mismatches, pyomo_levels

    from equilibria._local_refs import nus333_dir

    rb = _rb()
    exp = "TBL45A" if shock_method == "johansen" else "TBL46A"
    m, _, codes = rb.solve_exercise(exp, nus333_dir(), shock_method=shock_method)
    want = case(f"burfisher_{shock_method}")
    assert codes == want["codes"] == {"base": 1, "check": 1, "shock": 1}
    bad = level_mismatches(pyomo_levels(m), want["levels"])
    assert not bad, f"{len(bad)} cells differ, first: {bad[:10]}"
