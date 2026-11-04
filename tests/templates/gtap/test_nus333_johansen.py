"""Johansen: the shock as one linear step from check, as the Burfisher book solves six exercises.

GEMPACK's Johansen method solves the model linearized at the starting point once:
every variable moves by its first-order response to the shock, with no further
iteration. Burfisher's ``.EXP`` files use it for TBL45A/B/C, TBL64, ME4A and ME4B,
so the book's published numbers for them are Johansen results.

``solve_multiperiod(..., shock_method="johansen")`` gives that solution: the
derivative of the solution path at the check point, times the full shock. Each
variable then sits at ``check × (1 + %change)``, or at ``check + change`` when its
level is not positive.

Oracle: the book's GEMPACK Johansen runs (``tests/fixtures/gtap7_gempack/burfisher``),
compared with the comparator's linear mode. In a Johansen solution a product is
the sum of the % changes of its factors, not the product of the new levels.

Government demand (``qgd``, ``qgm``) is left out: it is the largest gap with
GEMPACK in every comparison, nonlinear too, and it already shows in the slope at
the benchmark (#201).
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest
from tests.templates.gtap._nus333 import ROOT, nus333_params

JOHANSEN_IN_THE_BOOK = ("TBL45A", "TBL45B", "TBL45C", "TBL64", "ME4A", "ME4B")
GOVERNMENT_DEMAND = {"qgd", "qgm"}  # #201


def _modules() -> tuple[Any, Any]:
    nus333_params()  # SKIP without the dataset
    sys.path[:0] = [str(ROOT / "scripts" / "gtap")]
    from equilibria._local_refs import path_capi_src

    sys.path.insert(0, str(path_capi_src()))
    cb: Any = importlib.import_module("compare_burfisher_gempack")
    return cb, cb.rb


def _solve(exp: str, shock_method: str, scale: float = 1.0):
    cb, rb = _modules()
    from equilibria._local_refs import nus333_dir

    prm, rows = rb.EXERCISES[exp]
    rows = [(name, idx, pct * scale) for name, idx, pct in rows]
    return rb.solve_exercise(exp, nus333_dir(), shock_method=shock_method, rows=rows)


def _report(exp: str, m, p, *, linear: bool):
    cb, _ = _modules()
    from equilibria.blocks.gtap._derived_params import xscale_data

    return cb.run(
        exp, cb.cg.model_levels(m), dict(xscale_data(p, p.sets)), linear=linear
    )


@pytest.mark.integration
@pytest.mark.parametrize("exp", JOHANSEN_IN_THE_BOOK)
def test_johansen_reproduces_the_books_result(exp: str) -> None:
    m, p, codes = _solve(exp, "johansen")
    assert codes == {"base": 1, "check": 1, "shock": 1}
    r = _report(exp, m, p, linear=True)
    assert r.overall["cells"] == 414
    bad = [c for c in r.cells if not c.passed and c.variable not in GOVERNMENT_DEMAND]
    assert not bad, bad[:5]


@pytest.mark.integration
def test_the_nonlinear_solution_is_not_the_books_johansen() -> None:
    """The check above can fail: TBL45A solved exactly misses the book's numbers."""
    m, p, codes = _solve("TBL45A", "nonlinear")
    assert codes == {"base": 1, "check": 1, "shock": 1}
    r = _report("TBL45A", m, p, linear=False)
    bad = [c for c in r.cells if not c.passed and c.variable not in GOVERNMENT_DEMAND]
    assert len(bad) > 10


@pytest.mark.integration
def test_johansen_is_linear_in_the_shock() -> None:
    """Half the shock gives half of every % change, which no nonlinear solve does."""
    from pyomo.environ import Var, value

    def pct(m) -> dict:
        out = {}
        for v in m.component_objects(Var):
            for i in v:
                if isinstance(i, tuple) and i[-1] == "shock":
                    x0 = value(v[(*i[:-1], "check")])
                    if x0 > 1e-6:
                        out[v.name, i] = value(v[i]) / x0 - 1
        return out

    full, half = (
        pct(_solve("TBL55", "johansen")[0]),
        pct(_solve("TBL55", "johansen", 0.5)[0]),
    )
    moved = [k for k in full if abs(full[k]) > 1e-4]
    assert len(moved) > 100
    for k in moved:
        assert half[k] == pytest.approx(full[k] / 2, rel=1e-4, abs=1e-9), k


def test_johansen_refuses_shock_steps() -> None:
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    with pytest.raises(ValueError, match="johansen"):
        solve_multiperiod(
            None, None, None, shock_method="johansen", shock_steps=(0.5, 1.0)
        )


def test_shock_method_is_checked() -> None:
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    with pytest.raises(ValueError, match="shock_method"):
        solve_multiperiod(None, None, None, shock_method="gragg")
