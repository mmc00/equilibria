"""Johansen: the shock as one linear step from check, as the Burfisher book solves six exercises.

GEMPACK's Johansen method solves the model linearized at the starting point once:
every variable moves by its first-order response to the shock, with no further
iteration. Burfisher's ``.EXP`` files use it for TBL45A/B/C, TBL64, ME4A and ME4B,
so the book's published numbers for them are Johansen results.

``gtap_static(shock_method="johansen")`` gives that solution: the
derivative of the solution path at the check point, times the full shock. Each
variable then sits at ``check × (1 + %change)``, or at ``check + change`` when its
level is not positive.

Oracle: the book's GEMPACK Johansen runs (``data/gtap7/nus333/gempack_burfisher``),
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
from tests.templates.gtap._routes import ROUTES

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
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("exp", JOHANSEN_IN_THE_BOOK)
def test_johansen_reproduces_the_books_result(exp: str, route: str) -> None:
    m, p, codes = _solve(exp, "johansen")
    assert codes == {"base": 1, "check": 1, "shock": 1}
    r = _report(exp, m, p, linear=True)
    assert r.overall["cells"] == 414
    bad = [c for c in r.cells if not c.passed and c.variable not in GOVERNMENT_DEMAND]
    assert not bad, bad[:5]


@pytest.mark.integration
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("exp", JOHANSEN_IN_THE_BOOK)
def test_johansen_moves_the_global_investment_price(exp: str, route: str) -> None:
    """A cell indexed by the period alone gets its linear step too.

    ``pigbl`` is GEMPACK's ``pcgdswld`` (both the net-investment-weighted price
    of investment). The comparator above has no row for it, so the book's
    ``.sl4`` is read directly; GEMPACK moves it 0.75 to 1.88 % in these runs.
    """
    from equilibria.babel.har.sl4 import read_sl4_changes
    from equilibria.sources import reference

    m, _, codes = _solve(exp, "johansen")
    assert codes == {"base": 1, "check": 1, "shock": 1}
    pct = 100.0 * (m.pigbl["shock"].value / m.pigbl["check"].value - 1.0)
    run = reference("nus333", "gempack", experiment=exp.lower(), method="book")
    sl4 = read_sl4_changes(run / "solution.sl4")
    # Measured 2026-11-17: within 1e-4 pp in all six (before the fix: +0.03 %)
    assert pct == pytest.approx(float(sl4.changes["pcgdswld"].values), abs=1e-3)


@pytest.mark.integration
@pytest.mark.parametrize("route", ROUTES)
def test_the_nonlinear_solution_is_not_the_books_johansen(route: str) -> None:
    """The check above can fail: TBL45A solved exactly misses the book's numbers."""
    m, p, codes = _solve("TBL45A", "nonlinear")
    assert codes == {"base": 1, "check": 1, "shock": 1}
    r = _report("TBL45A", m, p, linear=False)
    bad = [c for c in r.cells if not c.passed and c.variable not in GOVERNMENT_DEMAND]
    assert len(bad) > 10


@pytest.mark.integration
@pytest.mark.parametrize("route", ROUTES)
def test_johansen_is_linear_in_the_shock(route: str) -> None:
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
