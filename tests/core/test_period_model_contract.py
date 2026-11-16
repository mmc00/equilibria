"""The PeriodModel contract: per-variable and per-row operations.

Every backend runs the same tests: ``MAKERS`` lists one factory per backend, each
building the same small model (periods ``a`` and ``b``):

* ``x[i]`` (i in i1, i2): ``NonNegativeReals``, lb 0.5, level 1.0
* ``y``: scalar, level 2.0
* ``z[i]``: no level
* rows ``e[i]`` (``x[i] == y``) and ``g`` (``y == 2``), one copy per period
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import pyomo.environ as pyo
import pytest

from equilibria.backends.pyomo_periods import PyomoPeriodModel, reflect

I1, I2 = ("i1",), ("i2",)


def _pyomo() -> PyomoPeriodModel:
    sp: Any = pyo.ConcreteModel()
    sp.I = pyo.Set(initialize=["i1", "i2"])
    sp.x = pyo.Var(
        sp.I, domain=pyo.NonNegativeReals, bounds=(0.5, None), initialize=1.0
    )
    sp.y = pyo.Var(initialize=2.0)
    sp.z = pyo.Var(sp.I)
    sp.e = pyo.Constraint(sp.I, rule=lambda m, i: m.x[i] == m.y)
    sp.g = pyo.Constraint(expr=sp.y == 2)
    return PyomoPeriodModel(reflect(sp, ("a", "b")))


def _family() -> Any:
    """The same model as ``MPArrays`` (keys with the period last)."""
    import numpy as np

    from equilibria.family.mp_arrays import MPArrays, RowArr, VarArr
    from equilibria.family.periods import FamilyPeriodModel

    def var(cells: list[tuple], level: float, lb: float) -> VarArr:
        keys = [(*c, t) for t in ("a", "b") for c in cells]
        n = len(keys)
        return VarArr(
            keys,
            np.full(n, level),
            np.zeros(n, dtype=bool),
            np.full(n, lb),
            np.full(n, np.inf),
        )

    def row(cells: list[tuple]) -> RowArr:
        keys = [(*c, t) for t in ("a", "b") for c in cells]
        return RowArr(keys, np.ones(len(keys), dtype=bool))

    mp = MPArrays(
        vars={
            "x": var([I1, I2], 1.0, 0.5),
            "y": var([()], 2.0, -np.inf),
            "z": var([I1, I2], np.nan, -np.inf),
        },
        rows={"e": row([I1, I2]), "g": row([()])},
    )
    return FamilyPeriodModel(mp, periods=("a", "b"))


MAKERS: list[Callable[[], Any]] = [_pyomo, _family]


@pytest.fixture(params=MAKERS, ids=lambda f: f.__name__.strip("_"))
def m(request: pytest.FixtureRequest) -> Any:
    return request.param()


def test_names(m: Any) -> None:
    assert set(m.var_names()) == {"x", "y", "z"}
    assert set(m.row_names()) == {"e", "g"}
    assert m.has_var("x") and not m.has_var("w")
    assert m.has_row("g") and not m.has_row("x")


def test_var_cells_are_one_period(m: Any) -> None:
    assert m.var_cells("x", "a") == (I1, I2)
    assert m.var_cells("y", "b") == ((),)


def test_row_cells_are_one_period(m: Any) -> None:
    assert m.row_cells("e", "b") == (I1, I2)
    assert m.row_cells("g", "a") == ((),)


def test_values_nan_without_level(m: Any) -> None:
    assert m.values("x", "a") == {I1: 1.0, I2: 1.0}
    assert all(math.isnan(v) for v in m.values("z", "a").values())


def test_set_values_one_period(m: Any) -> None:
    m.set_values("x", "a", {I1: 3.0})
    assert m.values("x", "a") == {I1: 3.0, I2: 1.0}
    assert m.values("x", "b") == {I1: 1.0, I2: 1.0}


def test_set_values_accepts_nan(m: Any) -> None:
    m.set_values("y", "a", {(): math.nan})
    assert math.isnan(m.value("y", (), "a"))


def test_fix_cells_none_keeps_level(m: Any) -> None:
    m.set_values("x", "a", {I1: 3.0})
    m.fix_cells("x", "a", {I1: None, I2: 7.0})
    assert m.fixed_cells("x", "a") == frozenset({I1, I2})
    assert m.values("x", "a") == {I1: 3.0, I2: 7.0}
    assert m.fixed_cells("x", "b") == frozenset()


def test_unfix_cells(m: Any) -> None:
    m.fix_cells("x", "a", {I1: None, I2: None})
    m.unfix_cells("x", "a", [I2])
    assert m.fixed_cells("x", "a") == frozenset({I1})


def test_fix_all_keeps_levels_and_skips_keep(m: Any) -> None:
    m.set_values("x", "a", {I1: 3.0})
    m.fix_all("x", "a", keep=frozenset({I2}))
    assert m.fixed_cells("x", "a") == frozenset({I1})
    assert m.values("x", "a") == {I1: 3.0, I2: 1.0}
    assert m.fixed_cells("x", "b") == frozenset()
    m.fix_all("z", "a")
    assert m.fixed_cells("z", "a") == frozenset({I1, I2})
    assert all(math.isnan(v) for v in m.values("z", "a").values())


def test_unfix_all_skips_keep(m: Any) -> None:
    for t in ("a", "b"):
        m.fix_cells("x", t, {I1: None, I2: None})
    m.unfix_all("x", "a", keep=frozenset({I1}))
    assert m.fixed_cells("x", "a") == frozenset({I1})
    assert m.fixed_cells("x", "b") == frozenset({I1, I2})
    m.unfix_all("x", "a")
    assert m.fixed_cells("x", "a") == frozenset()


def test_set_all_rows_active_one_period(m: Any) -> None:
    m.set_all_rows_active("e", "a", False)
    m.set_all_rows_active("g", "a", False)
    assert m.active_rows("e", "a") == frozenset()
    assert m.active_rows("g", "a") == frozenset()
    assert m.active_rows("e", "b") == frozenset({I1, I2})
    m.set_all_rows_active("e", "a", True)
    assert m.active_rows("e", "a") == frozenset({I1, I2})


def test_set_bounds_none_side_untouched(m: Any) -> None:
    m.set_bounds("x", "a", {I1: (None, 9.0)})
    m.set_bounds("x", "a", {I2: (0.7, None)})
    assert m.bounds("x", "a") == {I1: (0.5, 9.0), I2: (0.7, None)}


def test_free_cells_drops_domain_floor(m: Any) -> None:
    m.free_cells("x", "a", [I1])
    assert m.bounds("x", "a") == {I1: (None, None), I2: (0.5, None)}
    assert m.bounds("x", "b")[I1] == (0.5, None)


def test_set_rows_active_one_period(m: Any) -> None:
    m.set_rows_active("e", "a", [I1], False)
    assert m.active_rows("e", "a") == frozenset({I2})
    assert m.active_rows("e", "b") == frozenset({I1, I2})
    m.set_rows_active("e", "a", [I1], True)
    assert m.active_rows("e", "a") == frozenset({I1, I2})


def test_pyomo_free_cells_domain_is_reals() -> None:
    pm = _pyomo()
    pm.free_cells("x", "a", [I1])
    cell = cast("Any", pm.pyomo.x)["i1", "a"]
    assert cell.domain is pyo.Reals and cell.lb is None


def test_pyomo_rows_without_period_never_listed() -> None:
    pm = _pyomo()
    pm.pyomo.h = pyo.Constraint(["i1"], rule=lambda mm, i: mm.y["a"] == 2)
    assert pm.row_cells("h", "a") == ()
    assert pm.active_rows("h", "a") == frozenset()


def test_family_free_cells_lb_is_minus_inf() -> None:
    fm = _family()
    fm.free_cells("x", "a", [I1])
    x = fm.mp.vars["x"]
    i = x.pos(("i1", "a"))
    assert x.lb[i] == -math.inf and x.ub[i] == math.inf
    assert fm.backend == "family"
