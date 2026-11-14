"""Shock strategies over PeriodModel: in steps (GEMPACK Gragg path) and Johansen."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pyomo.environ as pyo
import pytest

from equilibria.backends.pyomo_periods import PyomoPeriodModel, reflect
from equilibria.templates.gtap.strategies import in_steps, johansen, shocked_cells

PERIODS = ("base", "check", "shock")


def _model(chk: float = 2.0, full: float = 2.5) -> PyomoPeriodModel:
    """``x`` an instrument (fixed), ``y`` free; the shock moves ``x`` from
    ``chk`` to ``full``."""
    sp: Any = pyo.ConcreteModel()
    x = cast("Any", pyo.Var(initialize=chk))
    sp.x = x
    x.fix()
    sp.y = pyo.Var(initialize=2 * chk)
    sp.e = pyo.Constraint(rule=lambda mm: mm.y == 2 * mm.x)
    m = PyomoPeriodModel(reflect(sp, PERIODS))
    m.fix("x", (), "shock", full)
    return m


def _solver(f: Callable[[float], float], log: list[float], fail_at: int = 0):
    """Sets ``y = f(x)`` in the shock period; fails on call ``fail_at`` (1-based)."""

    def solve(m: Any) -> str:
        x = m.value("x", (), "shock")
        log.append(x)
        if fail_at and len(log) == fail_at:
            return "code 2"
        m.set_value("y", (), "shock", f(x))
        return "optimal"

    return solve


def test_shocked_cells() -> None:
    assert shocked_cells(_model()) == [("x", (), 2.0, 2.5)]
    assert shocked_cells(_model(full=2.0)) == []


def test_in_steps_path_is_geometric() -> None:
    log: list[float] = []
    m = _model()
    assert in_steps((0.5,))(m, _solver(lambda x: 2 * x, log)) == "optimal"
    assert log == pytest.approx([2.0 * (2.5 / 2.0) ** 0.5, 2.5])


def test_in_steps_stops_at_first_failure() -> None:
    log: list[float] = []
    m = _model()
    status = in_steps((0.25, 0.5, 1.0))(m, _solver(lambda x: 2 * x, log, fail_at=2))
    assert status == "code 2"
    assert len(log) == 2
    assert m.value("x", (), "shock") == 2.5 and m.is_fixed("x", (), "shock")


def test_in_steps_linear_path_when_not_positive() -> None:
    log: list[float] = []
    m = _model(chk=-1.0, full=1.0)
    in_steps((0.5,))(m, _solver(lambda x: 2 * x, log))
    assert log == pytest.approx([0.0, 1.0])


def test_johansen_linear_model_is_exact() -> None:
    m = _model(chk=-1.0, full=1.0)
    m.set_value("y", (), "check", -2.0)
    assert johansen()(m, _solver(lambda x: 2 * x, [])) == "optimal"
    assert m.value("y", (), "shock") == pytest.approx(2.0, abs=1e-9)
    assert m.value("x", (), "shock") == 1.0


def test_johansen_positive_uses_log_slope() -> None:
    m = _model(chk=2.0, full=2.2)
    m.set_value("y", (), "check", 4.0)
    johansen()(m, _solver(lambda x: x**2, []))
    pct = 0.1
    assert m.value("y", (), "shock") == pytest.approx(4.0 * (1 + 2 * pct), rel=1e-9)


def test_johansen_runs_after_at_each_point() -> None:
    seen: list[float] = []
    m = _model()
    m.set_value("y", (), "check", 4.0)
    johansen(after=lambda mm: seen.append(mm.value("x", (), "shock")))(
        m, _solver(lambda x: 2 * x, [])
    )
    assert len(seen) == 4


def test_johansen_failure_returns_status_and_restores_shock() -> None:
    m = _model()
    m.set_value("y", (), "check", 4.0)
    status = johansen()(m, _solver(lambda x: 2 * x, [], fail_at=1))
    assert status == "code 2"
    assert m.value("x", (), "shock") == 2.5


@pytest.mark.parametrize("strategy", [in_steps((0.5,)), johansen()])
def test_no_shock_raises(strategy: Any) -> None:
    with pytest.raises(ValueError, match="no shocked cell"):
        strategy(_model(full=2.0), _solver(lambda x: 2 * x, []))
