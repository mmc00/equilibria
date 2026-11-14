"""ipopt(): a solver a recipe can name."""

from __future__ import annotations

from typing import Any, cast

import pyomo.environ as pyo
import pytest

from equilibria.backends.pyomo_periods import PyomoPeriodModel
from equilibria.solver.ipopt import ipopt


def test_solves_a_square_system() -> None:
    m = cast("Any", pyo.ConcreteModel())
    m.x = pyo.Var(initialize=0.0)
    m.y = pyo.Var(initialize=0.0)
    m.a = pyo.Constraint(expr=m.x + m.y == 3)
    m.b = pyo.Constraint(expr=m.x - m.y == 1)
    assert ipopt()(PyomoPeriodModel(m)) == "optimal"
    assert pyo.value(m.x) == pytest.approx(2.0)
    assert pyo.value(m.y) == pytest.approx(1.0)


def test_reports_an_infeasible_system_instead_of_raising() -> None:
    m = cast("Any", pyo.ConcreteModel())
    m.x = pyo.Var(initialize=0.0)
    m.a = pyo.Constraint(expr=m.x == 1)
    m.b = pyo.Constraint(expr=m.x == 2)
    assert ipopt()(PyomoPeriodModel(m)) != "optimal"
