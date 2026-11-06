"""equilibria.report.summary: the end-of-solve summary (Walras, residuals, worst equations)."""

from __future__ import annotations

import io

import pytest
from pyomo.environ import ConcreteModel, Constraint, Var
from rich.console import Console

from equilibria.report import summary


def _model():
    """Three equations with residuals 0, 0.5 and 0.01 in `shock`, one in `check`."""
    m = ConcreteModel()
    m.t = ["check", "shock"]
    m.x = Var(m.t, initialize=1.0)
    m.y = Var(m.t, initialize={"check": 9.0, "shock": 2.5})  # check: 7.0 off
    m.z = Var(m.t, initialize=3.01)
    m.walras = Var(m.t, initialize={"check": 0.0, "shock": -2e-9})
    m.eq_x = Constraint(m.t, rule=lambda m, t: m.x[t] == 1.0)
    m.eq_y = Constraint(m.t, rule=lambda m, t: m.y[t] == 2.0)
    m.eq_z = Constraint(m.t, rule=lambda m, t: m.z[t] == 3.0)
    # an inequality that holds has no residual, even though body != bound
    m.eq_cap = Constraint(m.t, rule=lambda m, t: m.x[t] <= 5.0)
    return m


def _quiet():
    return Console(file=io.StringIO(), width=100, record=True)


def test_residual_stats_and_worst_equations_of_the_period():
    s = summary(_model(), period="shock", console=_quiet())

    assert s.n_equations == 4
    assert s.max_residual == pytest.approx(0.5)
    assert s.mean_residual == pytest.approx((0.5 + 0.01) / 4)
    assert s.median_residual == pytest.approx(0.005)
    assert [w.equation for w in s.worst[:2]] == ["eq_y[shock]", "eq_z[shock]"]
    assert s.worst[0].residual == pytest.approx(0.5)


def test_other_periods_do_not_count():
    s = summary(_model(), period="check", console=_quiet())

    assert s.max_residual == pytest.approx(7.0)
    assert s.worst[0].equation == "eq_y[check]"


def test_walras_is_read_for_the_period():
    s = summary(_model(), period="shock", console=_quiet())

    assert s.walras == pytest.approx(-2e-9)


def test_model_without_walras_or_periods():
    m = ConcreteModel()
    m.a = Var(initialize=4.0)
    m.eq = Constraint(expr=m.a == 4.0)

    s = summary(m, console=_quiet())

    assert s.walras is None
    assert s.n_equations == 1
    assert s.max_residual == 0.0


def test_termination_code_comes_from_the_results():
    results = {"shock": {"code": 1, "residual": 1e-10}}

    s = summary(_model(), results, period="shock", console=_quiet())

    assert s.code == 1


def test_panel_shows_the_numbers():
    console = _quiet()
    summary(_model(), {"shock": {"code": 1}}, period="shock", console=console)
    out = console.export_text()

    assert "SOLVE COMPLETE" in out
    assert "eq_y[shock]" in out
    assert "5e-01" in out  # max residual
    assert "-2e-09" in out  # walras
