"""reflect: a single-period Pyomo model becomes one indexed by a trailing period;
PyomoPeriodModel gives the run machinery its operations over it."""

from __future__ import annotations

from typing import Any, cast

import pyomo.environ as pyo
import pytest

from equilibria.backends.pyomo_periods import (
    PyomoPeriodModel,
    exogenous,
    period_residual,
    reflect,
)

PERIODS = ("base", "check", "shock")


def single_period() -> Any:
    """x[i] == p*y + tau[i]; y == 2; tau fixed; p a mutable parameter."""
    sp: Any = pyo.ConcreteModel()
    sp.I = pyo.Set(initialize=["a", "b"])
    sp.p = pyo.Param(initialize=3.0, mutable=True)
    sp.y = pyo.Var(initialize=2.0, bounds=(0.5, 10.0))
    tau = cast("Any", pyo.Var(sp.I, initialize={"a": 0.1, "b": 0.2}, domain=pyo.Reals))
    sp.tau = tau
    tau.fix()
    sp.x = pyo.Var(sp.I, initialize={"a": 6.1, "b": 6.2}, bounds=(1e-5, None))
    sp.c = pyo.Constraint(sp.I, rule=lambda m, i: m.x[i] == m.p * m.y + m.tau[i])
    sp.d = pyo.Constraint(expr=sp.y == 2)
    sp.obj = pyo.Objective(expr=0.0)
    return sp


def test_vars_get_a_trailing_period_and_keep_their_state() -> None:
    m = reflect(single_period(), PERIODS)
    assert list(m.t) == list(PERIODS)
    for t in PERIODS:
        assert m.x["a", t].value == pytest.approx(6.1)
        assert m.x["b", t].lb == 1e-5 and m.x["b", t].ub is None
        assert m.y[t].value == 2.0
        assert (m.y[t].lb, m.y[t].ub) == (0.5, 10.0)
        assert m.tau["a", t].domain is pyo.Reals


def test_fixed_cells_are_fixed_in_every_period_and_listed_as_exogenous() -> None:
    m = reflect(single_period(), PERIODS)
    assert all(m.tau[i, t].fixed for i in ("a", "b") for t in PERIODS)
    assert not m.x["a", "base"].fixed
    assert exogenous(m) == {("tau", ("a",)), ("tau", ("b",))}


def test_each_period_row_reads_its_own_period() -> None:
    m = reflect(single_period(), PERIODS)
    assert period_residual(m, "shock") == pytest.approx(0.0, abs=1e-12)
    m.x["a", "shock"].set_value(7.0)
    assert period_residual(m, "shock") == pytest.approx(0.9)
    assert period_residual(m, "check") == pytest.approx(0.0, abs=1e-12)


def test_period_residual_skips_inactive_rows() -> None:
    m = reflect(single_period(), PERIODS)
    m.x["a", "shock"].set_value(7.0)
    m.c["a", "shock"].deactivate()
    assert period_residual(m, "shock") == pytest.approx(0.0, abs=1e-12)


def test_mutable_params_are_folded_to_their_value() -> None:
    sp = single_period()
    m = reflect(sp, PERIODS)
    sp.p.set_value(99.0)
    assert period_residual(m, "check") == pytest.approx(0.0, abs=1e-12)


def test_objective_is_not_copied() -> None:
    m = reflect(single_period(), PERIODS)
    assert not list(m.component_objects(pyo.Objective))


def test_scalar_rows_are_indexed_by_period() -> None:
    m = reflect(single_period(), PERIODS)
    assert set(m.d) == set(PERIODS)


@pytest.mark.parametrize("periods", [(), ("base", "base")])
def test_bad_periods_raise(periods: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        reflect(single_period(), periods)


def test_mutable_params_in_bounds_are_folded_too() -> None:
    sp = single_period()
    sp.e = pyo.Constraint(expr=sp.y == sp.p - 1)  # rhs constant: p sits in a bound
    sp.r = pyo.Constraint(expr=pyo.inequality(sp.p - 3, sp.y, sp.p + 7))
    m = reflect(sp, PERIODS)
    sp.p.set_value(99.0)
    assert m.e["check"].lb == m.e["check"].ub == 2.0
    assert (m.r["check"].lb, m.r["check"].ub) == (0.0, 10.0)


def test_period_model_lists_periods_cells_and_exogenous() -> None:
    pm = PyomoPeriodModel(reflect(single_period(), PERIODS))
    assert pm.periods == PERIODS
    assert set(pm.cells("check")) == {
        ("x", ("a",)), ("x", ("b",)), ("y", ()), ("tau", ("a",)), ("tau", ("b",))
    }  # fmt: skip
    assert pm.exogenous() == {("tau", ("a",)), ("tau", ("b",))}


def test_period_model_reads_writes_fixes_one_cell() -> None:
    pm = PyomoPeriodModel(reflect(single_period(), PERIODS))
    pm.set_value("x", ("a",), "shock", 7.0)
    assert pm.value("x", ("a",), "shock") == 7.0
    assert pm.value("x", ("a",), "check") == pytest.approx(6.1)
    pm.fix("y", (), "base")
    assert pm.is_fixed("y", (), "base") and pm.value("y", (), "base") == 2.0
    pm.fix("tau", ("b",), "shock", 0.0)
    assert pm.value("tau", ("b",), "shock") == 0.0
    pm.unfix("y", (), "base")
    assert not pm.is_fixed("y", (), "base")


def test_period_model_activates_one_period_and_measures_it() -> None:
    pm = PyomoPeriodModel(reflect(single_period(), PERIODS))
    pm.set_value("x", ("a",), "shock", 7.0)
    pm.activate_rows("check")
    assert pm.pyomo.c["a", "check"].active and not pm.pyomo.c["a", "shock"].active
    assert pm.residual("shock") == pytest.approx(0.0, abs=1e-12)  # inactive rows
    pm.activate_rows("shock")
    assert pm.residual("shock") == pytest.approx(0.9)


def test_period_model_clone_is_independent() -> None:
    pm = PyomoPeriodModel(reflect(single_period(), PERIODS))
    copy = pm.clone()
    copy.set_value("x", ("a",), "check", 9.0)
    assert pm.value("x", ("a",), "check") == pytest.approx(6.1)
    assert copy.exogenous() == pm.exogenous()
