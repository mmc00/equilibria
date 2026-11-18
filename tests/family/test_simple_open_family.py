"""simple_open on the family backend through the core driver (ADR-0011): the
same runs as ``tests/templates/simple_open/test_driver_run.py``, with no Pyomo."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("jax")

from equilibria.core import Exercise  # noqa: E402
from equilibria.templates.simple_open import SimpleOpenEconomy  # noqa: E402
from tests.family.test_family_e2e import _spies  # noqa: E402

TARIFF_ABOLITION = {"tau_m": 0.0}


def _family() -> SimpleOpenEconomy:
    return SimpleOpenEconomy(backend="family")


def _cells(m: Any, period: str) -> dict[tuple[str, tuple], float]:
    return {(n, i): m.value(n, i, period) for n, i in m.cells(period)}


def test_base_check_shock_solve_through_the_driver() -> None:
    results = Exercise(_family(), shock=TARIFF_ABOLITION).run()
    assert results["base"]["code"] == "not solved"
    assert results["base"]["residual"] < 1e-9
    assert results["check"]["code"] == results["shock"]["code"] == "optimal"
    assert results["shock"]["residual"] < 1e-8


def test_check_returns_the_base() -> None:
    ex = Exercise(_family(), shock=TARIFF_ABOLITION)
    ex.run()
    base, check = _cells(ex.model, "base"), _cells(ex.model, "check")
    for cell, value in base.items():
        assert check[cell] == pytest.approx(value, rel=1e-9, abs=1e-9), cell


def test_the_shock_moves_only_the_shock_period() -> None:
    ex = Exercise(_family(), shock=TARIFF_ABOLITION)
    ex.run()
    m = ex.model
    assert m is not None
    goods = _family().calibration().goods
    assert all(m.value("tau_m", (g,), "shock") == 0.0 for g in goods)
    assert all(m.value("tau_m", (g,), "check") > 0.0 for g in goods)


def test_no_pyomo_model_built_in_a_run() -> None:
    with _spies() as calls:
        Exercise(_family(), shock=TARIFF_ABOLITION).run()
    assert calls == {"ConcreteModel": 0, "PyomoNLP": 0, "identify_variables": 0}


def test_runs_do_not_leak() -> None:
    template = _family()
    first = Exercise(template, shock=TARIFF_ABOLITION)
    first.run()
    second = Exercise(template, shock=TARIFF_ABOLITION)
    second.run()
    assert first.model is not second.model
    assert _cells(first.model, "shock") == _cells(second.model, "shock")
    # a run without the shock starts from the calibration, not from the shocked run
    third = Exercise(template)
    third.run()
    assert _cells(third.model, "shock") == pytest.approx(_cells(third.model, "base"))


def test_pyomo_default_unchanged() -> None:
    assert SimpleOpenEconomy().backend == "pyomo"
