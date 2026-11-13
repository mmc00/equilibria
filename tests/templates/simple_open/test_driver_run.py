"""simple_open runs base, check and shock through the core driver (ADR-0011)."""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo
import pytest
from tests.templates.simple_open._helpers import run_tariff_abolition

from equilibria.blocks.equilibrium import Numeraire
from equilibria.core import Exercise
from equilibria.templates.simple_open import SimpleOpenEconomy, stdcge_static

TARIFF_ABOLITION = {"tau_m": 0.0}


def _cells(m: Any, period: str) -> dict[tuple[str, tuple], float]:
    out = {}
    for v in m.component_objects(pyo.Var, active=True):
        for k in v:
            key = k if isinstance(k, tuple) else (k,)
            if key[-1] == period:
                out[(v.name, key[:-1])] = pyo.value(v[k])
    return out


def test_base_check_shock_solve_through_the_driver() -> None:
    results = Exercise(SimpleOpenEconomy(), shock=TARIFF_ABOLITION).run()
    assert results["base"]["code"] == "not solved"
    assert results["base"]["residual"] < 1e-9
    assert results["check"]["code"] == results["shock"]["code"] == "optimal"
    assert results["shock"]["residual"] < 1e-8


def test_check_returns_the_base() -> None:
    ex = Exercise(SimpleOpenEconomy(), shock=TARIFF_ABOLITION)
    ex.run()
    base, check = _cells(ex.model, "base"), _cells(ex.model, "check")
    for cell, value in base.items():
        assert check[cell] == pytest.approx(value, rel=1e-9, abs=1e-9), cell


def test_the_shock_moves_only_the_shock_period() -> None:
    ex = Exercise(SimpleOpenEconomy(), shock=TARIFF_ABOLITION)
    ex.run()
    tau_m = ex.model.tau_m
    goods = SimpleOpenEconomy().calibration().goods
    assert all(pyo.value(tau_m[g, "shock"]) == 0.0 for g in goods)
    assert all(pyo.value(tau_m[g, "check"]) > 0.0 for g in goods)


@pytest.mark.parametrize(
    "shock",
    [{"Q": 0.0}, {"tau_mm": 0.0}, {"tau_m": {"XYZ": 0.0}}],
    ids=["free variable", "typo", "missing index"],
)
def test_a_shock_on_a_non_instrument_or_missing_index_fails(shock: dict) -> None:
    with pytest.raises(ValueError, match=r"Q|tau_mm|XYZ"):
        Exercise(SimpleOpenEconomy(), shock=shock).run()


def _other_numeraire(template: SimpleOpenEconomy) -> Numeraire:
    cal = template.calibration()
    other = next(f for f in cal.factors if f != template.numeraire_factor())
    return Numeraire(numeraire=other, values=cal.values, start=cal.base)


def test_an_overwrite_that_matches_no_block_fails() -> None:
    template = SimpleOpenEconomy()
    stray = _other_numeraire(template).model_copy(update={"name": "NoSuchBlock"})
    with pytest.raises(ValueError, match="NoSuchBlock"):
        Exercise(template, overwrite=[stray]).run()


def test_overwrite_does_not_leak_into_the_next_exercise() -> None:
    template = SimpleOpenEconomy()
    first = Exercise(template, overwrite=[_other_numeraire(template)])
    assert first.run()["check"]["code"] == "optimal"
    other = _other_numeraire(template).numeraire
    assert pyo.value(first.model.WF[other, "check"]) == pytest.approx(1.0)

    second = Exercise(template)
    assert second.run()["check"]["code"] == "optimal"
    num = template.numeraire_factor()
    assert pyo.value(second.model.WF[num, "check"]) == pytest.approx(1.0)
    assert len(second.model.eq_num_con) == 3  # one numeraire row per period


def test_two_exercises_from_one_cached_model_are_independent() -> None:
    template = SimpleOpenEconomy()
    first = Exercise(template)
    first.run()
    good = template.calibration().goods[0]
    first.model.tau_m[good, "check"].fix(0.5)

    second = Exercise(template)
    second.run()
    calibrated = float(template.calibration().values["tau_m"][0])
    assert pyo.value(second.model.tau_m[good, "check"]) == pytest.approx(calibrated)
    assert second.model is not first.model


def test_a_failed_solve_is_reported_and_leaves_the_cache_clean() -> None:
    template = SimpleOpenEconomy()

    def diverge(model: Any) -> str:
        for good in template.calibration().goods:  # a solver that wanders off
            model.M[good, "base"].set_value(1e6)
            model.M[good, "check"].set_value(1e6)
        return "infeasible"

    failed = Exercise(template, recipe=stdcge_static(solver=diverge))
    results = failed.run()
    assert [r["code"] for r in results.values()] == [
        "not solved",
        "infeasible",
        "infeasible",
    ]
    assert results["check"]["residual"] > 1.0

    again = Exercise(template, shock=TARIFF_ABOLITION)
    assert again.run()["check"]["code"] == "optimal"
    good = template.calibration().goods[0]
    base_imports = template.calibration().base["M"][0]
    # base is never solved: it shows exactly what the template handed over
    assert pyo.value(again.model.M[good, "base"]) == pytest.approx(base_imports)


def test_changing_the_template_rebuilds_its_model() -> None:
    template = SimpleOpenEconomy()
    run_tariff_abolition(template)
    template.sigma = 8.0
    changed = run_tariff_abolition(template).model
    fresh = run_tariff_abolition(SimpleOpenEconomy(sigma=8.0)).model
    for good in template.calibration().goods:
        assert pyo.value(changed.M[good, "shock"]) == pytest.approx(
            pyo.value(fresh.M[good, "shock"]), rel=1e-9
        )
