"""simple_open matches Hosoe's stdcge in GAMS in every cell of base, check, shock."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pyomo.environ as pyo
import pytest
from tests.templates.simple_open.conftest import gams_points

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values
from equilibria.compare import Mapping, compare, from_gdx, from_pyomo
from equilibria.templates.simple_open import SimpleOpenEconomy
from equilibria.templates.simple_open.template import equivalent_variation

POINTS = ("base", "check", "shock")


@pytest.fixture(scope="module")
def run() -> dict[str, Any]:
    return SimpleOpenEconomy().run()


@pytest.mark.parametrize("point", POINTS)
def test_matches_gams(run: dict[str, Any], point: str) -> None:
    report = compare(
        from_pyomo(run[point], period=point),
        from_gdx(gams_points() / f"{point}.gdx", period=point),
        Mapping.builtin("stdcge"),
        reference="b",
    )
    overall = report.overall
    assert overall["cells"] > 40
    assert overall["missing_in_a"] == overall["missing_in_b"] == 0
    assert overall["passed"] == overall["cells"], report.summary


def test_ev_matches_gams(run: dict[str, Any]) -> None:
    gams = read_parameter_values(read_gdx(gams_points() / "shock.gdx"), "EV")[()]
    ours = equivalent_variation(SimpleOpenEconomy().calibration(), run["shock"])
    assert ours == pytest.approx(gams, rel=1e-8, abs=1e-8)


def test_walras_is_zero_after_solving(run: dict[str, Any]) -> None:
    for point in ("check", "shock"):
        assert abs(pyo.value(run[point].WALRAS)) <= 1e-8, point


def test_shock_does_not_leak_into_next_run(run: dict[str, Any]) -> None:
    template = SimpleOpenEconomy()
    again = template.run()
    tau_m = template.calibration().values["tau_m"]
    check = cast("Any", again["check"])
    shock = cast("Any", run["shock"])
    got = [pyo.value(check.tau_m[i]) for i in check.I]
    np.testing.assert_allclose(got, tau_m, rtol=0, atol=0)
    assert all(pyo.value(shock.tau_m[i]) == 0 for i in shock.I)
