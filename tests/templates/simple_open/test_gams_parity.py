"""simple_open matches Hosoe's stdcge in GAMS in every cell of base, check, shock,
run through the core driver (one model with the three periods)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pyomo.environ as pyo
import pytest
from tests.templates.simple_open._helpers import run_tariff_abolition
from tests.templates.simple_open.conftest import gams_points

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values
from equilibria.compare import Mapping, compare, from_dict, from_gdx
from equilibria.core import Exercise
from equilibria.templates.simple_open import SimpleOpenEconomy
from equilibria.templates.simple_open.template import equivalent_variation

POINTS = ("base", "check", "shock")


@pytest.fixture(scope="module")
def model() -> Any:
    return run_tariff_abolition(SimpleOpenEconomy()).model


def _period(model: Any, period: str) -> dict[str, dict[tuple, float]]:
    """One period's cells, keyed as in a single-period model."""
    cells: dict[str, dict[tuple, float]] = defaultdict(dict)
    for v in model.component_objects(pyo.Var, active=True):
        for k in v:
            key = k if isinstance(k, tuple) else (k,)
            if key[-1] == period and v[k].value is not None:
                cells[v.name][key[:-1]] = pyo.value(v[k])
    return cells


@pytest.mark.parametrize("point", POINTS)
def test_matches_gams(model: Any, point: str) -> None:
    report = compare(
        from_dict(_period(model, point), period=point),
        from_gdx(gams_points() / f"{point}.gdx", period=point),
        Mapping.builtin("stdcge"),
        reference="b",
    )
    overall = report.overall
    assert overall["cells"] > 40
    assert overall["missing_in_a"] == overall["missing_in_b"] == 0
    assert overall["passed"] == overall["cells"], report.summary


def test_ev_matches_gams(model: Any) -> None:
    gams = read_parameter_values(read_gdx(gams_points() / "shock.gdx"), "EV")[()]
    ours = equivalent_variation(SimpleOpenEconomy().calibration(), model)
    assert ours == pytest.approx(gams, rel=1e-8, abs=1e-8)


def test_walras_is_zero_after_solving(model: Any) -> None:
    for point in ("check", "shock"):
        assert abs(pyo.value(model.WALRAS[point])) <= 1e-8, point


def test_shock_does_not_leak_into_next_run() -> None:
    template = SimpleOpenEconomy()
    run_tariff_abolition(template)
    again = Exercise(template)
    again.run()
    tau_m = template.calibration().values["tau_m"]
    for t in POINTS:
        got = [pyo.value(again.model.tau_m[i, t]) for i in template.calibration().goods]
        np.testing.assert_allclose(got, tau_m, rtol=0, atol=0)
