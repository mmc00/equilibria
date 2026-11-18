"""simple_open on the family backend matches Hosoe's stdcge in GAMS in every cell of
base, check and shock (the tariff abolition), with no Pyomo model.

The tolerance is 1e-9 relative (Marlon, 2026-11-18), tighter than the stdcge map's
1e-6: measured, both backends sit 4.2e-11 from GAMS at worst (the shock's E[MLK]),
which is GAMS's own precision (``scripts/simple_open/measure_gams_gap.py``). The map
file is shared with the Pyomo test and keeps its 1e-6; this test tightens a copy.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Any

import numpy as np
import pytest

pytest.importorskip("jax")

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values  # noqa: E402
from equilibria.compare import Mapping, compare, from_dict, from_gdx  # noqa: E402
from equilibria.compare.mapping import Tolerance  # noqa: E402
from equilibria.core import Exercise  # noqa: E402
from equilibria.templates.simple_open import SimpleOpenEconomy  # noqa: E402
from equilibria.templates.simple_open.template import (  # noqa: E402
    equivalent_variation,
)
from tests.templates.simple_open.conftest import gams_points  # noqa: E402

POINTS = ("base", "check", "shock")
TOL_REL = 1e-9
TARIFF_ABOLITION = {"tau_m": 0.0}


def _mapping() -> Mapping:
    """``stdcge`` with every variable at ``TOL_REL`` (abs_floor as in the map)."""
    m = Mapping.builtin("stdcge")
    tight = tuple(
        dataclasses.replace(v, tol=Tolerance(TOL_REL, v.tol.abs_floor))
        for v in m.variables
    )
    return dataclasses.replace(m, variables=tight)


@pytest.fixture(scope="module")
def model() -> Any:
    ex = Exercise(SimpleOpenEconomy(backend="family"), shock=TARIFF_ABOLITION)
    results = ex.run()
    assert results["check"]["code"] == results["shock"]["code"] == "optimal", results
    return ex.model


def _period(model: Any, period: str) -> dict[str, dict[tuple, float]]:
    cells: dict[str, dict[tuple, float]] = defaultdict(dict)
    for name, index in model.cells(period):
        cells[name][index] = model.value(name, index, period)
    return cells


@pytest.mark.parametrize("point", POINTS)
def test_matches_gams(model: Any, point: str) -> None:
    report = compare(
        from_dict(_period(model, point), period=point),
        from_gdx(gams_points() / f"{point}.gdx", period=point),
        _mapping(),
        reference="b",
    )
    overall = report.overall
    assert overall["cells"] > 40
    assert overall["missing_in_a"] == overall["missing_in_b"] == 0
    assert overall["passed"] == overall["cells"], report.summary


def test_ev_matches_gams(model: Any) -> None:
    gams = read_parameter_values(read_gdx(gams_points() / "shock.gdx"), "EV")[()]
    ours = equivalent_variation(SimpleOpenEconomy().calibration(), model)
    assert ours == pytest.approx(gams, rel=TOL_REL, abs=TOL_REL)


def test_walras_is_zero_after_solving(model: Any) -> None:
    for point in ("check", "shock"):
        assert abs(model.value("WALRAS", (), point)) <= 1e-8, point


def test_shock_does_not_leak_into_next_run() -> None:
    template = SimpleOpenEconomy(backend="family")
    Exercise(template, shock=TARIFF_ABOLITION).run()
    again = Exercise(template)
    again.run()
    assert again.model is not None
    tau_m = template.calibration().values["tau_m"]
    goods = template.calibration().goods
    for t in POINTS:
        got = [again.model.value("tau_m", (i,), t) for i in goods]
        np.testing.assert_allclose(got, tau_m, rtol=0, atol=0)
