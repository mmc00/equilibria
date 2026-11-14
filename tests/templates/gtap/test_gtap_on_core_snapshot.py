"""The core route prepares every GTAP period as ``solve_multiperiod`` does.

Each case of ``_period_snapshot.CASES`` runs through ``Exercise(GTAP, gtap_static)``
with a fake solver that photographs the model, and the photos must equal the ones
the old driver saved in the fixture: same calls, same closure per call, every
component's hash. The fixture is the old driver's output on ``main`` and is never
regenerated from this route.
"""

from __future__ import annotations

from typing import Any

import pytest
from tests.templates.gtap import _period_snapshot as snap

pytestmark = pytest.mark.skipif(
    not snap.DATASET.exists(), reason="gtap7_3x3 dataset not present"
)


@pytest.mark.parametrize("case", sorted(snap.CASES))
def test_exercise_prepares_like_the_driver(case: str, monkeypatch) -> None:
    got = snap.take(case, monkeypatch, route="exercise")
    data = snap.load_fixture()
    assert case in data, f"no saved photo for {case}"
    problems = snap.diff(data[case], got)
    assert not problems, f"{case}:\n  " + "\n  ".join(problems[:40])


def test_endowment_check_only_after_optimal(monkeypatch) -> None:
    from equilibria.core.exercise import Exercise
    from equilibria.solver import path_gtap
    from equilibria.templates.gtap import shocks
    from equilibria.templates.gtap.recipe import gtap_static
    from equilibria.templates.gtap.template import GTAP

    class _Fails:
        @staticmethod
        def _run_path_capi_nonlinear_full(*_: Any, **__: Any) -> dict:
            return {"termination_code": 2, "residual": 1.0}

    def _never(*_: Any) -> None:
        raise AssertionError("the endowment check ran after a failed solve")

    monkeypatch.setattr(path_gtap, "_load_path_capi", lambda: _Fails)
    monkeypatch.setattr(shocks, "check_endowment_shock_entered", _never)
    p = snap._load_params()
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    closure = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        if_sub=False,
        numeraire="pnum",
    )
    res = Exercise(template=GTAP(p, closure), recipe=gtap_static()).run()
    assert res["shock"]["code"] == "code 2"
