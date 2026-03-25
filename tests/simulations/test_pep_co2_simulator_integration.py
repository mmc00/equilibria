from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

from equilibria.simulations.pep_co2 import PepCO2Simulator
from equilibria.simulations.adapters.pep_co2 import PepCO2Adapter
from equilibria.templates.pep_calibration_unified import PEPModelState
from equilibria.templates.pep_co2_data import get_state_co2_block


def _base_state() -> PEPModelState:
    state = PEPModelState(sets={"J": ["agr", "ser"]})
    return state


def test_pep_co2_simulator_runs_sector_and_scale_scenarios(monkeypatch: Any) -> None:
    captured: list[dict[str, Any]] = []

    def _fake_fit(self: PepCO2Adapter) -> PEPModelState:
        state = _base_state()
        self._sets = dict(state.sets)
        block = {
            "co2_intensity": {"agr": 1.0, "ser": 2.0},
            "tco2b": {"agr": 10.0, "ser": 20.0},
            "tco2scal": 1.0,
        }
        setattr(state, "_pep_co2", block)
        return state

    def _fake_solve(
        _self: PepCO2Adapter,
        state: PEPModelState,
        *,
        initial_vars: Any | None,
        reference_results_gdx: Any | None,
        reference_slice: str,
        scenario: Any | None = None,
    ) -> tuple[Any, Any, dict[str, Any]]:
        _ = reference_results_gdx, reference_slice, scenario
        block = copy.deepcopy(get_state_co2_block(state))
        captured.append(
            {
                "block": block,
                "initial_vars": copy.deepcopy(initial_vars),
                "scenario_name": None if scenario is None else scenario.name,
            }
        )
        vars_obj = {
            "tco2scal": float(block["tco2scal"]),
            "tco2b": dict(block["tco2b"]),
            "co2_intensity": dict(block["co2_intensity"]),
        }
        solver = SimpleNamespace(params={"mock": 1.0})
        solution = SimpleNamespace(
            converged=True,
            iterations=1,
            final_residual=0.0,
            message="ok",
            variables=vars_obj,
        )
        return solver, solution, {"passed": True}

    def _fake_key(_self: PepCO2Adapter, vars_obj: Any) -> dict[str, float]:
        return {
            "tco2scal": float(vars_obj["tco2scal"]),
            "tco2b_sum": float(sum(vars_obj["tco2b"].values())),
            "co2_intensity_sum": float(sum(vars_obj["co2_intensity"].values())),
        }

    monkeypatch.setattr(PepCO2Adapter, "fit_base_state", _fake_fit)
    monkeypatch.setattr(PepCO2Adapter, "solve_state", _fake_solve)
    monkeypatch.setattr(PepCO2Adapter, "key_indicators", _fake_key)

    sim = PepCO2Simulator(
        co2_intensity={"agr": 1.0, "ser": 2.0},
        tco2b={"agr": 10.0, "ser": 20.0},
    ).fit()

    report = sim.run_scenarios(
        scenarios=[
            sim.shock(var="tco2scal", multiplier=2.0, name="double_scale"),
            sim.shock(var="tco2b", index="agr", op="add", value=5.0, name="agr_plus_five"),
        ],
        warm_start=True,
        include_base=True,
    )

    assert report["model"] == "pep_co2"
    assert report["base"]["solve"]["key_indicators"]["tco2scal"] == pytest.approx(1.0)
    assert report["base"]["solve"]["key_indicators"]["tco2b_sum"] == pytest.approx(30.0)
    assert report["scenarios"][0]["solve"]["key_indicators"]["tco2scal"] == pytest.approx(2.0)
    assert report["scenarios"][1]["solve"]["key_indicators"]["tco2b_sum"] == pytest.approx(35.0)

    assert len(captured) == 3
    assert captured[0]["scenario_name"] is None
    assert captured[0]["initial_vars"] is None
    assert captured[1]["scenario_name"] == "double_scale"
    assert captured[1]["initial_vars"]["tco2scal"] == pytest.approx(1.0)
    assert captured[2]["scenario_name"] == "agr_plus_five"
    assert captured[2]["initial_vars"]["tco2scal"] == pytest.approx(2.0)
    assert captured[2]["block"]["tco2b"]["agr"] == pytest.approx(15.0)
