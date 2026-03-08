from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

from equilibria.simulations import (
    PepSimulator,
    export_tax,
    government_spending,
    import_shock,
)
from equilibria.simulations.adapters.pep import PepAdapter
from equilibria.templates.pep_calibration_unified import PEPModelState


def _base_state() -> PEPModelState:
    return PEPModelState(
        sets={"I": ["agr", "ser"]},
        trade={
            "ttixO": {"agr": 0.10, "ser": 0.20},
            "PWMO": {"agr": 1.00, "ser": 2.00},
            "PWXO": {"agr": 1.10, "ser": 1.20},
            "EXDO": {"agr": 10.0, "ser": 5.0},
            "PEO": {"agr": 2.0, "ser": 3.0},
            "tmrg_X": {
                ("agr", "agr"): 0.1,
                ("ser", "agr"): 0.2,
                ("agr", "ser"): 0.0,
                ("ser", "ser"): 0.1,
            },
            "PCO": {"agr": 1.0, "ser": 2.0},
        },
        income={
            "TICTO": 10.0,
            "TIMTO": 20.0,
            "TIXTO": 0.0,
            "TPRCTSO": 0.0,
            "YGKO": 100.0,
            "TDHTO": 5.0,
            "TDFTO": 7.0,
            "TPRODNO": 8.0,
            "YGTRO": 9.0,
            "YGO": 0.0,
        },
        consumption={"GO": 100.0},
    )


def test_pep_simulator_end_to_end_scenarios_with_optional_reference(monkeypatch: Any) -> None:
    captured: list[dict[str, Any]] = []

    def _fake_fit(self: PepAdapter) -> PEPModelState:
        state = _base_state()
        self._sets = dict(state.sets)
        return state

    def _fake_solve(
        _self: PepAdapter,
        state: PEPModelState,
        *,
        initial_vars: Any | None,
        reference_results_gdx: Any | None,
        reference_slice: str,
    ) -> tuple[Any, Any, dict[str, Any]]:
        captured.append(
            {
                "state": copy.deepcopy(state),
                "initial_vars": copy.deepcopy(initial_vars),
                "reference_results_gdx": reference_results_gdx,
                "reference_slice": reference_slice,
            }
        )
        vars_obj = {
            "GO": float(state.consumption.get("GO", 0.0)),
            "ttix": dict(state.trade.get("ttixO", {})),
            "PWM": dict(state.trade.get("PWMO", {})),
        }
        solver = SimpleNamespace(params={"mock": 1.0})
        solution = SimpleNamespace(
            converged=True,
            iterations=1,
            final_residual=0.0,
            message="ok",
            variables=vars_obj,
        )
        validation = {"passed": True}
        return solver, solution, validation

    def _fake_compare(
        self: PepAdapter,
        *,
        solution_vars: Any,
        solution_params: dict[str, Any],
        reference_results_gdx: Any,
        reference_slice: str,
        abs_tol: float,
        rel_tol: float,
    ) -> dict[str, Any]:
        _ = self, solution_vars, solution_params, reference_results_gdx, abs_tol, rel_tol
        return {"passed": True, "gams_slice": reference_slice}

    def _fake_key(self: PepAdapter, vars_obj: Any) -> dict[str, float]:
        _ = self
        return {
            "GO": float(vars_obj["GO"]),
            "ttix_sum": float(sum(vars_obj["ttix"].values())),
            "PWM_sum": float(sum(vars_obj["PWM"].values())),
        }

    monkeypatch.setattr(PepAdapter, "fit_base_state", _fake_fit)
    monkeypatch.setattr(PepAdapter, "solve_state", _fake_solve)
    monkeypatch.setattr(PepAdapter, "compare_with_reference", _fake_compare)
    monkeypatch.setattr(PepAdapter, "key_indicators", _fake_key)

    sim = PepSimulator().fit()
    report = sim.run_scenarios(
        scenarios=[
            export_tax(multiplier=0.75),
            import_shock(multiplier=1.25),
            government_spending(multiplier=1.2),
        ],
        reference_results_gdx="Results.gdx",
        warm_start=True,
        include_base=True,
    )

    assert report["base"]["comparison"]["passed"] is True
    assert report["base"]["solve"]["key_indicators"]["ttix_sum"] == pytest.approx(0.30)
    assert report["base"]["solve"]["key_indicators"]["PWM_sum"] == pytest.approx(3.00)
    assert report["base"]["solve"]["key_indicators"]["GO"] == pytest.approx(100.0)

    s_export = report["scenarios"][0]
    s_import = report["scenarios"][1]
    s_gov = report["scenarios"][2]
    assert s_export["name"] == "export_tax"
    assert s_import["name"] == "import_shock"
    assert s_gov["name"] == "government_spending"
    assert s_export["solve"]["key_indicators"]["ttix_sum"] == pytest.approx(0.225)
    assert s_import["solve"]["key_indicators"]["PWM_sum"] == pytest.approx(3.75)
    assert s_gov["solve"]["key_indicators"]["GO"] == pytest.approx(120.0)
    assert s_export["comparison"]["gams_slice"] == "sim1"
    assert s_import["comparison"]["gams_slice"] == "sim1"
    assert s_gov["comparison"]["gams_slice"] == "sim1"

    assert len(captured) == 4
    assert captured[0]["initial_vars"] is None
    assert isinstance(captured[1]["initial_vars"], dict)
    assert captured[1]["initial_vars"]["GO"] == 100.0
    assert captured[2]["initial_vars"]["GO"] == 100.0
    assert captured[3]["initial_vars"]["GO"] == 100.0
