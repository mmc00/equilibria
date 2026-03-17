from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from equilibria.simulations import Scenario, Shock, Simulator, register_adapter
from equilibria.simulations.adapters.base import BaseModelAdapter
from equilibria.simulations.types import ShockDefinition


@dataclass
class _FakeSolution:
    converged: bool = True
    iterations: int = 1
    final_residual: float = 0.0
    message: str = "ok"
    variables: dict[str, float] = field(default_factory=dict)


class _FakeSolver:
    def __init__(self) -> None:
        self.params = {"dummy": 1.0}


class _FakeAdapter(BaseModelAdapter):
    def __init__(self, *, offset: float = 0.0) -> None:
        self.offset = float(offset)
        self.solve_initials: list[Any] = []

    def fit_base_state(self) -> dict[str, float]:
        return {"x": 10.0}

    def available_shocks(self) -> list[ShockDefinition]:
        return [
            ShockDefinition(
                var="X",
                kind="scalar",
                domain=None,
                ops=("set", "scale", "add"),
                description="Test scalar.",
            )
        ]

    def apply_shock(self, state: dict[str, float], shock: Shock) -> None:
        if shock.var.strip().lower() != "x":
            raise ValueError("Only X is supported in fake adapter.")
        current = float(state.get("x", 0.0))
        value = float(shock.values)
        if shock.op == "set":
            state["x"] = value
            return
        if shock.op == "scale":
            state["x"] = current * value
            return
        if shock.op == "add":
            state["x"] = current + value
            return
        raise ValueError(f"Unsupported op: {shock.op}")

    def solve_state(
        self,
        state: dict[str, float],
        *,
        initial_vars: Any | None,
        reference_results_gdx: Path | None,
        reference_slice: str,
        scenario: Scenario | None = None,
    ) -> tuple[Any, Any, dict[str, Any]]:
        _ = reference_results_gdx, reference_slice, scenario
        self.solve_initials.append(initial_vars)
        x = float(state.get("x", 0.0)) + self.offset
        solution = _FakeSolution(
            converged=True,
            iterations=len(self.solve_initials),
            final_residual=0.0,
            message="ok",
            variables={"x": x},
        )
        return _FakeSolver(), solution, {"passed": True}

    def compare_with_reference(
        self,
        *,
        solution_vars: Any,
        solution_params: dict[str, Any],
        reference_results_gdx: Path,
        reference_slice: str,
        abs_tol: float,
        rel_tol: float,
    ) -> dict[str, Any]:
        _ = solution_params, reference_results_gdx, abs_tol, rel_tol
        return {
            "passed": True,
            "slice": reference_slice,
            "x": float(solution_vars["x"]),
        }

    def key_indicators(self, vars_obj: Any) -> dict[str, float]:
        return {"x": float(vars_obj["x"])}


class _FailingScenarioAdapter(_FakeAdapter):
    def solve_state(
        self,
        state: dict[str, float],
        *,
        initial_vars: Any | None,
        reference_results_gdx: Path | None,
        reference_slice: str,
        scenario: Scenario | None = None,
    ) -> tuple[Any, Any, dict[str, Any]]:
        _ = reference_results_gdx, reference_slice, scenario
        self.solve_initials.append(initial_vars)
        call_no = len(self.solve_initials)
        converged = call_no != 2
        x = float(state.get("x", 0.0)) + self.offset
        if not converged:
            x = 999.0
        solution = _FakeSolution(
            converged=converged,
            iterations=call_no,
            final_residual=0.0 if converged else 1.0,
            message="ok" if converged else "failed",
            variables={"x": x},
        )
        return _FakeSolver(), solution, {"passed": converged}


def test_simulator_fit_and_available_shocks() -> None:
    register_adapter("fake", _FakeAdapter)
    sim = Simulator(model="fake", offset=2.5).fit()
    shocks = sim.available_shocks()
    assert len(shocks) == 1
    assert shocks[0].var == "X"
    assert shocks[0].ops == ("set", "scale", "add")


def test_simulator_run_scenarios_warm_start_and_reference() -> None:
    register_adapter("fake", _FakeAdapter)
    sim = Simulator(model="fake", offset=2.5).fit()

    report = sim.run_scenarios(
        scenarios=[
            Scenario(
                name="plus_two",
                shocks=[Shock(var="X", op="add", values=2.0)],
                closure={"fixed": ["X"]},
            ),
            Scenario(name="half", shocks=[Shock(var="X", op="scale", values=0.5)]),
        ],
        reference_results_gdx="output/Results.gdx",
        warm_start=True,
    )

    assert report["model"] == "fake"
    assert report["capabilities"]["has_solver"] is True
    assert report["capabilities"]["has_reference_compare"] is True
    assert report["base"]["solve"]["key_indicators"]["x"] == pytest.approx(12.5)
    assert report["base"]["comparison"]["passed"] is True

    scenarios = report["scenarios"]
    assert len(scenarios) == 2
    assert scenarios[0]["name"] == "plus_two"
    assert scenarios[0]["closure"] == {"fixed": ["X"]}
    assert scenarios[0]["solve"]["key_indicators"]["x"] == pytest.approx(14.5)
    assert scenarios[0]["comparison"]["slice"] == "sim1"
    assert scenarios[1]["name"] == "half"
    assert scenarios[1]["solve"]["key_indicators"]["x"] == pytest.approx(7.5)

    adapter = sim.adapter
    assert isinstance(adapter, _FakeAdapter)
    assert adapter.solve_initials[0] is None
    assert adapter.solve_initials[1] == {"x": 12.5}
    assert adapter.solve_initials[2] == {"x": 14.5}


def test_simulator_warm_start_uses_last_converged_solution_only() -> None:
    register_adapter("fake_failing", _FailingScenarioAdapter)
    sim = Simulator(model="fake_failing").fit()

    report = sim.run_scenarios(
        scenarios=[
            Scenario(name="fails", shocks=[Shock(var="X", op="add", values=2.0)]),
            Scenario(name="after_fail", shocks=[Shock(var="X", op="scale", values=0.5)]),
        ],
        warm_start=True,
    )

    assert report["base"]["solve"]["converged"] is True
    assert report["scenarios"][0]["solve"]["converged"] is False
    assert report["scenarios"][1]["solve"]["converged"] is True

    adapter = sim.adapter
    assert isinstance(adapter, _FailingScenarioAdapter)
    assert adapter.solve_initials[0] is None
    assert adapter.solve_initials[1] == {"x": 10.0}
    assert adapter.solve_initials[2] == {"x": 10.0}


def test_simulator_rejects_duplicate_scenario_names() -> None:
    register_adapter("fake", _FakeAdapter)
    sim = Simulator(model="fake").fit()

    with pytest.raises(ValueError, match="Duplicate scenario name"):
        sim.run_scenarios(
            scenarios=[
                Scenario(name="shock", shocks=[Shock(var="X", op="set", values=1.0)]),
                Scenario(name="SHOCK", shocks=[Shock(var="X", op="set", values=2.0)]),
            ],
            include_base=False,
        )


def test_simulator_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        Simulator(model="does-not-exist")
