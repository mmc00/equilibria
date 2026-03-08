from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from equilibria.simulations import (
    IEEMSimulator,
    Scenario,
    Shock,
    Simulator,
    available_mapping_runtimes,
    clear_mapping_runtime,
    get_mapping_runtime,
    register_mapping_runtime,
)


@pytest.fixture(autouse=True)
def _clear_runtime_registry() -> None:
    clear_mapping_runtime()
    yield
    clear_mapping_runtime()


def test_register_mapping_runtime_autoloads_hooks_for_ieem() -> None:
    def solve_fn(
        state: dict[str, float],
        *,
        initial_vars: object | None,
        reference_results_gdx: Path | None,
        reference_slice: str,
    ) -> tuple[object, object, dict[str, object]]:
        _ = initial_vars, reference_results_gdx
        x = float(state["x"])
        return (
            SimpleNamespace(params={"source": "registry"}),
            SimpleNamespace(
                converged=True,
                iterations=1,
                final_residual=0.0,
                message=f"runtime:{reference_slice}",
                variables={"x": x},
            ),
            {"passed": True, "mode": "runtime"},
        )

    def indicators_fn(vars_obj: object) -> dict[str, float]:
        x = float(vars_obj["x"])
        return {"x": x, "x10": 10.0 * x}

    register_mapping_runtime(
        "ieem",
        solve_fn=solve_fn,
        key_indicators_fn=indicators_fn,
    )

    sim = IEEMSimulator(base_state={"x": 5.0}).fit()
    report = sim.run_scenarios(
        scenarios=[Scenario(name="plus", shocks=[Shock(var="x", op="add", values=2.0)])],
        include_base=True,
    )

    assert "ieem" in available_mapping_runtimes()
    assert get_mapping_runtime("ieem") is not None
    assert report["capabilities"]["has_solver"] is True
    assert report["capabilities"]["has_reference_compare"] is False
    assert report["capabilities"]["mode"] == "state_with_solver_hook"
    assert report["base"]["solve"]["message"] == "runtime:base"
    assert report["base"]["solve"]["key_indicators"]["x10"] == 50.0
    assert report["scenarios"][0]["solve"]["key_indicators"]["x"] == 7.0


def test_explicit_hooks_override_registered_runtime() -> None:
    def registry_solve(
        state: dict[str, float],
        *,
        initial_vars: object | None,
        reference_results_gdx: Path | None,
        reference_slice: str,
    ) -> tuple[object, object, dict[str, object]]:
        _ = state, initial_vars, reference_results_gdx, reference_slice
        return (
            SimpleNamespace(params={"source": "registry"}),
            SimpleNamespace(
                converged=True,
                iterations=1,
                final_residual=0.0,
                message="registry",
                variables={"x": 1.0},
            ),
            {"passed": True, "mode": "registry"},
        )

    def explicit_solve(
        state: dict[str, float],
        *,
        initial_vars: object | None,
        reference_results_gdx: Path | None,
        reference_slice: str,
    ) -> tuple[object, object, dict[str, object]]:
        _ = initial_vars, reference_results_gdx, reference_slice
        return (
            SimpleNamespace(params={"source": "explicit"}),
            SimpleNamespace(
                converged=True,
                iterations=1,
                final_residual=0.0,
                message="explicit",
                variables={"x": float(state["x"])},
            ),
            {"passed": True, "mode": "explicit"},
        )

    register_mapping_runtime("gtap", solve_fn=registry_solve)

    sim = Simulator(
        model="gtap",
        base_state={"x": 9.0},
        solve_fn=explicit_solve,
    ).fit()
    report = sim.run_scenarios(
        scenarios=[Scenario(name="s", shocks=[Shock(var="x", op="set", values=4.0)])],
        include_base=True,
    )

    assert report["base"]["solve"]["message"] == "explicit"
    assert report["base"]["validation"]["mode"] == "explicit"
    assert report["base"]["solve"]["key_indicators"]["x"] == 9.0
    assert report["scenarios"][0]["solve"]["key_indicators"]["x"] == 4.0


def test_register_mapping_runtime_requires_at_least_one_hook() -> None:
    with pytest.raises(ValueError, match="at least one hook"):
        register_mapping_runtime("icio")

