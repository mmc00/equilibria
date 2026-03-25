from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from equilibria.simulations import (
    GTAPSimulator,
    ICIOSimulator,
    IEEMSimulator,
    PepCO2Simulator,
    Scenario,
    Shock,
    Simulator,
    available_models,
)


def test_ieem_adapter_state_scenarios_without_solver() -> None:
    sim = Simulator(
        model="ieem",
        base_state={
            "G": 100.0,
            "PWM": {"agr": 1.0, "ser": 2.0},
        },
    ).fit()

    catalog = {item.var: item for item in sim.available_shocks()}
    assert "G" in catalog
    assert "PWM" in catalog
    assert catalog["PWM"].kind == "indexed"
    assert catalog["PWM"].members == ("agr", "ser")

    report = sim.run_scenarios(
        scenarios=[
            Scenario(
                name="import_shock",
                shocks=[Shock(var="PWM", op="scale", values={"*": 1.25})],
            ),
            Scenario(
                name="gov_spending",
                shocks=[Shock(var="G", op="add", values=10.0)],
            ),
        ],
        include_base=True,
        warm_start=False,
    )

    assert report["capabilities"]["has_solver"] is False
    assert report["capabilities"]["has_reference_compare"] is False
    assert report["capabilities"]["mode"] == "state_only_no_solver"
    assert report["base"]["solve"]["converged"] is True
    assert report["base"]["validation"]["mode"] == "no_solver"
    assert report["base"]["solve"]["key_indicators"]["G"] == 100.0
    assert report["base"]["solve"]["key_indicators"]["PWM_sum"] == 3.0

    s1 = report["scenarios"][0]
    s2 = report["scenarios"][1]
    assert s1["name"] == "import_shock"
    assert s1["solve"]["key_indicators"]["PWM_sum"] == 3.75
    assert s2["name"] == "gov_spending"
    assert s2["solve"]["key_indicators"]["G"] == 110.0


def test_gtap_and_icio_registered() -> None:
    gtap = Simulator(model="gtap", base_state={"tau": {"agr": 0.1}}).fit()
    icio = Simulator(model="icio", base_state={"A": {"c1": 1.0, "c2": 2.0}}).fit()
    assert any(item.var == "tau" for item in gtap.available_shocks())
    assert any(item.var == "A" for item in icio.available_shocks())


def test_no_solver_adapters_report_compare_not_implemented() -> None:
    sim = Simulator(model="gtap", base_state={"x": 1.0}).fit()
    report = sim.run_scenarios(
        scenarios=[Scenario(name="s", shocks=[Shock(var="x", op="set", values=2.0)])],
        reference_results_gdx="dummy.gdx",
    )
    assert report["capabilities"]["has_solver"] is False
    comparison = report["scenarios"][0]["comparison"]
    assert comparison["passed"] is False
    assert "not implemented" in comparison["reason"]


def test_ieem_adapter_supports_native_hook_solver_and_compare() -> None:
    solve_initials: list[object | None] = []
    compare_slices: list[str] = []

    def solve_fn(
        state: dict[str, float],
        *,
        initial_vars: object | None,
        reference_results_gdx: Path | None,
        reference_slice: str,
    ) -> tuple[object, object, dict[str, object]]:
        _ = reference_results_gdx
        solve_initials.append(initial_vars)
        return (
            SimpleNamespace(params={"mode": "hook"}),
            SimpleNamespace(
                converged=True,
                iterations=len(solve_initials),
                final_residual=0.0,
                message=f"hooked:{reference_slice}",
                variables={"x": float(state["x"])},
            ),
            {"passed": True, "mode": "hooked"},
        )

    def compare_fn(
        *,
        solution_vars: object,
        solution_params: dict[str, object],
        reference_results_gdx: Path,
        reference_slice: str,
        abs_tol: float,
        rel_tol: float,
    ) -> dict[str, object]:
        _ = reference_results_gdx, abs_tol, rel_tol
        compare_slices.append(reference_slice)
        return {
            "passed": True,
            "slice": reference_slice,
            "mode": solution_params["mode"],
            "x": float(solution_vars["x"]),
        }

    def key_indicators_fn(vars_obj: object) -> dict[str, float]:
        x = float(vars_obj["x"])
        return {"x": x, "x2": 2.0 * x}

    sim = Simulator(
        model="ieem",
        base_state={"x": 10.0},
        solve_fn=solve_fn,
        compare_fn=compare_fn,
        key_indicators_fn=key_indicators_fn,
    ).fit()

    report = sim.run_scenarios(
        scenarios=[Scenario(name="plus_three", shocks=[Shock(var="x", op="add", values=3.0)])],
        reference_results_gdx="dummy.gdx",
        warm_start=True,
    )

    assert report["capabilities"]["has_solver"] is True
    assert report["capabilities"]["has_reference_compare"] is True
    assert report["capabilities"]["mode"] == "state_with_solver_and_compare_hooks"
    assert report["base"]["validation"]["mode"] == "hooked"
    assert report["base"]["solve"]["message"] == "hooked:base"
    assert report["base"]["solve"]["key_indicators"]["x"] == 10.0
    assert report["base"]["comparison"]["passed"] is True
    assert report["base"]["comparison"]["slice"] == "base"
    assert report["scenarios"][0]["solve"]["key_indicators"]["x"] == 13.0
    assert report["scenarios"][0]["solve"]["key_indicators"]["x2"] == 26.0
    assert report["scenarios"][0]["comparison"]["slice"] == "sim1"
    assert compare_slices == ["base", "sim1"]
    assert solve_initials[0] is None
    assert solve_initials[1] == {"x": 10.0}


def test_gtap_adapter_solver_hook_without_compare_reports_capability() -> None:
    def solve_fn(
        state: dict[str, float],
        *,
        initial_vars: object | None,
        reference_results_gdx: Path | None,
        reference_slice: str,
    ) -> tuple[object, object, dict[str, object]]:
        _ = initial_vars, reference_results_gdx, reference_slice
        return (
            SimpleNamespace(params={}),
            SimpleNamespace(
                converged=True,
                iterations=1,
                final_residual=0.0,
                message="hooked",
                variables={"x": float(state["x"])},
            ),
            {"passed": True, "mode": "hooked"},
        )

    sim = Simulator(model="gtap", base_state={"x": 1.0}, solve_fn=solve_fn).fit()
    report = sim.run_scenarios(
        scenarios=[Scenario(name="s", shocks=[Shock(var="x", op="set", values=2.0)])],
        reference_results_gdx="dummy.gdx",
    )
    assert report["capabilities"]["has_solver"] is True
    assert report["capabilities"]["has_reference_compare"] is False
    assert report["capabilities"]["mode"] == "state_with_solver_hook"
    assert report["scenarios"][0]["comparison"]["passed"] is False
    assert "not implemented" in report["scenarios"][0]["comparison"]["reason"]


def test_available_models_includes_pep_pep_co2_ieem_gtap_icio() -> None:
    models = available_models()
    assert "pep" in models
    assert "pep_co2" in models
    assert "ieem" in models
    assert "gtap" in models
    assert "icio" in models


def test_convenience_simulator_wrappers_for_multi_models() -> None:
    pep_co2 = PepCO2Simulator(co2_intensity={"agr": 1.0})
    ieem = IEEMSimulator(base_state={"x": 1.0}).fit()
    gtap = GTAPSimulator(base_state={"x": 2.0}).fit()
    icio = ICIOSimulator(base_state={"x": 3.0}).fit()

    assert pep_co2.model == "pep_co2"
    assert ieem.model == "ieem"
    assert gtap.model == "gtap"
    assert icio.model == "icio"
