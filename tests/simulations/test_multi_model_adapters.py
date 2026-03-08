from __future__ import annotations

from equilibria.simulations import (
    GTAPSimulator,
    ICIOSimulator,
    IEEMSimulator,
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
    comparison = report["scenarios"][0]["comparison"]
    assert comparison["passed"] is False
    assert "not implemented" in comparison["reason"]


def test_available_models_includes_pep_ieem_gtap_icio() -> None:
    models = available_models()
    assert "pep" in models
    assert "ieem" in models
    assert "gtap" in models
    assert "icio" in models


def test_convenience_simulator_wrappers_for_multi_models() -> None:
    ieem = IEEMSimulator(base_state={"x": 1.0}).fit()
    gtap = GTAPSimulator(base_state={"x": 2.0}).fit()
    icio = ICIOSimulator(base_state={"x": 3.0}).fit()

    assert ieem.model == "ieem"
    assert gtap.model == "gtap"
    assert icio.model == "icio"
