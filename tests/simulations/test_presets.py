from __future__ import annotations

from typing import Any

from equilibria.simulations import (
    PepSimulator,
    available_presets,
    export_tax,
    government_spending,
    import_price,
    import_shock,
    make_preset,
)


def test_export_tax_preset_shape() -> None:
    scenario = export_tax(multiplier=0.8)
    assert scenario.name == "export_tax"
    assert scenario.reference_slice == "sim1"
    assert len(scenario.shocks) == 1
    assert scenario.shocks[0].var == "ttix"
    assert scenario.shocks[0].op == "scale"
    assert scenario.shocks[0].values == {"*": 0.8}


def test_import_price_preset_normalizes_commodity_name() -> None:
    scenario = import_price(commodity="AGR", multiplier=1.1)
    assert scenario.name == "import_price_agr"
    assert scenario.shocks[0].var == "PWM"
    assert scenario.shocks[0].values == {"agr": 1.1}


def test_import_shock_and_government_spending_presets() -> None:
    s1 = import_shock(multiplier=1.3)
    s2 = government_spending(multiplier=1.15)
    assert s1.shocks[0].values == {"*": 1.3}
    assert s2.shocks[0].values == 1.15


def test_preset_registry_and_make_preset() -> None:
    names = available_presets()
    assert "export_tax" in names
    assert "import_shock" in names
    assert make_preset("EXPORT_TAX", multiplier=0.7).name == "export_tax"


def test_pep_simulator_wrapper_delegates_to_run_scenarios(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def _fake_run_scenarios(_self: PepSimulator, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(PepSimulator, "run_scenarios", _fake_run_scenarios)

    sim = PepSimulator()
    out = sim.run_export_tax(
        multiplier=0.72,
        reference_results_gdx="results.gdx",
        include_base=False,
    )

    assert out == {"ok": True}
    assert captured["reference_results_gdx"] == "results.gdx"
    assert captured["include_base"] is False
    scenarios = captured["scenarios"]
    assert len(scenarios) == 1
    assert scenarios[0].name == "export_tax"
    assert scenarios[0].shocks[0].values == {"*": 0.72}


def test_pep_simulator_run_preset_validates_name() -> None:
    sim = PepSimulator()
    try:
        sim.run_preset("does_not_exist")
    except ValueError as exc:
        assert "Unknown preset" in str(exc)
        return
    raise AssertionError("Expected ValueError for unknown preset.")
