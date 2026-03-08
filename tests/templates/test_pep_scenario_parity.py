from __future__ import annotations

import pytest

from equilibria.templates.pep_calibration_unified import PEPModelState
from equilibria.templates.pep_model_equations import PEPModelVariables
from equilibria.templates.pep_scenario_parity import (
    PEPImportPriceParityRunner,
    PEPScenarioParityRunner,
    get_solution_value,
)


def test_export_tax_shock_scales_ttix_and_aggregates() -> None:
    state = PEPModelState(
        trade={
            "ttixO": {"agr": 0.20, "ser": 0.10},
            "TIXO": {"agr": 100.0, "ser": 50.0},
        },
        income={
            "TICTO": 10.0,
            "TIMTO": 20.0,
            "TIXTO": 150.0,
            "TPRCTSO": 180.0,
            "YGKO": 500.0,
            "TDHTO": 20.0,
            "TDFTO": 30.0,
            "TPRODNO": 40.0,
            "YGTRO": 50.0,
            "YGO": 640.0,
        },
    )

    shocked = PEPScenarioParityRunner._clone_with_export_tax_shock(state, multiplier=0.75)

    assert shocked.trade["ttixO"]["agr"] == pytest.approx(0.15)
    assert shocked.trade["ttixO"]["ser"] == pytest.approx(0.075)
    assert shocked.trade["TIXO"]["agr"] == 75.0
    assert shocked.trade["TIXO"]["ser"] == 37.5
    assert shocked.income["TIXTO"] == 112.5
    assert shocked.income["TPRCTSO"] == 142.5
    assert shocked.income["YGO"] == 782.5

    # Original state must remain unchanged.
    assert state.trade["ttixO"]["agr"] == 0.20
    assert state.trade["TIXO"]["agr"] == 100.0
    assert state.income["TIXTO"] == 150.0


def test_get_solution_value_alias_mapping() -> None:
    vars_obj = PEPModelVariables()
    vars_obj.e = 1.25
    vars_obj.PT = {"agr": 1.0007}
    vars_obj.RK = {"cap": 1.0012}
    params = {
        "ttix": {"agr": 0.075},
        "PWX": {"agr": 1.1},
        "PT": {"agr": 1.0},
    }

    assert get_solution_value(vars_obj, "valttix", ("agr",), params) == 0.075
    assert get_solution_value(vars_obj, "valPWX", ("agr",), params) == 1.1
    assert get_solution_value(vars_obj, "valPT", ("agr",), params) == 1.0007
    assert get_solution_value(vars_obj, "valRK", ("cap",), params) == 1.0012
    assert get_solution_value(vars_obj, "vale", (), params) == 1.25


def test_export_tax_homotopy_path_default() -> None:
    runner = PEPScenarioParityRunner(
        export_tax_multiplier=0.75,
        export_tax_homotopy=True,
        export_tax_homotopy_steps=5,
    )
    assert runner._should_use_export_tax_homotopy() is True
    assert runner._build_export_tax_homotopy_path() == pytest.approx(
        [0.95, 0.9, 0.85, 0.8, 0.75]
    )


def test_export_tax_homotopy_disabled_on_non_excel_or_non_ipopt() -> None:
    runner_ipopt_gams = PEPScenarioParityRunner(
        init_mode="gams",
        method="ipopt",
        export_tax_homotopy=True,
    )
    runner_simple_excel = PEPScenarioParityRunner(
        init_mode="excel",
        method="auto",
        export_tax_homotopy=True,
    )
    assert runner_ipopt_gams._should_use_export_tax_homotopy() is False
    assert runner_simple_excel._should_use_export_tax_homotopy() is False


def test_import_price_shock_scales_pwmo_target_only() -> None:
    state = PEPModelState(
        trade={
            "PWMO": {"agr": 1.0, "food": 2.0},
        }
    )

    shocked = PEPImportPriceParityRunner._clone_with_import_price_shock(
        state,
        commodity="agr",
        multiplier=1.25,
    )

    assert shocked.trade["PWMO"]["agr"] == pytest.approx(1.25)
    assert shocked.trade["PWMO"]["food"] == pytest.approx(2.0)
    assert state.trade["PWMO"]["agr"] == pytest.approx(1.0)


def test_import_price_runner_defaults() -> None:
    runner = PEPImportPriceParityRunner()
    assert runner.import_price_commodity == "agr"
    assert runner.import_price_multiplier == pytest.approx(1.25)
