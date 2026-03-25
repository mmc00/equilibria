from __future__ import annotations

import pytest

from equilibria.templates.pep_co2_model_equations import PEPCO2ModelEquations
from equilibria.templates.pep_co2_model_solver import PEPCO2IPOPTSolver
from equilibria.templates.pep_model_equations import PEPModelVariables


def test_pep_co2_overrides_eq39_and_eq66_with_specific_carbon_tax() -> None:
    eqs = PEPCO2ModelEquations(
        sets={"J": ["agr"], "I": [], "H": [], "F": [], "K": [], "L": [], "AGNG": []},
        parameters={
            "ttip": {"agr": 0.10},
            "co2_intensity": {"agr": 2.0},
            "tco2b": {"agr": 3.0},
            "tco2scal": 1.5,
        },
    )

    vars_obj = PEPModelVariables()
    vars_obj.PP["agr"] = 4.0
    vars_obj.XST["agr"] = 10.0
    vars_obj.PIXCON = 2.0
    vars_obj.PT["agr"] = 22.4
    vars_obj.TIP["agr"] = 184.0

    price_residuals = eqs.price_residuals(vars_obj)
    government_residuals = eqs.government_residuals(vars_obj)

    assert price_residuals["EQ66_agr"] == pytest.approx(0.0, abs=1e-12)
    assert government_residuals["EQ39_agr"] == pytest.approx(0.0, abs=1e-12)


def test_pep_co2_sync_leaves_macro_block_untouched_when_tax_is_inactive() -> None:
    solver = PEPCO2IPOPTSolver.__new__(PEPCO2IPOPTSolver)
    solver.sets = {"J": ["agr"], "AGNG": []}
    solver.params = {
        "co2_intensity": {"agr": 2.0},
        "tco2b": {"agr": 0.0},
        "tco2scal": 1.0,
    }

    vars_obj = PEPModelVariables()
    vars_obj.PIXCON = 1.0
    vars_obj.XST["agr"] = 4.0
    vars_obj.PT["agr"] = 3.5
    vars_obj.TIP["agr"] = 8.0
    vars_obj.TIPT = 8.0
    vars_obj.TPRODN = 10.0
    vars_obj.YG = 20.0
    vars_obj.SG = 5.0

    solver._sync_co2_block(vars_obj, include_pt=True)

    assert vars_obj.PT["agr"] == pytest.approx(3.5)
    assert vars_obj.TIP["agr"] == pytest.approx(8.0)
    assert vars_obj.TIPT == pytest.approx(8.0)
    assert vars_obj.TPRODN == pytest.approx(10.0)
    assert vars_obj.YG == pytest.approx(20.0)
    assert vars_obj.SG == pytest.approx(5.0)
    assert vars_obj.co2_total_emissions == pytest.approx(8.0)
    assert vars_obj.co2_total_tax == pytest.approx(0.0)
