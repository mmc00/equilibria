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


def test_pep_co2_sync_updates_pt_tip_when_tax_is_active() -> None:
    solver = PEPCO2IPOPTSolver.__new__(PEPCO2IPOPTSolver)
    solver.sets = {"J": ["agr"], "AGNG": []}
    solver.params = {
        "ttip": {"agr": 0.10},
        "co2_intensity": {"agr": 2.0},
        "tco2b": {"agr": 3.0},
        "tco2scal": 1.5,
    }

    vars_obj = PEPModelVariables()
    vars_obj.PP["agr"] = 4.0
    vars_obj.XST["agr"] = 10.0
    vars_obj.PIXCON = 2.0
    vars_obj.PT["agr"] = 0.0
    vars_obj.TIP["agr"] = 0.0
    vars_obj.TIWT = 1.0
    vars_obj.TIKT = 2.0
    vars_obj.TDHT = 3.0
    vars_obj.TDFT = 4.0
    vars_obj.TPRCTS = 5.0
    vars_obj.YGK = 6.0
    vars_obj.YGTR = 7.0
    vars_obj.G = 8.0

    solver._sync_co2_block(vars_obj, include_pt=True)

    assert vars_obj.PT["agr"] == pytest.approx(22.4)
    assert vars_obj.TIP["agr"] == pytest.approx(184.0)
    assert vars_obj.TIPT == pytest.approx(184.0)
    assert vars_obj.TPRODN == pytest.approx(187.0)
    assert vars_obj.YG == pytest.approx(212.0)
    assert vars_obj.SG == pytest.approx(204.0)
    assert vars_obj.co2_total_tax == pytest.approx(180.0)


def test_pep_co2_prepare_warm_start_syncs_macro_block() -> None:
    solver = PEPCO2IPOPTSolver.__new__(PEPCO2IPOPTSolver)
    solver.sets = {
        "J": ["trans"],
        "I": ["road"],
        "I1": ["road"],
        "AGNG": [],
        "AGD": [],
        "H": [],
        "K": [],
        "L": [],
    }
    solver.params = {
        "ttip": {"trans": 0.10},
        "co2_intensity": {"trans": 2.0},
        "tco2b": {"trans": 0.50},
        "tco2scal": 1.0,
        "gamma_INV": {"road": 0.25},
    }
    solver.blockwise_macro_alpha = 1.0
    solver._enforce_fixed_closure_levels = lambda vars_obj: None

    vars_obj = PEPModelVariables()
    vars_obj.PP["trans"] = 1.0
    vars_obj.XST["trans"] = 100.0
    vars_obj.PIXCON = 1.0
    vars_obj.PIXINV = 2.0
    vars_obj.PT["trans"] = 1.1
    vars_obj.TIP["trans"] = 11.0
    vars_obj.PC["road"] = 2.0
    vars_obj.P[("trans", "road")] = 3.0
    vars_obj.VSTK["road"] = 3.0
    vars_obj.SH["hh"] = 10.0
    vars_obj.SF["firm"] = 5.0
    vars_obj.SROW = 2.0
    vars_obj.TIWT = 1.0
    vars_obj.TIKT = 2.0
    vars_obj.TDHT = 3.0
    vars_obj.TDFT = 4.0
    vars_obj.TPRCTS = 5.0
    vars_obj.YGK = 6.0
    vars_obj.YGTR = 7.0
    vars_obj.G = 8.0
    vars_obj.IT = 0.0

    solver._prepare_initial_guess_for_solve(vars_obj)

    expected_pt = 2.1
    expected_tip = 110.0
    expected_sg = (6.0 + 3.0 + 4.0 + (1.0 + 2.0 + expected_tip) + 5.0 + 7.0) - 8.0
    expected_it = 10.0 + 5.0 + expected_sg
    expected_gfcf = expected_it - (2.0 * 3.0)
    expected_ratio = expected_pt / 1.1
    expected_inv = 0.25 * expected_gfcf / 2.0
    expected_q = expected_inv + 3.0

    assert vars_obj.PT["trans"] == pytest.approx(expected_pt)
    assert vars_obj.TIP["trans"] == pytest.approx(expected_tip)
    assert vars_obj.P[("trans", "road")] == pytest.approx(3.0 * expected_ratio)
    assert vars_obj.SG == pytest.approx(expected_sg)
    assert vars_obj.IT == pytest.approx(expected_it)
    assert vars_obj.GFCF == pytest.approx(expected_gfcf)
    assert vars_obj.INV["road"] == pytest.approx(expected_inv)
    assert vars_obj.GFCF_REAL == pytest.approx(expected_gfcf / 2.0)
    assert vars_obj.Q["road"] == pytest.approx(expected_q)
