from __future__ import annotations

import pytest

from equilibria.templates.pep_model_equations import PEPModelEquations, PEPModelVariables


def _minimal_sets() -> dict[str, list[str]]:
    return {
        "I": ["agr", "food"],
        "H": ["h1", "h2"],
        "J": [],
        "K": [],
        "L": [],
    }


def test_eq81_uses_benchmark_consumption_weights_co0() -> None:
    sets = _minimal_sets()
    params = {
        "PCO0": {"agr": 2.0, "food": 1.0},
        "CO0": {
            ("agr", "h1"): 10.0,
            ("agr", "h2"): 5.0,
            ("food", "h1"): 2.0,
            ("food", "h2"): 3.0,
        },
    }
    eqs = PEPModelEquations(sets, params)

    vars_obj = PEPModelVariables()
    vars_obj.PC = {"agr": 3.0, "food": 2.0}

    # Deliberately different from CO0 to ensure EQ81 does not use current C(i,h).
    vars_obj.C = {
        ("agr", "h1"): 1.0,
        ("agr", "h2"): 1.0,
        ("food", "h1"): 60.0,
        ("food", "h2"): 40.0,
    }

    # Expected ratio with CO0 weights: (3*15 + 2*5) / (2*15 + 1*5) = 55/35.
    vars_obj.PIXCON = 55.0 / 35.0

    residuals = eqs.price_residuals(vars_obj)
    assert residuals["EQ81"] == pytest.approx(0.0, abs=1e-12)


def test_eq81_falls_back_to_current_consumption_when_co0_missing() -> None:
    sets = _minimal_sets()
    params = {
        "PCO0": {"agr": 2.0, "food": 1.0},
    }
    eqs = PEPModelEquations(sets, params)

    vars_obj = PEPModelVariables()
    vars_obj.PC = {"agr": 3.0, "food": 2.0}
    vars_obj.C = {
        ("agr", "h1"): 1.0,
        ("agr", "h2"): 1.0,
        ("food", "h1"): 60.0,
        ("food", "h2"): 40.0,
    }

    # Expected ratio with C weights: (3*2 + 2*100) / (2*2 + 1*100) = 206/104.
    vars_obj.PIXCON = 206.0 / 104.0

    residuals = eqs.price_residuals(vars_obj)
    assert residuals["EQ81"] == pytest.approx(0.0, abs=1e-12)
