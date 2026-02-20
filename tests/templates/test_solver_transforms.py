"""Tests for canonical solver transforms and guards."""

from __future__ import annotations

import numpy as np

from equilibria.solver.guards import rebuild_tax_detail_from_rates
from equilibria.solver.transforms import pep_array_to_variables, pep_variables_to_array
from equilibria.templates.pep_model_equations import PEPModelVariables


def _sets() -> dict[str, list[str]]:
    return {
        "J": ["agr", "ind"],
        "I": ["agr", "ser"],
        "H": ["hrp"],
        "F": ["firm"],
        "AG": ["hrp", "firm", "gvt"],
        "K": ["cap"],
        "L": ["usk"],
    }


def test_transform_roundtrip_preserves_vector_order() -> None:
    sets = _sets()
    base_vars = PEPModelVariables()
    template = pep_variables_to_array(base_vars, sets)
    test_vec = np.linspace(0.2, float(len(template)) + 0.2, len(template))

    unpacked = pep_array_to_variables(test_vec, sets)
    roundtrip = pep_variables_to_array(unpacked, sets)

    assert np.allclose(test_vec, roundtrip)


def test_transform_applies_price_floor() -> None:
    sets = _sets()
    base_vars = PEPModelVariables()
    template = pep_variables_to_array(base_vars, sets)
    test_vec = np.linspace(0.2, float(len(template)) + 0.2, len(template))
    test_vec[0] = -5.0  # first packed slot = WC('agr')

    unpacked = pep_array_to_variables(test_vec, sets)
    roundtrip = pep_variables_to_array(unpacked, sets)

    assert roundtrip[0] == 0.1


def test_rebuild_tax_detail_from_rates() -> None:
    sets = _sets()
    vars = PEPModelVariables()
    vars.W = {"usk": 2.0}
    vars.LD = {("usk", "agr"): 3.0, ("usk", "ind"): 4.0}
    vars.R = {("cap", "agr"): 5.0, ("cap", "ind"): 6.0}
    vars.KD = {("cap", "agr"): 7.0, ("cap", "ind"): 8.0}
    vars.PP = {"agr": 9.0, "ind": 10.0}
    vars.XST = {"agr": 11.0, "ind": 12.0}
    vars.TIP = {"agr": 123.0, "ind": 456.0}

    params = {
        "ttiw": {("usk", "agr"): 0.1, ("usk", "ind"): 0.2},
        "ttik": {("cap", "agr"): 0.3, ("cap", "ind"): 0.4},
        "ttip": {"agr": 0.5, "ind": 0.6},
    }

    rebuild_tax_detail_from_rates(vars, sets, params, include_tip=False)
    assert vars.TIW[("usk", "agr")] == 0.1 * 2.0 * 3.0
    assert vars.TIW[("usk", "ind")] == 0.2 * 2.0 * 4.0
    assert vars.TIK[("cap", "agr")] == 0.3 * 5.0 * 7.0
    assert vars.TIK[("cap", "ind")] == 0.4 * 6.0 * 8.0
    assert vars.TIP["agr"] == 123.0
    assert vars.TIP["ind"] == 456.0

    rebuild_tax_detail_from_rates(vars, sets, params, include_tip=True)
    assert vars.TIP["agr"] == 0.5 * 9.0 * 11.0
    assert vars.TIP["ind"] == 0.6 * 10.0 * 12.0

