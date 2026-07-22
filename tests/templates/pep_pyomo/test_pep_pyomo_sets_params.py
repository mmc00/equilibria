"""Foundation test: PEPSets + PEPParams load faithfully from a calibrated PEPModelState."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SAM = ROOT / "src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx"
VALPAR = ROOT / "src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx"


@pytest.fixture(scope="module")
def state():
    from equilibria.templates.pep_calibration_unified import PEPModelCalibrator

    return PEPModelCalibrator(sam_file=SAM, val_par_file=VALPAR).calibrate()


@pytest.mark.skipif(not SAM.exists(), reason="pep2 SAM not present")
def test_sets_from_state(state):
    from equilibria.templates.pep_pyomo.pep_pyomo_sets import PEPSets

    s = PEPSets.from_state(state)
    assert s.H == ["hrp", "hup", "hrr", "hur"]
    assert s.I == ["agr", "food", "othind", "ser", "adm"]
    assert s.J == ["agr", "ind", "ser", "adm"]
    assert s.walras_i == "agr"
    assert s.I1 == ["food", "othind", "ser", "adm"]  # agr dropped
    assert "gvt" not in s.AGNG and "row" not in s.AGD


@pytest.mark.skipif(not SAM.exists(), reason="pep2 SAM not present")
def test_params_from_state(state):
    from equilibria.templates.pep_pyomo.pep_pyomo_parameters import PEPParams

    p = PEPParams(state)
    # benchmark levels present and indexed
    assert p["io", "agr"] != 0.0 or p["io", "ind"] != 0.0  # Leontief io share exists
    assert isinstance(p["KDO"], dict)
    # derived params recomputed
    gi = p["gamma_INV"]
    assert abs(sum(gi.values()) - 1.0) < 1e-9  # investment shares sum to 1
    assert p["eta"] == 1.0 and p["kmob"] == 1.0
    # missing name raises (a typo, not a benign zero)
    with pytest.raises(KeyError):
        p["not_a_param"]
