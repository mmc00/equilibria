"""derive_calibration produces a consistent GTAP6 calibration for gtap6_3x3."""

from __future__ import annotations

from pathlib import Path

from equilibria.templates.gtap6.gtap6_calibration import derive_calibration
from equilibria.templates.gtap6.gtap6_parameters import GTAP6Parameters
from equilibria.templates.gtap6.gtap6_sets import GTAP6Sets

DATASET = Path(__file__).resolve().parents[3] / "datasets" / "gtap6_3x3"


def test_derive_calibration_gtap6_3x3():
    sets = GTAP6Sets()
    sets.load_from_har(DATASET / "sets.har", default_path=DATASET / "default.prm")
    params = GTAP6Parameters()
    params.load_from_har(DATASET / "basedata.har", DATASET / "default.prm", sets)

    derived = derive_calibration(sets, params)

    assert derived is not None
    # Regional income must be positive for every region once calibrated.
    for r in sets.r:
        assert derived.y_0[r] > 0.0
