"""The detection/prescription split: a prescription is 'hypothesis' until a MEASURING
run validated the predicted EFFECT (seeded + converged + effect observed). A static
tool's recipe is always hypothesis; an empty/partial validation can NEVER mark it
confirmed. This guards the rule that cost 3 holdfix attempts when recipes were shown
as verdicts."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from _parity_json import (make_detection, make_prescription, make_validation,
                          with_diagnosis, make_violation)


def test_static_recipe_is_hypothesis():
    p = make_prescription("freeze pf", validated_by=None)
    assert p["status"] == "hypothesis"
    assert p["validated_by"] is None


def test_ran_but_not_converged_stays_hypothesis():
    # a measuring tool ran but code=2 → make_validation returns None → not confirmed
    v = make_validation(tool="probe", converged=False, predicted_effect="match up",
                        predicted_effect_observed=False, evidence="code=2")
    assert v is None
    p = make_prescription("freeze pf", validated_by=v)
    assert p["status"] == "hypothesis"


def test_converged_but_effect_absent_stays_hypothesis():
    # THE pft=1.0 case: code=1 but the predicted effect (match up) did NOT happen
    v = make_validation(tool="probe", converged=True, predicted_effect="match>73%",
                        predicted_effect_observed=False, evidence="code=1 but match 60.7%")
    assert v is None  # predicted_effect_observed False → no validation record
    p = make_prescription("freeze pf", validated_by=v)
    assert p["status"] == "hypothesis"


def test_all_three_confirms():
    v = make_validation(tool="probe", converged=True, predicted_effect="match>73%",
                        predicted_effect_observed=True, evidence="code=1, match 99%")
    assert v is not None
    p = make_prescription("replicate free-row", validated_by=v)
    assert p["status"] == "confirmed"
    assert p["validated_by"]["tool"] == "probe"


def test_partial_validation_dict_cannot_forge_confirmed():
    # a hand-built dict missing predicted_effect_observed must NOT confirm
    forged = {"tool": "x", "converged": True}  # no predicted_effect_observed
    p = make_prescription("x", validated_by=forged)
    assert p["status"] == "hypothesis"
    assert p["validated_by"] is None


def test_detection_can_be_firm_independent_of_prescription():
    v = make_violation("pf", [], "holdfix_missing", 1.0)
    with_diagnosis(
        v,
        detection=make_detection("GAMS freezes pf", "var.fx(tsim-1)", confidence="firm"),
        prescription=make_prescription("freeze pf in Python", validated_by=None),
    )
    assert v["detection"]["confidence"] == "firm"      # observation firm
    assert v["prescription"]["status"] == "hypothesis"  # recipe unproven
