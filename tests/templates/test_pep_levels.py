from __future__ import annotations

from equilibria.templates.pep_levels import (
    EquilibriaLevelsExtractor,
    GAMSLevelsExtractor,
    LevelsComparator,
)
from equilibria.templates.pep_model_equations import PEPModelVariables


def test_equilibria_levels_extractor_reads_aliases() -> None:
    vars_obj = PEPModelVariables()
    vars_obj.PT = {"agr": 1.0007}
    vars_obj.RK = {"cap": 1.0012}
    vars_obj.e = 1.25
    params = {
        "PWX": {"agr": 1.1},
        "ttix": {"agr": 0.075},
    }

    extractor = EquilibriaLevelsExtractor(vars_obj, params)

    assert extractor.get("valPT", ("agr",)) == 1.0007
    assert extractor.get("valRK", ("cap",)) == 1.0012
    assert extractor.get("valPWX", ("agr",)) == 1.1
    assert extractor.get("valttix", ("agr",)) == 0.075
    assert extractor.get("vale", ()) == 1.25


def test_gams_levels_slice_records_filters_by_scenario() -> None:
    records = [
        (("agr", "base"), 1.0),
        (("agr", "sim1"), 1.2),
        (("ser",), 2.0),
    ]

    base = GAMSLevelsExtractor.slice_records(records, "base")
    sim1 = GAMSLevelsExtractor.slice_records(records, "sim1")

    assert base == {("agr",): 1.0, ("ser",): 2.0}
    assert sim1 == {("agr",): 1.2}


def test_levels_comparator_reports_mismatch_and_missing() -> None:
    vars_obj = PEPModelVariables()
    vars_obj.PT = {"agr": 1.0}
    vars_obj.RK = {"cap": 1.0}

    eq = EquilibriaLevelsExtractor(vars_obj, {})
    gams_levels = {
        "valPT": {("agr",): 1.1},
        "valRK": {("cap",): 1.0},
        "valttix": {("agr",): 0.2},
    }

    cmp_ = LevelsComparator(abs_tol=1e-6, rel_tol=1e-6)
    report = cmp_.compare(gams_levels, eq)

    assert report.compared == 2
    assert report.missing_in_equilibria == 1
    assert report.mismatches == 1
    assert report.passed is False
    assert report.top_mismatches[0]["symbol"] == "valPT"
