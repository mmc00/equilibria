from __future__ import annotations

from pathlib import Path

from equilibria.qa.sam_checks import run_sam_data_contracts, run_sam_qa_from_file


def test_sam_qa_detects_export_value_mismatch_in_synthetic_data() -> None:
    sam_data = {
        "sam_matrix": {
            ("X", "AGR", "AG", "ROW"): 100.0,
            ("J", "AGR", "X", "AGR"): 50.0,
            ("I", "AGR", "X", "AGR"): 10.0,
            ("AG", "GVT", "X", "AGR"): 5.0,
        }
    }
    sets = {
        "I": ["agr"],
        "J": ["agr"],
        "L": ["usk"],
        "K": ["cap"],
        "H": ["hrp"],
        "AG": ["hrp", "gvt", "row"],
    }

    report = run_sam_data_contracts(sam_data, sets=sets)
    checks = {check.code: check for check in report.checks}

    assert not report.passed
    assert not checks["EXP001"].passed
    assert checks["EXP001"].failures == 1


def test_sam_qa_passes_for_pep2_reference_excel() -> None:
    sam_file = Path("src/equilibria/templates/reference/pep2/data/SAM-V2_0.xls")
    assert sam_file.exists()

    report = run_sam_qa_from_file(sam_file)
    assert report.passed


def test_sam_qa_fails_for_known_cri_unfixed_excel() -> None:
    sam_file = Path("src/equilibria/templates/reference/pep2/data/SAM-CRI-gams.xlsx")
    assert sam_file.exists()

    report = run_sam_qa_from_file(sam_file, gdp_rel_tol=0.08)
    checks = {check.code: check for check in report.checks}

    assert not report.passed
    assert not checks["EXP001"].passed
