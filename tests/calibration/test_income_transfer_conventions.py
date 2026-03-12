from __future__ import annotations

from equilibria.templates.pep_calibration_income import IncomeCalibrator


def test_government_transfer_income_excludes_gvt_self_transfers() -> None:
    calibrator = IncomeCalibrator(
        sam_data={
            "sam_matrix": {
                ("AG", "GVT", "AG", "GVT"): 5.0,
                ("AG", "GVT", "AG", "HRP"): 3.0,
                ("AG", "GVT", "AG", "ROW"): 2.0,
            }
        },
        sets={
            "H": [],
            "F": [],
            "K": [],
            "L": [],
            "J": [],
            "I": [],
            "AG": ["hrp", "gvt", "row"],
            "AGNG": ["hrp", "row"],
            "AGD": ["hrp", "gvt"],
        },
    )

    calibrator._calibrate_government_income()

    assert calibrator.result.YGTRO == 5.0
    assert calibrator.result.YGO == 5.0
