from __future__ import annotations

from pathlib import Path

from equilibria.babel.gdx.reader import read_gdx
from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_calibration_unified_excel import _build_sam_data_from_excel
from equilibria.templates.pep_dynamic_sets import derive_dynamic_sets_from_sam


PEP2_ROOT = Path("src/equilibria/templates/reference/pep2")


def test_dynamic_sets_gdx_vs_excel_match() -> None:
    sam_gdx = read_gdx(PEP2_ROOT / "data" / "SAM-V2_0.gdx")
    sam_xls = _build_sam_data_from_excel(PEP2_ROOT / "data" / "SAM-V2_0.xls")

    sets_gdx = derive_dynamic_sets_from_sam(sam_gdx)
    sets_xls = derive_dynamic_sets_from_sam(sam_xls)

    assert sets_gdx == sets_xls
    assert sets_gdx["AG"] == ["hrp", "hup", "hrr", "hur", "firm", "gvt", "row"]
    assert sets_gdx["H"] == ["hrp", "hup", "hrr", "hur"]
    assert sets_gdx["F"] == ["firm"]


def test_unified_calibrator_uses_dynamic_sets() -> None:
    calibrator = PEPModelCalibrator(
        sam_file=PEP2_ROOT / "data" / "SAM-V2_0.gdx",
        val_par_file=PEP2_ROOT / "data" / "VAL_PAR.xlsx",
        dynamic_sets=True,
    )
    state = calibrator.calibrate()
    expected = derive_dynamic_sets_from_sam(calibrator.sam_data)
    assert state.sets == expected
