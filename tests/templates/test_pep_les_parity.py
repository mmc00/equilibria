"""LES parity checks against pep2 GAMS baseline."""

from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values
from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_calibration_unified_dynamic import (
    PEPModelCalibratorDynamic,
    PEPModelCalibratorDynamicSAM,
    PEPModelCalibratorExcelDynamic,
)
from equilibria.templates.pep_calibration_unified_excel import PEPModelCalibratorExcel
from equilibria.templates.pep_val_par_loader import load_val_par


ROOT = Path(__file__).resolve().parents[2]
PEP2 = ROOT / "src/equilibria/templates/reference/pep2"
PEP2_DATA = PEP2 / "data"
PEP2_RESULTS = PEP2 / "scripts" / "Results_ipopt.gdx"
PEP2_SAM_CONNECT = PEP2_DATA / "SAM-V2_0_connect.xlsx"
PEP2_SAM_XLSX = PEP2_DATA / "SAM-V2_0.xlsx"


def _resolve_excel_sam_file() -> Path:
    if PEP2_SAM_XLSX.exists():
        return PEP2_SAM_XLSX
    if PEP2_SAM_CONNECT.exists():
        return PEP2_SAM_CONNECT
    pytest.skip("Excel SAM baseline not available")


def _build_base_gdx() -> object:
    c = PEPModelCalibrator(
        sam_file=PEP2_DATA / "SAM-V2_0.gdx",
        val_par_file=PEP2_DATA / "VAL_PAR.xlsx",
        dynamic_sets=False,
    )
    return c.calibrate()


def _build_base_excel() -> object:
    c = PEPModelCalibratorExcel(
        sam_file=_resolve_excel_sam_file(),
        val_par_file=PEP2_DATA / "VAL_PAR.xlsx",
        dynamic_sets=False,
    )
    return c.calibrate()


def _build_dynamic_gdx() -> object:
    c = PEPModelCalibratorDynamic(
        sam_file=PEP2_DATA / "SAM-V2_0.gdx",
        val_par_file=PEP2_DATA / "VAL_PAR.xlsx",
    )
    return c.calibrate()


def _build_dynamic_excel() -> object:
    c = PEPModelCalibratorExcelDynamic(
        sam_file=_resolve_excel_sam_file(),
        val_par_file=PEP2_DATA / "VAL_PAR.xlsx",
    )
    return c.calibrate()


def _build_dynamic_sam_gdx() -> object:
    c = PEPModelCalibratorDynamicSAM(
        sam_file=PEP2_DATA / "SAM-V2_0.gdx",
        val_par_file=PEP2_DATA / "VAL_PAR.xlsx",
    )
    return c.calibrate()


def _expected_cmin_base() -> dict[tuple[str, str], float]:
    if not PEP2_RESULTS.exists():
        pytest.skip("Results_ipopt.gdx baseline not available")
    gdx = read_gdx(PEP2_RESULTS)
    vals = read_parameter_values(gdx, "valCMIN")
    out: dict[tuple[str, str], float] = {}
    for (i, h, scen), v in vals.items():
        if str(scen).upper() == "BASE":
            out[(str(i).lower(), str(h).lower())] = float(v)
    return out


def test_val_par_loader_reads_original_les_layout() -> None:
    d = load_val_par(PEP2_DATA / "VAL_PAR.xlsx")
    assert all(abs(d["frisch"][h] + 1.5) < 1e-12 for h in ["hrp", "hup", "hrr", "hur"])
    assert len(d["sigma_Y"]) >= 20
    assert abs(d["sigma_Y"][("agr", "hrp")] - 0.7) < 1e-12


@pytest.mark.parametrize(
    "state_builder",
    [
        _build_base_gdx,
        _build_base_excel,
        _build_dynamic_gdx,
        _build_dynamic_excel,
        _build_dynamic_sam_gdx,
    ],
)
@pytest.mark.skipif(not PEP2_RESULTS.exists(), reason="Results_ipopt.gdx baseline not available")
def test_les_parity_cmin_and_frisch(state_builder) -> None:
    state = state_builder()
    les = state.les_parameters

    frisch = les["frisch"]
    assert all(abs(frisch[h] + 1.5) < 1e-9 for h in ["hrp", "hup", "hrr", "hur"])

    expected_cmin = _expected_cmin_base()
    got_cmin = les["CMINO"]

    # Compare only keys present in Results.gdx BASE slice.
    diffs = []
    for key, exp in expected_cmin.items():
        got = float(got_cmin[key])
        diffs.append(abs(got - exp))
    assert diffs
    assert max(diffs) < 1e-6
