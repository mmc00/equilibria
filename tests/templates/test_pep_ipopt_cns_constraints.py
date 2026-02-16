"""Ensure IPOPT uses CNS-style hard constraints across PEP templates."""

from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_calibration_unified_dynamic import (
    PEPModelCalibratorDynamic,
    PEPModelCalibratorDynamicSAM,
    PEPModelCalibratorExcelDynamic,
    PEPModelCalibratorExcelDynamicSAM,
)
from equilibria.templates.pep_calibration_unified_excel import PEPModelCalibratorExcel
from equilibria.templates.pep_model_solver_ipopt import IPOPTSolver


ROOT = Path(__file__).resolve().parents[2]
PEP2_DATA = ROOT / "src/equilibria/templates/reference/pep2/data"


def _build_base_gdx() -> object:
    c = PEPModelCalibrator(
        sam_file=PEP2_DATA / "SAM-V2_0.gdx",
        val_par_file=PEP2_DATA / "VAL_PAR.gdx",
        dynamic_sets=False,
    )
    return c.calibrate()


def _build_base_excel() -> object:
    c = PEPModelCalibratorExcel(
        sam_file=PEP2_DATA / "SAM-V2_0_connect.xlsx",
        val_par_file=PEP2_DATA / "VAL_PAR.xlsx",
        dynamic_sets=False,
    )
    return c.calibrate()


def _build_dynamic_gdx() -> object:
    c = PEPModelCalibratorDynamic(
        sam_file=PEP2_DATA / "SAM-V2_0.gdx",
        val_par_file=PEP2_DATA / "VAL_PAR.gdx",
    )
    return c.calibrate()


def _build_dynamic_excel() -> object:
    c = PEPModelCalibratorExcelDynamic(
        sam_file=PEP2_DATA / "SAM-V2_0_connect.xlsx",
        val_par_file=PEP2_DATA / "VAL_PAR.xlsx",
    )
    return c.calibrate()


def _build_dynamic_sam_gdx() -> object:
    c = PEPModelCalibratorDynamicSAM(
        sam_file=PEP2_DATA / "SAM-V2_0.gdx",
        val_par_file=PEP2_DATA / "VAL_PAR.gdx",
    )
    return c.calibrate()


def _build_dynamic_sam_excel() -> object:
    c = PEPModelCalibratorExcelDynamicSAM(
        sam_file=PEP2_DATA / "SAM-CRI-gams-fixed.xlsx",
        val_par_file=PEP2_DATA / "VAL_PAR-CRI-gams.xlsx",
    )
    return c.calibrate()


@pytest.mark.parametrize(
    "state_builder",
    [
        _build_base_gdx,
        _build_base_excel,
        _build_dynamic_gdx,
        _build_dynamic_excel,
        _build_dynamic_sam_gdx,
        _build_dynamic_sam_excel,
    ],
)
def test_ipopt_hard_constraints_cover_all_equations(state_builder) -> None:
    state = state_builder()
    solver = IPOPTSolver(state, tolerance=1e-6, max_iterations=1)

    vars0 = solver._create_initial_guess()
    residuals = solver.equations.calculate_all_residuals(vars0)
    hard = solver._build_hard_constraints()

    assert len(residuals) > 200
    assert len(hard) == len(residuals)
    assert set(hard) == set(residuals.keys())
