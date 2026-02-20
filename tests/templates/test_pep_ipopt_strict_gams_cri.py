"""Regression checks for CRI gams initialization behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.templates.pep_calibration_unified_dynamic import (
    PEPModelCalibratorExcelDynamicSAM,
)
from equilibria.templates.pep_model_solver import PEPModelSolver

ROOT = Path(__file__).resolve().parents[2]
PEP2_DATA = ROOT / "src/equilibria/templates/reference/pep2/data"
PEP2_SCRIPTS = ROOT / "src/equilibria/templates/reference/pep2/scripts"


def test_cri_gams_requires_compatible_baseline() -> None:
    calibrator = PEPModelCalibratorExcelDynamicSAM(
        sam_file=PEP2_DATA / "SAM-CRI-gams-fixed.xlsx",
        val_par_file=PEP2_DATA / "VAL_PAR-CRI-gams.xlsx",
    )
    state = calibrator.calibrate()

    solver = PEPModelSolver(
        calibrated_state=state,
        tolerance=1e-8,
        max_iterations=200,
        init_mode="gams",
        gams_results_gdx=PEP2_SCRIPTS / "Results_ipopt_excel_reference.gdx",
        gams_results_slice="sim1",
    )

    with pytest.raises(RuntimeError):
        solver.solve(method="ipopt")
