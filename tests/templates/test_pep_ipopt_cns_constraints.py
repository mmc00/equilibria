"""Ensure IPOPT uses CNS-style hard constraints across PEP templates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_calibration_unified_dynamic import (
    PEPModelCalibratorDynamic,
    PEPModelCalibratorDynamicSAM,
    PEPModelCalibratorExcelDynamic,
    PEPModelCalibratorExcelDynamicSAM,
)
from equilibria.templates.pep_calibration_unified_excel import PEPModelCalibratorExcel
from equilibria.templates.pep_model_solver_ipopt import CGEProblem, IPOPTSolver


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


def test_ipopt_bounds_follow_gams_domains_for_nonfixed_variables() -> None:
    state = _build_dynamic_sam_excel()
    solver = IPOPTSolver(state, tolerance=1e-6, max_iterations=1)

    vars0 = solver._create_initial_guess()
    x0 = solver._variables_to_array(vars0)
    lb, ub = solver._build_variable_bounds(x_ref=x0)

    fixed = np.isfinite(lb) & np.isfinite(ub) & np.isclose(lb, ub)
    nonfixed = ~fixed

    assert np.isinf(ub[nonfixed]).any()
    assert np.isneginf(lb[nonfixed]).any()
    assert not np.any(np.isfinite(ub[nonfixed]) & (ub[nonfixed] >= 1e10))


def test_ipopt_square_feasible_messages_are_accepted() -> None:
    assert IPOPTSolver._ipopt_reports_square_feasible("Feasible point for square problem found.")
    assert IPOPTSolver._ipopt_reports_square_feasible("Solved To Acceptable Level")
    assert not IPOPTSolver._ipopt_reports_square_feasible(
        "Maximum number of iterations exceeded"
    )


def test_ipopt_problem_uses_constant_feasibility_objective() -> None:
    state = _build_dynamic_sam_excel()
    solver = IPOPTSolver(state, tolerance=1e-6, max_iterations=1)

    vars0 = solver._create_initial_guess()
    x0 = solver._variables_to_array(vars0)

    problem = CGEProblem(
        equations=solver.equations,
        sets=solver.sets,
        n_variables=len(x0),
        variable_info={},
        residual_weights=solver._build_residual_weights(),
        hard_constraints=solver._build_hard_constraints(),
    )

    assert problem.objective(x0) == 0.0
    np.testing.assert_allclose(problem.gradient(x0), np.zeros_like(x0))
    assert np.isfinite(problem.jacobian(x0)).all()
