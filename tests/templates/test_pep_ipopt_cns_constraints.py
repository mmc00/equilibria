"""Ensure IPOPT uses CNS-style hard constraints across PEP templates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from equilibria.solver.transforms import pep_array_to_variables
from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_calibration_unified_dynamic import (
    PEPModelCalibratorDynamic,
    PEPModelCalibratorDynamicSAM,
    PEPModelCalibratorExcelDynamic,
    PEPModelCalibratorExcelDynamicSAM,
)
from equilibria.templates.pep_calibration_unified_excel import PEPModelCalibratorExcel
from equilibria.templates.pep_constraint_jacobian import PEPConstraintJacobianHarness
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


def test_ipopt_contract_equation_include_filters_hard_constraints() -> None:
    state = _build_dynamic_sam_excel()
    solver = IPOPTSolver(
        state,
        contract={
            "equations": {
                "include": ["EQ85", "EQ86"],
            }
        },
        tolerance=1e-6,
        max_iterations=1,
    )

    hard = solver._build_hard_constraints()
    expected = set(f"EQ85_{l}" for l in state.sets["L"]) | set(
        f"EQ86_{k}" for k in state.sets["K"]
    )

    assert set(hard) == expected


def test_ipopt_activation_masks_all_active_affect_equations_and_bounds() -> None:
    state = _build_base_gdx()
    default_solver = IPOPTSolver(state, tolerance=1e-6, max_iterations=1)
    all_active_solver = IPOPTSolver(
        state,
        contract={
            "equations": {
                "activation_masks": "all_active",
            }
        },
        tolerance=1e-6,
        max_iterations=1,
    )

    inactive_import = next(
        i for i in state.sets["I"] if abs(default_solver.params.get("IMO0", {}).get(i, 0.0)) <= 1e-12
    )

    default_hard = default_solver._build_hard_constraints()
    all_active_hard = all_active_solver._build_hard_constraints()
    assert f"EQ41_{inactive_import}" not in default_hard
    assert f"EQ41_{inactive_import}" in all_active_hard

    vars0_default = default_solver._create_initial_guess()
    x0_default = default_solver._variables_to_array(vars0_default)
    lb_default, ub_default = default_solver._build_variable_bounds(x_ref=x0_default)
    names_default = default_solver._last_bound_names

    vars0_all = all_active_solver._create_initial_guess()
    x0_all = all_active_solver._variables_to_array(vars0_all)
    lb_all, ub_all = all_active_solver._build_variable_bounds(x_ref=x0_all)
    names_all = all_active_solver._last_bound_names

    pm_name = f"PM[{inactive_import}]"
    pm_idx_default = names_default.index(pm_name)
    pm_idx_all = names_all.index(pm_name)

    assert np.isclose(lb_default[pm_idx_default], ub_default[pm_idx_default])
    assert not np.isclose(lb_all[pm_idx_all], ub_all[pm_idx_all])


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


def test_constraint_harness_preserves_declared_order_and_scaling() -> None:
    state = _build_dynamic_sam_excel()
    solver = IPOPTSolver(state, tolerance=1e-6, max_iterations=1)

    vars0 = solver._create_initial_guess()
    x0 = solver._variables_to_array(vars0)
    hard = solver._build_hard_constraints()[:4]
    harness = PEPConstraintJacobianHarness(
        equations=solver.equations,
        sets=solver.sets,
        n_variables=len(x0),
        hard_constraints=hard,
    )

    residual_dict = solver.equations.calculate_all_residuals(vars0)
    raw = np.array([residual_dict[name] for name in hard], dtype=float)
    expected = raw / np.maximum(np.abs(raw), 1.0)

    np.testing.assert_allclose(harness.evaluate_constraints(x0), expected)
    assert harness.constraint_names == tuple(hard)


def test_constraint_harness_reports_dense_row_major_structure() -> None:
    state = _build_dynamic_sam_excel()
    solver = IPOPTSolver(state, tolerance=1e-6, max_iterations=1)

    vars0 = solver._create_initial_guess()
    x0 = solver._variables_to_array(vars0)
    hard = solver._build_hard_constraints()
    harness = PEPConstraintJacobianHarness(
        equations=solver.equations,
        sets=solver.sets,
        n_variables=len(x0),
        hard_constraints=hard,
        sparsity_reference_x=x0,
    )

    rows, cols = harness.jacobian_structure()
    rows2, cols2 = harness.jacobian_structure()
    values = harness.evaluate_jacobian_values(x0)

    assert rows.shape == cols.shape
    assert values.shape == rows.shape
    assert len(rows) < len(hard) * len(x0)
    np.testing.assert_array_equal(rows, rows2)
    np.testing.assert_array_equal(cols, cols2)
    assert np.all(rows[:-1] <= rows[1:])


def test_constraint_harness_uses_analytic_eq66_derivatives() -> None:
    state = _build_dynamic_sam_excel()
    solver = IPOPTSolver(state, tolerance=1e-6, max_iterations=1)

    vars0 = solver._create_initial_guess()
    x0 = solver._variables_to_array(vars0)
    solver._build_variable_bounds(x_ref=x0)
    names = solver._last_bound_names
    harness = PEPConstraintJacobianHarness(
        equations=solver.equations,
        sets=solver.sets,
        n_variables=len(x0),
        hard_constraints=["EQ66_agr"],
        variable_names=names,
        sparsity_reference_x=x0,
    )

    harness.evaluate_constraints(x0)
    rows, cols = harness.jacobian_structure()
    values = harness.evaluate_jacobian_values(x0)
    assert rows.tolist() == [0, 0]

    observed = {names[col]: value for col, value in zip(cols.tolist(), values.tolist(), strict=False)}
    scale = float(harness._constraint_scale[0])
    ttip = solver.equations.params.get("ttip", {}).get("agr", 0.0)

    assert set(observed) == {"PT[agr]", "PP[agr]"}
    assert observed["PT[agr]"] == pytest.approx(1.0 / scale)
    assert observed["PP[agr]"] == pytest.approx(-(1.0 + ttip) / scale)


def test_constraint_harness_uses_analytic_eq95_derivatives() -> None:
    state = _build_dynamic_sam_excel()
    solver = IPOPTSolver(state, tolerance=1e-6, max_iterations=1)

    vars0 = solver._create_initial_guess()
    x0 = solver._variables_to_array(vars0)
    solver._build_variable_bounds(x_ref=x0)
    names = solver._last_bound_names
    harness = PEPConstraintJacobianHarness(
        equations=solver.equations,
        sets=solver.sets,
        n_variables=len(x0),
        hard_constraints=["EQ95"],
        variable_names=names,
        sparsity_reference_x=x0,
    )

    harness.evaluate_constraints(x0)
    rows, cols = harness.jacobian_structure()
    values = harness.evaluate_jacobian_values(x0)
    assert rows.tolist() == [0, 0, 0]

    observed = {names[col]: value for col, value in zip(cols.tolist(), values.tolist(), strict=False)}
    scale = float(harness._constraint_scale[0])
    pixgvt = vars0.PIXGVT

    assert set(observed) == {"G_REAL", "G", "PIXGVT"}
    assert observed["G_REAL"] == pytest.approx(1.0 / scale)
    assert observed["G"] == pytest.approx(-(1.0 / pixgvt) / scale)
    assert observed["PIXGVT"] == pytest.approx((vars0.G / (pixgvt ** 2)) / scale)


def test_constraint_harness_scales_analytic_eq79_derivatives() -> None:
    state = _build_dynamic_sam_excel()
    solver = IPOPTSolver(state, tolerance=1e-6, max_iterations=1)

    vars0 = solver._create_initial_guess()
    x0 = solver._variables_to_array(vars0)
    solver._build_variable_bounds(x_ref=x0)
    names = solver._last_bound_names

    q_idx = names.index("Q[agr]")
    x_scaled = x0.copy()
    x_scaled[q_idx] = x_scaled[q_idx] * 5.0 + 10.0

    harness = PEPConstraintJacobianHarness(
        equations=solver.equations,
        sets=solver.sets,
        n_variables=len(x_scaled),
        hard_constraints=["EQ79_agr"],
        variable_names=names,
        sparsity_reference_x=x_scaled,
    )

    harness.evaluate_constraints(x_scaled)
    rows, cols = harness.jacobian_structure()
    values = harness.evaluate_jacobian_values(x_scaled)
    observed = {names[col]: value for col, value in zip(cols.tolist(), values.tolist(), strict=False)}
    scale = float(harness._constraint_scale[0])
    vars_scaled = pep_array_to_variables(x_scaled, solver.sets)

    assert rows.tolist() == [0, 0, 0, 0, 0, 0]
    assert set(observed) == {"PC[agr]", "Q[agr]", "PM[agr]", "IM[agr]", "PD[agr]", "DD[agr]"}
    assert observed["PC[agr]"] == pytest.approx(vars_scaled.Q["agr"] / scale)
    assert observed["Q[agr]"] == pytest.approx(vars_scaled.PC["agr"] / scale)
    assert observed["PM[agr]"] == pytest.approx(-vars_scaled.IM["agr"] / scale)
    assert observed["IM[agr]"] == pytest.approx(-vars_scaled.PM["agr"] / scale)
    assert observed["PD[agr]"] == pytest.approx(-vars_scaled.DD["agr"] / scale)
    assert observed["DD[agr]"] == pytest.approx(-vars_scaled.PD["agr"] / scale)


def test_constraint_harness_uses_analytic_eq64_derivatives() -> None:
    state = _build_dynamic_sam_excel()
    solver = IPOPTSolver(state, tolerance=1e-6, max_iterations=1)

    vars0 = solver._create_initial_guess()
    x0 = solver._variables_to_array(vars0)
    solver._build_variable_bounds(x_ref=x0)
    names = solver._last_bound_names
    harness = PEPConstraintJacobianHarness(
        equations=solver.equations,
        sets=solver.sets,
        n_variables=len(x0),
        hard_constraints=["EQ64_agr"],
        variable_names=names,
        sparsity_reference_x=x0,
    )

    harness.evaluate_constraints(x0)
    rows, cols = harness.jacobian_structure()
    values = harness.evaluate_jacobian_values(x0)
    observed = {names[col]: value for col, value in zip(cols.tolist(), values.tolist(), strict=False)}
    scale = float(harness._constraint_scale[0])
    beta_m = solver.equations.params.get("beta_M", {}).get("agr", 0.0)
    sigma_m = solver.equations.params.get("sigma_M", {}).get("agr", 2.0)
    pd_i = vars0.PD["agr"]
    pm_i = vars0.PM["agr"]
    dd_i = vars0.DD["agr"]
    alloc = ((beta_m / (1.0 - beta_m)) * (pd_i / pm_i)) ** sigma_m
    expected_im = alloc * dd_i

    assert set(observed) == {"IM[agr]", "DD[agr]", "PD[agr]", "PM[agr]"}
    assert observed["IM[agr]"] == pytest.approx(1.0 / scale)
    assert observed["DD[agr]"] == pytest.approx(-alloc / scale)
    assert observed["PD[agr]"] == pytest.approx(-(expected_im * sigma_m / pd_i) / scale)
    assert observed["PM[agr]"] == pytest.approx((expected_im * sigma_m / pm_i) / scale)


def test_constraint_harness_uses_analytic_eq63_derivatives() -> None:
    state = _build_dynamic_sam_excel()
    solver = IPOPTSolver(state, tolerance=1e-6, max_iterations=1)

    vars0 = solver._create_initial_guess()
    x0 = solver._variables_to_array(vars0)
    solver._build_variable_bounds(x_ref=x0)
    names = solver._last_bound_names
    harness = PEPConstraintJacobianHarness(
        equations=solver.equations,
        sets=solver.sets,
        n_variables=len(x0),
        hard_constraints=["EQ63_agr"],
        variable_names=names,
        sparsity_reference_x=x0,
    )

    harness.evaluate_constraints(x0)
    rows, cols = harness.jacobian_structure()
    values = harness.evaluate_jacobian_values(x0)
    observed = {names[col]: value for col, value in zip(cols.tolist(), values.tolist(), strict=False)}
    scale = float(harness._constraint_scale[0])
    rho_m = solver.equations.params.get("rho_M", {}).get("agr", -0.5)
    b_m = solver.equations.params.get("B_M", {}).get("agr", 1.0)
    beta_m = solver.equations.params.get("beta_M", {}).get("agr", 0.5)
    im_i = vars0.IM["agr"]
    dd_i = vars0.DD["agr"]
    term = beta_m * (im_i ** (-rho_m)) + (1.0 - beta_m) * (dd_i ** (-rho_m))
    coeff = b_m * (term ** ((-1.0 / rho_m) - 1.0))

    assert set(observed) == {"Q[agr]", "IM[agr]", "DD[agr]"}
    assert observed["Q[agr]"] == pytest.approx(1.0 / scale)
    assert observed["IM[agr]"] == pytest.approx(-(coeff * beta_m * (im_i ** (-rho_m - 1.0))) / scale)
    assert observed["DD[agr]"] == pytest.approx(-(coeff * (1.0 - beta_m) * (dd_i ** (-rho_m - 1.0))) / scale)


def test_ipopt_builds_structural_closure_validation_report() -> None:
    state = _build_dynamic_sam_excel()
    solver = IPOPTSolver(state, config="default_ipopt")

    report = solver.build_closure_validation_report()

    assert report.system_shape == "square"
    assert report.is_valid is True
    assert report.numeraire == "e"
    assert report.numeraire_is_fixed is True
    assert report.active_equation_count > 200
    assert report.free_endogenous_variable_count > 200


def test_ipopt_contract_override_changes_explicit_fixed_closure_bounds() -> None:
    state = _build_dynamic_sam_excel()
    solver = IPOPTSolver(
        state,
        contract={
            "closure": {
                "fixed": ["PWM", "CMIN", "VSTK", "TR_SELF"],
                "endogenous": ["G", "CAB", "IT", "SH", "SF", "SG", "SROW"],
            }
        },
        config="default_ipopt",
    )

    vars0 = solver._create_initial_guess()
    x0 = solver._variables_to_array(vars0)
    lb, ub = solver._build_variable_bounds(x_ref=x0)
    names = solver._last_bound_names

    g_idx = names.index("G")
    cab_idx = names.index("CAB")

    assert not np.isclose(lb[g_idx], ub[g_idx])
    assert not np.isclose(lb[cab_idx], ub[cab_idx])


def test_ipopt_bounds_config_can_release_closure_fixed_symbol() -> None:
    state = _build_dynamic_sam_excel()
    solver = IPOPTSolver(
        state,
        contract={
            "closure": {
                "fixed": ["G", "CAB", "KS", "LS", "PWM", "PWX", "CMIN", "VSTK", "TR_SELF"],
                "endogenous": ["IT", "SH", "SF", "SG", "SROW"],
            },
            "bounds": {
                "fixed_from_closure": False,
                "free": ["PWM"],
            },
        },
        config="default_ipopt",
    )

    vars0 = solver._create_initial_guess()
    x0 = solver._variables_to_array(vars0)
    lb, ub = solver._build_variable_bounds(x_ref=x0)
    names = solver._last_bound_names

    pwm_idx = names.index(f"PWM[{state.sets['I'][0]}]")
    assert np.isneginf(lb[pwm_idx])
    assert np.isposinf(ub[pwm_idx])


def test_ipopt_closure_report_flags_unsupported_contract_symbols() -> None:
    state = _build_dynamic_sam_excel()
    solver = IPOPTSolver(
        state,
        contract={
            "closure": {
                "fixed": ["G", "LEON"],
                "endogenous": ["SG"],
            }
        },
        config="default_ipopt",
    )

    report = solver.build_closure_validation_report()

    assert report.is_valid is False
    assert report.unsupported_fixed_symbols == ("LEON",)
    assert report.unsupported_endogenous_symbols == ()


def test_ipopt_contract_supports_pwx_fixed_closure_bounds() -> None:
    state = _build_dynamic_sam_excel()
    solver = IPOPTSolver(
        state,
        contract={
            "closure": {
                "fixed": ["G", "CAB", "PWM", "PWX", "CMIN", "VSTK", "TR_SELF"],
                "endogenous": ["IT", "SH", "SF", "SG", "SROW"],
            }
        },
        config="default_ipopt",
    )

    vars0 = solver._create_initial_guess()
    x0 = solver._variables_to_array(vars0)
    lb, ub = solver._build_variable_bounds(x_ref=x0)
    names = solver._last_bound_names

    pwx_idx = names.index(f"PWX[{state.sets['I'][0]}]")
    assert np.isclose(lb[pwx_idx], ub[pwx_idx])


def test_ipopt_contract_supports_ls_fixed_closure_bounds() -> None:
    state = _build_dynamic_sam_excel()
    solver = IPOPTSolver(
        state,
        contract={
            "closure": {
                "fixed": ["G", "CAB", "LS", "PWM", "PWX", "CMIN", "VSTK", "TR_SELF"],
                "endogenous": ["IT", "SH", "SF", "SG", "SROW"],
            }
        },
        config="default_ipopt",
    )

    vars0 = solver._create_initial_guess()
    x0 = solver._variables_to_array(vars0)
    lb, ub = solver._build_variable_bounds(x_ref=x0)
    names = solver._last_bound_names

    ls_idx = names.index(f"LS[{state.sets['L'][0]}]")
    assert np.isclose(lb[ls_idx], ub[ls_idx])


def test_ipopt_contract_supports_ks_fixed_closure_bounds_mobile_capital() -> None:
    state = _build_dynamic_sam_excel()
    solver = IPOPTSolver(
        state,
        contract={
            "closure": {
                "fixed": ["G", "CAB", "KS", "LS", "PWM", "PWX", "CMIN", "VSTK", "TR_SELF"],
                "endogenous": ["IT", "SH", "SF", "SG", "SROW"],
                "capital_mobility": "mobile",
            }
        },
        config="default_ipopt",
    )

    vars0 = solver._create_initial_guess()
    x0 = solver._variables_to_array(vars0)
    lb, ub = solver._build_variable_bounds(x_ref=x0)
    names = solver._last_bound_names

    ks_idx = names.index(f"KS[{state.sets['K'][0]}]")
    assert np.isclose(lb[ks_idx], ub[ks_idx])


def test_ipopt_contract_maps_ks_to_kd_when_capital_is_sector_specific() -> None:
    state = _build_dynamic_sam_excel()
    solver = IPOPTSolver(
        state,
        contract={
            "closure": {
                "fixed": ["G", "CAB", "KS", "LS", "PWM", "PWX", "CMIN", "VSTK", "TR_SELF"],
                "endogenous": ["IT", "SH", "SF", "SG", "SROW"],
                "capital_mobility": "sector_specific",
            }
        },
        config="default_ipopt",
    )

    vars0 = solver._create_initial_guess()
    x0 = solver._variables_to_array(vars0)
    lb, ub = solver._build_variable_bounds(x_ref=x0)
    names = solver._last_bound_names

    kd_idx = names.index(f"KD[{state.sets['K'][0]},{state.sets['J'][0]}]")
    ks_idx = names.index(f"KS[{state.sets['K'][0]}]")

    assert np.isclose(lb[kd_idx], ub[kd_idx])
    assert not np.isclose(lb[ks_idx], ub[ks_idx])
