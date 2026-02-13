"""Parity tests for pep2 Excel loading across Python and GAMS."""

from __future__ import annotations

import math
import os
import re
import subprocess
from pathlib import Path

import pytest

from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_calibration_unified_excel import PEPModelCalibratorExcel
from equilibria.templates.pep_model_solver import PEPModelSolver


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PEP2_ROOT = PROJECT_ROOT / "src/equilibria/templates/reference/pep2"
PEP2_SCRIPTS = PEP2_ROOT / "scripts"
DEFAULT_GAMS_BIN = Path("/Library/Frameworks/GAMS.framework/Versions/48/Resources/gams")
DEFAULT_GDXDUMP_BIN = Path("/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump")


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def _gdxdump_value(gdxdump_bin: Path, gdx_file: Path, symbol: str, label: str = "BASE") -> float | None:
    out = subprocess.check_output(
        [str(gdxdump_bin), str(gdx_file), f"symb={symbol}"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    pat = re.compile(r"'([^']+)'\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)")
    values = {m.group(1): float(m.group(2)) for m in pat.finditer(out)}
    return values.get(label)


@pytest.mark.integration
def test_pep2_python_gdx_vs_excel_equation_consistent() -> None:
    """Python benchmark initialization should match for GDX and Excel SAM loaders."""
    sam_gdx = PEP2_ROOT / "data" / "SAM-V2_0.gdx"
    sam_xls = PEP2_ROOT / "data" / "SAM-V2_0.xls"
    val_par = PEP2_ROOT / "data" / "VAL_PAR.xlsx"

    state_gdx = PEPModelCalibrator(sam_file=sam_gdx, val_par_file=val_par).calibrate()
    state_xls = PEPModelCalibratorExcel(sam_file=sam_xls, val_par_file=val_par).calibrate()

    solver_gdx = PEPModelSolver(calibrated_state=state_gdx, init_mode="equation_consistent")
    solver_xls = PEPModelSolver(calibrated_state=state_xls, init_mode="equation_consistent")

    vars_gdx = solver_gdx._create_initial_guess()
    vars_xls = solver_xls._create_initial_guess()

    res_gdx = solver_gdx.equations.calculate_all_residuals(vars_gdx)
    res_xls = solver_xls.equations.calculate_all_residuals(vars_xls)

    assert _rms(list(res_gdx.values())) <= 1e-8
    assert _rms(list(res_xls.values())) <= 1e-8

    # Check equation-by-equation parity (machine precision expected)
    for eq in sorted(set(res_gdx) | set(res_xls)):
        assert abs(res_xls.get(eq, 0.0) - res_gdx.get(eq, 0.0)) <= 1e-9, eq

    # Check key benchmark scalars
    assert abs(vars_xls.GDP_BP - vars_gdx.GDP_BP) <= 1e-9
    assert abs(vars_xls.GDP_MP - vars_gdx.GDP_MP) <= 1e-9
    assert abs(vars_xls.GDP_IB - vars_gdx.GDP_IB) <= 1e-9
    assert abs(vars_xls.GDP_FD - vars_gdx.GDP_FD) <= 1e-9
    assert abs(vars_xls.IT - vars_gdx.IT) <= 1e-9
    assert abs(vars_xls.CAB - vars_gdx.CAB) <= 1e-9


@pytest.mark.slow
@pytest.mark.gams
def test_pep2_gams_excel_vs_python_excel_key_scalars() -> None:
    """GAMS Excel-loading benchmark should match Python Excel-loading benchmark."""
    gams_bin = Path(os.environ.get("GAMS_BIN", str(DEFAULT_GAMS_BIN)))
    gdxdump_bin = Path(os.environ.get("GDXDUMP_BIN", str(DEFAULT_GDXDUMP_BIN)))
    if not gams_bin.exists():
        pytest.skip(f"GAMS not found: {gams_bin}")
    if not gdxdump_bin.exists():
        pytest.skip(f"gdxdump not found: {gdxdump_bin}")

    model_file = PEP2_SCRIPTS / "PEP-1-1_v2_1_ipopt_excel.gms"
    if not model_file.exists():
        pytest.skip(f"Missing model file: {model_file}")

    subprocess.run(
        [str(gams_bin), model_file.name, "lo=0"],
        cwd=PEP2_SCRIPTS,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    results_gdx = PEP2_SCRIPTS / "Results.gdx"
    assert results_gdx.exists(), "Missing GAMS Results.gdx after run"

    sam_xls = PEP2_ROOT / "data" / "SAM-V2_0.xls"
    val_par = PEP2_ROOT / "data" / "VAL_PAR.xlsx"
    state_xls = PEPModelCalibratorExcel(sam_file=sam_xls, val_par_file=val_par).calibrate()
    solver_xls = PEPModelSolver(calibrated_state=state_xls, init_mode="equation_consistent")
    vars_xls = solver_xls._create_initial_guess()

    pairs = [
        ("valGDP_BP", vars_xls.GDP_BP),
        ("valGDP_FD", vars_xls.GDP_FD),
        ("valGDP_IB", vars_xls.GDP_IB),
        ("valGDP_MP", vars_xls.GDP_MP),
        ("valIT", vars_xls.IT),
        ("valCAB", vars_xls.CAB),
        ("valYG", vars_xls.YG),
        ("valTPRODN", vars_xls.TPRODN),
        ("valTIWT", vars_xls.TIWT),
        ("valTIKT", vars_xls.TIKT),
    ]
    tol = 1e-6
    for symbol, py_val in pairs:
        gams_val = _gdxdump_value(gdxdump_bin, results_gdx, symbol, "BASE")
        assert gams_val is not None, f"Missing {symbol}('BASE') in Results.gdx"
        assert abs(py_val - gams_val) <= tol, (
            f"{symbol} mismatch: python={py_val:.12g}, gams={gams_val:.12g}, "
            f"diff={py_val - gams_val:.12g}"
        )
