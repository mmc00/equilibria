"""Integration test: pep2 GAMS parity between GDX-load and Excel-load variants."""

from __future__ import annotations

import csv
import os
import re
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PEP2_SCRIPTS = PROJECT_ROOT / "src/equilibria/templates/reference/pep2/scripts"
COMPARE_SCRIPT = PEP2_SCRIPTS / "compare_ipopt_vs_excel.sh"
DEFAULT_GAMS_BIN = Path("/Library/Frameworks/GAMS.framework/Versions/48/Resources/gams")
DEFAULT_GDXDIFF_BIN = Path("/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdiff")


def _collect_diff_symbols(diff_txt: Path) -> set[str]:
    symbols: set[str] = set()
    for line in diff_txt.read_text().splitlines():
        m = re.match(r"\s*([A-Za-z0-9_]+)\s+(Data are different|Keys are different)", line)
        if m:
            symbols.add(m.group(1))
    return symbols


def _max_abs_value(csv_file: Path) -> float:
    if not csv_file.exists():
        return 0.0
    max_abs = 0.0
    with csv_file.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                max_abs = max(max_abs, abs(float(row["Val"])))
            except (KeyError, TypeError, ValueError):
                continue
    return max_abs


def _run_gams_model(gams_bin: Path, model_file: str) -> None:
    subprocess.run(
        [str(gams_bin), model_file, "lo=0"],
        cwd=PEP2_SCRIPTS,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.mark.slow
@pytest.mark.gams
def test_pep2_gdx_vs_excel_loader_parity() -> None:
    """Ensure `ipopt` and `ipopt_excel` produce equivalent GAMS outputs."""
    gams_bin = Path(os.environ.get("GAMS_BIN", str(DEFAULT_GAMS_BIN)))
    if not gams_bin.exists():
        pytest.skip(f"GAMS not found: {gams_bin}")
    if not COMPARE_SCRIPT.exists():
        pytest.skip(f"Comparison script not found: {COMPARE_SCRIPT}")

    env = os.environ.copy()
    env["GAMS_BIN"] = str(gams_bin)

    subprocess.run(
        [str(COMPARE_SCRIPT)],
        cwd=PEP2_SCRIPTS,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    params_diff = PEP2_SCRIPTS / "gdxdiff_params_ipopt_vs_excel.txt"
    results_diff = PEP2_SCRIPTS / "gdxdiff_results_ipopt_vs_excel.txt"
    assert params_diff.exists(), "Missing parameter diff report"
    assert results_diff.exists(), "Missing results diff report"

    params_text = params_diff.read_text()
    assert "No differences found" in params_text, params_text

    # For results, only machine-noise differences are tolerated in these symbols.
    allowed_noise_symbols = {"valSH", "valTR", "valYHTR"}
    observed_symbols = _collect_diff_symbols(results_diff)
    assert observed_symbols.issubset(allowed_noise_symbols), (
        f"Unexpected result differences: {sorted(observed_symbols)}\n\n"
        f"{results_diff.read_text()}"
    )

    max_sh = _max_abs_value(PEP2_SCRIPTS / "dif_valSH.csv")
    max_tr = _max_abs_value(PEP2_SCRIPTS / "dif_valTR.csv")
    max_yhtr = _max_abs_value(PEP2_SCRIPTS / "dif_valYHTR.csv")
    tol = 1e-20
    assert max_sh <= tol, f"valSH diff too large: {max_sh}"
    assert max_tr <= tol, f"valTR diff too large: {max_tr}"
    assert max_yhtr <= tol, f"valYHTR diff too large: {max_yhtr}"


@pytest.mark.slow
@pytest.mark.gams
def test_pep2_hardcoded_vs_dynamic_sets_parity() -> None:
    """Ensure hardcoded-set and dynamic-set GAMS models produce identical results."""
    gams_bin = Path(os.environ.get("GAMS_BIN", str(DEFAULT_GAMS_BIN)))
    gdxdiff_bin = Path(os.environ.get("GDXDIFF_BIN", str(DEFAULT_GDXDIFF_BIN)))
    if not gams_bin.exists():
        pytest.skip(f"GAMS not found: {gams_bin}")
    if not gdxdiff_bin.exists():
        pytest.skip(f"gdxdiff not found: {gdxdiff_bin}")

    dynamic_model = "PEP-1-1_v2_1_ipopt_excel_dynamic_sets.gms"
    hardcoded_model = "PEP-1-1_v2_1_ipopt_excel.gms"
    if not (PEP2_SCRIPTS / dynamic_model).exists():
        pytest.skip(f"Missing model file: {dynamic_model}")
    if not (PEP2_SCRIPTS / hardcoded_model).exists():
        pytest.skip(f"Missing model file: {hardcoded_model}")

    # Run dynamic-sets model first and preserve its Results.gdx
    _run_gams_model(gams_bin, dynamic_model)
    dynamic_results = PEP2_SCRIPTS / "Results_ipopt_excel_dynamic_sets_test.gdx"
    subprocess.run(
        ["cp", "Results.gdx", dynamic_results.name],
        cwd=PEP2_SCRIPTS,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Run hardcoded reference and preserve Results.gdx
    _run_gams_model(gams_bin, hardcoded_model)
    hardcoded_results = PEP2_SCRIPTS / "Results_ipopt_excel_hardcoded_test.gdx"
    subprocess.run(
        ["cp", "Results.gdx", hardcoded_results.name],
        cwd=PEP2_SCRIPTS,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    diff_report = PEP2_SCRIPTS / "gdxdiff_results_hardcoded_vs_dynamic_sets_test.txt"
    subprocess.run(
        [str(gdxdiff_bin), hardcoded_results.name, dynamic_results.name],
        cwd=PEP2_SCRIPTS,
        check=False,
        stdout=diff_report.open("w"),
        stderr=subprocess.PIPE,
        text=True,
    )

    txt = diff_report.read_text()
    if "No differences found" in txt:
        return

    # Mirror the expected machine-noise footprint observed for equivalent runs.
    allowed_noise_symbols = {"valSH", "valTR", "valYHTR"}
    observed_symbols = _collect_diff_symbols(diff_report)
    assert observed_symbols.issubset(allowed_noise_symbols), txt
