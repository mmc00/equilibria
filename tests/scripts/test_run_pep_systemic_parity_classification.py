from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_pep_systemic_parity.py"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_systemic_parity_report_classifies_cri_data_contract_failure(tmp_path: Path) -> None:
    report = tmp_path / "cri_contract_report.json"
    sam_file = ROOT / "src/equilibria/templates/reference/pep2/data/SAM-CRI-gams.xlsx"
    assert sam_file.exists()

    proc = _run(
        [
            "--sam-file",
            str(sam_file),
            "--dynamic-sam",
            "--cri-fix-mode",
            "off",
            "--method",
            "none",
            "--save-report",
            str(report),
        ]
    )

    assert proc.returncode == 2, proc.stdout + "\n" + proc.stderr
    payload = json.loads(report.read_text())
    assert payload["sam_qa"]["passed"] is False
    assert payload["classification"]["kind"] == "data_contract"
    assert payload["classification"]["reason"] == "sam_qa_failed"


def test_systemic_parity_report_classifies_pep2_pass(tmp_path: Path) -> None:
    report = tmp_path / "pep2_pass_report.json"
    sam_file = ROOT / "src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx"
    val_par_file = ROOT / "src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx"
    assert sam_file.exists()
    assert val_par_file.exists()

    proc = _run(
        [
            "--sam-file",
            str(sam_file),
            "--val-par-file",
            str(val_par_file),
            "--init-mode",
            "excel",
            "--method",
            "none",
            "--save-report",
            str(report),
        ]
    )

    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    payload = json.loads(report.read_text())
    assert payload["classification"]["kind"] == "pass"
    assert payload["classification"]["reason"] == "init_gates_passed"
    assert payload["init"]["gates"]["overall_passed"] is True


def test_systemic_parity_report_classifies_solver_dynamics_after_init_pass(
    tmp_path: Path,
) -> None:
    report = tmp_path / "solver_dynamics_report.json"
    sam_file = ROOT / "src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx"
    val_par_file = ROOT / "src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx"
    assert sam_file.exists()
    assert val_par_file.exists()

    proc = _run(
        [
            "--sam-file",
            str(sam_file),
            "--val-par-file",
            str(val_par_file),
            "--init-mode",
            "excel",
            "--method",
            "simple_iteration",
            "--max-iterations",
            "0",
            "--save-report",
            str(report),
        ]
    )

    assert proc.returncode == 2, proc.stdout + "\n" + proc.stderr
    payload = json.loads(report.read_text())
    assert payload["init"]["gates"]["overall_passed"] is True
    assert payload["solve"]["converged"] is False
    assert payload["classification"]["kind"] == "solver_dynamics"
    assert payload["classification"]["reason"] == "solve_not_converged"


def test_systemic_parity_cri_fixed_classifies_solver_dynamics_after_init_pass(
    tmp_path: Path,
) -> None:
    report = tmp_path / "cri_solver_dynamics_report.json"
    sam_file = ROOT / "src/equilibria/templates/reference/pep2/data/SAM-CRI-gams-fixed.xlsx"
    val_par_file = ROOT / "src/equilibria/templates/reference/pep2/data/VAL_PAR-CRI-gams.xlsx"
    assert sam_file.exists()
    assert val_par_file.exists()

    proc = _run(
        [
            "--sam-file",
            str(sam_file),
            "--val-par-file",
            str(val_par_file),
            "--dynamic-sam",
            "--init-mode",
            "gams",
            "--gams-results-slice",
            "sim1",
            "--disable-strict-gams-baseline-check",
            "--init-gates-mode",
            "gams_anchor",
            "--method",
            "simple_iteration",
            "--max-iterations",
            "0",
            "--save-report",
            str(report),
        ]
    )

    assert proc.returncode == 2, proc.stdout + "\n" + proc.stderr
    payload = json.loads(report.read_text())
    assert payload["sam_qa"]["passed"] is True
    assert payload["init"]["gates"]["overall_passed"] is True
    assert payload["solve"]["converged"] is False
    assert payload["classification"]["kind"] == "solver_dynamics"
    assert payload["classification"]["reason"] == "solve_not_converged"


def test_systemic_parity_cri_fixed_passes_qa_and_macro_closure(tmp_path: Path) -> None:
    report = tmp_path / "cri_fixed_pass_report.json"
    sam_file = ROOT / "src/equilibria/templates/reference/pep2/data/SAM-CRI-gams-fixed.xlsx"
    val_par_file = ROOT / "src/equilibria/templates/reference/pep2/data/VAL_PAR-CRI-gams.xlsx"
    assert sam_file.exists()
    assert val_par_file.exists()

    proc = _run(
        [
            "--sam-file",
            str(sam_file),
            "--val-par-file",
            str(val_par_file),
            "--dynamic-sam",
            "--init-mode",
            "gams",
            "--gams-results-slice",
            "sim1",
            "--disable-strict-gams-baseline-check",
            "--init-gates-mode",
            "gams_anchor",
            "--method",
            "none",
            "--save-report",
            str(report),
        ]
    )

    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    payload = json.loads(report.read_text())
    assert payload["sam_qa"]["passed"] is True
    assert payload["classification"]["kind"] == "pass"
    blocks = {
        b.get("block"): b
        for b in payload.get("init", {}).get("gates", {}).get("blocks", [])
    }
    macro = blocks.get("macro_closure")
    assert macro is not None
    assert macro.get("passed") is True
