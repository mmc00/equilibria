from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "parity" / "generate_pep_gams_nlp_reference.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("generate_pep_gams_nlp_reference", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_main_skip_gams_generates_manifest(tmp_path: Path, monkeypatch: Any) -> None:
    module = _load_module()

    gms_script = tmp_path / "reference.gms"
    gms_script.write_text(
        """
        OPTION NLP = ipopt;
        $if not set PEP_SOLVE_MODE $set PEP_SOLVE_MODE CNS
        $ifthenI "%PEP_SOLVE_MODE%" == "NLP"
        SOLVE PEP11 USING NLP MINIMIZING OBJ;
        $else
        SOLVE PEP11 USING CNS;
        $endif
        """
    )
    sam_file = tmp_path / "SAM.xlsx"
    sam_file.write_bytes(b"sam")
    val_par_file = tmp_path / "VAL_PAR.xlsx"
    val_par_file.write_bytes(b"val")
    out_dir = tmp_path / "out"
    for scenario_name in ("base", "government_spending"):
        scenario_dir = out_dir / "scenarios" / scenario_name / "scripts"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        (scenario_dir / "Results.gdx").write_bytes(f"{scenario_name}-results".encode())
        (scenario_dir / "Parameters.gdx").write_bytes(f"{scenario_name}-parameters".encode())
        (scenario_dir / "PreSolveLevels.gdx").write_bytes(f"{scenario_name}-presolve".encode())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_pep_gams_nlp_reference.py",
            "--gms-script",
            str(gms_script),
            "--sam-file",
            str(sam_file),
            "--val-par-file",
            str(val_par_file),
            "--output-dir",
            str(out_dir),
            "--skip-gams",
            "--scenario",
            "base",
            "--scenario",
            "government_spending",
        ],
    )

    code = module.main()
    assert code == 0

    manifest_path = out_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    assert payload["problem_type"] == "nlp"
    assert payload["solver"] == "ipopt"
    assert payload["scenario_slices"]["base"] == "sim1"
    assert payload["scenario_slices"]["government_spending"] == "sim1"
    assert payload["scenario_references"]["base"]["results_gdx"]["sha256"]
    assert payload["scenario_references"]["government_spending"]["results_gdx"]["sha256"]
