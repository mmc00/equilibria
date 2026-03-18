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
    results_gdx = tmp_path / "Results.gdx"
    results_gdx.write_bytes(b"results")
    parameters_gdx = tmp_path / "Parameters.gdx"
    parameters_gdx.write_bytes(b"parameters")
    presolve_gdx = tmp_path / "PreSolveLevels.gdx"
    presolve_gdx.write_bytes(b"presolve")
    out_dir = tmp_path / "out"

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
            "--results-gdx",
            str(results_gdx),
            "--parameters-gdx",
            str(parameters_gdx),
            "--presolve-levels-gdx",
            str(presolve_gdx),
            "--scenario-slice",
            "base=base",
            "--scenario-slice",
            "government_spending=sim1",
        ],
    )

    code = module.main()
    assert code == 0

    manifest_path = out_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    assert payload["problem_type"] == "nlp"
    assert payload["solver"] == "ipopt"
    assert payload["scenario_slices"]["base"] == "base"
    assert payload["scenario_slices"]["government_spending"] == "sim1"
    assert payload["results_gdx"]["sha256"]
