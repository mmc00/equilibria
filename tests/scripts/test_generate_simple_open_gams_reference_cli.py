from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "parity" / "generate_simple_open_gams_reference.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("generate_simple_open_gams_reference", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_main_skip_gams_builds_manifest(tmp_path: Path, monkeypatch: Any) -> None:
    module = _load_module()
    default_gdx = tmp_path / "default.gdx"
    flexible_gdx = tmp_path / "flexible.gdx"
    default_gdx.write_bytes(b"default")
    flexible_gdx.write_bytes(b"flexible")
    gms_script = tmp_path / "simple_open.gms"
    gms_script.write_text(
        """
        OPTION NLP = ipopt;
        SOLVE simple_open_v1_benchmark USING NLP MINIMIZING OBJ;
        """
    )
    out_dir = tmp_path / "reference"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_simple_open_gams_reference.py",
            "--gms-script",
            str(gms_script),
            "--output-dir",
            str(out_dir),
            "--skip-gams",
            "--gdx",
            f"simple_open_default={default_gdx}",
            "--gdx",
            f"flexible_external_balance={flexible_gdx}",
        ],
    )

    code = module.main()
    assert code == 0
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == "simple_open_gams_reference/v1"
    assert manifest["closure_references"]["simple_open_default"]["results_gdx"]["path"] == str(default_gdx.resolve())
    assert manifest["closure_references"]["flexible_external_balance"]["results_gdx"]["path"] == str(flexible_gdx.resolve())
