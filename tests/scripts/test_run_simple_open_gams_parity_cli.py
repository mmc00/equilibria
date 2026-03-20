from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "parity" / "run_simple_open_gams_parity.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_simple_open_gams_parity", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_main_gate_passes(monkeypatch: Any, tmp_path: Path) -> None:
    module = _load_module()
    default_gdx = tmp_path / "default.gdx"
    flexible_gdx = tmp_path / "flex.gdx"
    default_gdx.write_bytes(b"default")
    flexible_gdx.write_bytes(b"flex")
    out_file = tmp_path / "report.json"

    def fake_compare(*, contract: Any, gdx_path: Path, abs_tol: float) -> Any:
        closure_name = contract["closure"]["name"]
        return type(
            "_Result",
            (),
            {
                "to_dict": lambda self: {
                    "closure_name": closure_name,
                    "gdx_path": str(gdx_path),
                    "passed": True,
                    "active_closure_match": True,
                    "benchmark_compared": 9,
                    "benchmark_mismatches": 0,
                    "benchmark_max_abs_diff": 0.0,
                    "level_compared": 9,
                    "level_mismatches": 0,
                    "level_max_abs_diff": 0.0,
                    "residual_compared": 3,
                    "residual_mismatches": 0,
                    "residual_max_abs": 0.0,
                    "parameter_compared": 6,
                    "parameter_mismatches": 0,
                    "parameter_max_abs_diff": 0.0,
                    "modelstat": 1.0,
                    "solvestat": 1.0,
                    "details": {},
                }
            },
        )()

    monkeypatch.setattr(module, "compare_simple_open_gams_parity", fake_compare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_simple_open_gams_parity.py",
            "--gate",
            "--gdx",
            f"simple_open_default={default_gdx}",
            "--gdx",
            f"flexible_external_balance={flexible_gdx}",
            "--save-report",
            str(out_file),
        ],
    )

    code = module.main()
    assert code == 0
    payload = json.loads(out_file.read_text())
    assert payload["gate"]["passed"] is True
    assert payload["closures"]["simple_open_default"]["passed"] is True
    assert payload["closures"]["flexible_external_balance"]["passed"] is True


def test_cli_main_gate_fails(monkeypatch: Any, tmp_path: Path) -> None:
    module = _load_module()
    default_gdx = tmp_path / "default.gdx"
    flexible_gdx = tmp_path / "flex.gdx"
    default_gdx.write_bytes(b"default")
    flexible_gdx.write_bytes(b"flex")

    def fake_compare(*, contract: Any, gdx_path: Path, abs_tol: float) -> Any:
        closure_name = contract["closure"]["name"]
        passed = closure_name != "flexible_external_balance"
        return type(
            "_Result",
            (),
            {
                "to_dict": lambda self: {
                    "closure_name": closure_name,
                    "gdx_path": str(gdx_path),
                    "passed": passed,
                    "active_closure_match": passed,
                    "benchmark_compared": 9,
                    "benchmark_mismatches": 0 if passed else 1,
                    "benchmark_max_abs_diff": 0.0 if passed else 1e-3,
                    "level_compared": 9,
                    "level_mismatches": 0 if passed else 1,
                    "level_max_abs_diff": 0.0 if passed else 1e-3,
                    "residual_compared": 3,
                    "residual_mismatches": 0 if passed else 1,
                    "residual_max_abs": 0.0 if passed else 1e-3,
                    "parameter_compared": 6,
                    "parameter_mismatches": 0,
                    "parameter_max_abs_diff": 0.0,
                    "modelstat": 1.0 if passed else 7.0,
                    "solvestat": 1.0,
                    "details": {},
                }
            },
        )()

    monkeypatch.setattr(module, "compare_simple_open_gams_parity", fake_compare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_simple_open_gams_parity.py",
            "--gate",
            "--gdx",
            f"simple_open_default={default_gdx}",
            "--gdx",
            f"flexible_external_balance={flexible_gdx}",
        ],
    )

    code = module.main()
    assert code == 2


def test_cli_main_accepts_official_manifest(monkeypatch: Any, tmp_path: Path) -> None:
    module = _load_module()
    default_gdx = tmp_path / "default.gdx"
    flexible_gdx = tmp_path / "flex.gdx"
    default_gdx.write_bytes(b"default")
    flexible_gdx.write_bytes(b"flex")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "simple_open_gams_reference/v1",
                "generated_at": "2026-03-19T00:00:00+00:00",
                "model": "simple_open_v1",
                "source": "gams",
                "anchor": "stdcge",
                "problem_type": "nlp",
                "solver": "ipopt",
                "script_model_types": ["nlp"],
                "gms_script": {"path": "simple_open_v1_benchmark.gms", "sha256": "gms-sha"},
                "closure_references": {
                    "simple_open_default": {
                        "closure": "simple_open_default",
                        "results_gdx": {"path": str(default_gdx), "sha256": "default-sha"},
                    },
                    "flexible_external_balance": {
                        "closure": "flexible_external_balance",
                        "results_gdx": {"path": str(flexible_gdx), "sha256": "flex-sha"},
                    },
                },
            }
        )
    )

    def fake_compare(*, contract: Any, gdx_path: Path, abs_tol: float) -> Any:
        closure_name = contract["closure"]["name"]
        return type(
            "_Result",
            (),
            {
                "to_dict": lambda self: {
                    "closure_name": closure_name,
                    "gdx_path": str(gdx_path),
                    "passed": True,
                    "active_closure_match": True,
                    "benchmark_compared": 9,
                    "benchmark_mismatches": 0,
                    "benchmark_max_abs_diff": 0.0,
                    "level_compared": 9,
                    "level_mismatches": 0,
                    "level_max_abs_diff": 0.0,
                    "residual_compared": 3,
                    "residual_mismatches": 0,
                    "residual_max_abs": 0.0,
                    "parameter_compared": 7,
                    "parameter_mismatches": 0,
                    "parameter_max_abs_diff": 0.0,
                    "modelstat": 1.0,
                    "solvestat": 1.0,
                    "details": {},
                }
            },
        )()

    monkeypatch.setattr(module, "compare_simple_open_gams_parity", fake_compare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_simple_open_gams_parity.py",
            "--gate",
            "--reference-manifest",
            str(manifest),
            "--require-reference-manifest",
        ],
    )

    code = module.main()
    assert code == 0
