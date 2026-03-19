from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "cli" / "run_pep_core_scenarios_gate.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_pep_core_scenarios_gate", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakePepSimulator:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def fit(self) -> _FakePepSimulator:
        return self

    def run_scenarios(self, **kwargs: Any) -> dict[str, Any]:
        scenarios = kwargs["scenarios"]
        ref = kwargs.get("reference_results_gdx")
        base_reference_slice = kwargs.get("base_reference_slice", "base")
        if not scenarios:
            return {
                "base": {
                    "solve": {
                        "converged": True,
                        "iterations": 0,
                        "final_residual": 0.0,
                        "message": "base-ok",
                    },
                    "comparison": (
                        {
                            "gams_slice": base_reference_slice,
                            "passed": bool(ref),
                            "compared": 10,
                            "mismatches": 0 if ref else 0,
                            "missing": 0,
                            "max_abs_diff": 0.0,
                            "max_rel_diff": 0.0,
                        }
                        if ref
                        else None
                    ),
                    "reference_slice": base_reference_slice,
                },
                "scenarios": [],
            }

        scenario = scenarios[0]
        return {
            "base": None,
            "scenarios": [
                {
                    "name": scenario.name,
                    "solve": {
                        "converged": True,
                        "iterations": 1,
                        "final_residual": 1e-9,
                        "message": f"{scenario.name}-ok",
                    },
                    "comparison": (
                        {
                            "gams_slice": "sim1",
                            "passed": bool(ref),
                            "compared": 10,
                            "mismatches": 0 if ref else 0,
                            "missing": 0,
                            "max_abs_diff": 0.0,
                            "max_rel_diff": 0.0,
                        }
                        if ref
                        else None
                    ),
                    "reference_slice": "sim1",
                }
            ],
        }


def test_cli_main_success_with_reference_manifest(monkeypatch: Any, tmp_path: Path) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "PepSimulator", _FakePepSimulator)

    manifest = tmp_path / "refs.json"
    manifest.write_text(
        json.dumps(
            {
                "base": {"results_gdx": "base.gdx", "slice": "base"},
                "export_tax": {"results_gdx": "export.gdx", "slice": "sim1"},
                "import_price_agr": {"results_gdx": "imp_price.gdx", "slice": "sim1"},
                "import_shock": {"results_gdx": "imp.gdx", "slice": "sim1"},
                "government_spending": {"results_gdx": "gov.gdx", "slice": "sim1"},
            }
        )
    )
    out_file = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pep_core_scenarios_gate.py",
            "--reference-manifest",
            str(manifest),
            "--require-reference-manifest",
            "--save-report",
            str(out_file),
        ],
    )
    code = module.main()
    assert code == 0
    payload = json.loads(out_file.read_text())
    assert payload["scenarios"]["base"]["comparison"]["passed"] is True
    assert payload["scenarios"]["import_shock"]["comparison"]["passed"] is True
    assert payload["scenarios"]["government_spending"]["solve"]["converged"] is True


def test_cli_main_returns_2_when_reference_manifest_is_incomplete(monkeypatch: Any, tmp_path: Path) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "PepSimulator", _FakePepSimulator)

    manifest = tmp_path / "refs.json"
    manifest.write_text(
        json.dumps(
            {
                "base": {"results_gdx": "base.gdx", "slice": "base"},
                "export_tax": {"results_gdx": "export.gdx", "slice": "sim1"},
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pep_core_scenarios_gate.py",
            "--reference-manifest",
            str(manifest),
            "--require-reference-manifest",
        ],
    )
    code = module.main()
    assert code == 2


def test_cli_accepts_official_gams_nlp_reference_manifest(monkeypatch: Any, tmp_path: Path) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "PepSimulator", _FakePepSimulator)

    official_manifest = tmp_path / "official_manifest.json"
    official_manifest.write_text(
        json.dumps(
            {
                "schema_version": "pep_gams_nlp_reference/v1",
                "generated_at": "2026-03-18T00:00:00+00:00",
                "model": "pep",
                "source": "gams",
                "problem_type": "nlp",
                "solver": "ipopt",
                "script_model_types": ["nlp"],
                "gms_script": {"path": "reference.gms", "sha256": "gms-sha"},
                "sam_file": {"path": "sam.xlsx", "sha256": "sam-sha"},
                    "scenario_slices": {
                        "base": "sim1",
                        "export_tax": "sim1",
                        "import_price_agr": "sim1",
                        "import_shock": "sim1",
                    "government_spending": "sim1",
                },
                "scenario_references": {
                    "base": {
                        "slice": "sim1",
                        "results_gdx": {"path": "base.gdx", "sha256": "base-sha"},
                    },
                    "export_tax": {
                        "slice": "sim1",
                        "results_gdx": {"path": "export.gdx", "sha256": "export-sha"},
                    },
                    "import_price_agr": {
                        "slice": "sim1",
                        "results_gdx": {"path": "imp_price.gdx", "sha256": "imp-price-sha"},
                    },
                    "import_shock": {
                        "slice": "sim1",
                        "results_gdx": {"path": "imp.gdx", "sha256": "imp-sha"},
                    },
                    "government_spending": {
                        "slice": "sim1",
                        "results_gdx": {"path": "gov.gdx", "sha256": "gov-sha"},
                    },
                },
            }
        )
    )

    out_file = tmp_path / "official_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pep_core_scenarios_gate.py",
            "--reference-manifest",
            str(official_manifest),
            "--require-reference-manifest",
            "--save-report",
            str(out_file),
        ],
    )
    code = module.main()
    assert code == 0
    payload = json.loads(out_file.read_text())
    assert payload["scenarios"]["base"]["comparison"]["passed"] is True
    assert payload["scenarios"]["base"]["comparison"]["gams_slice"] == "sim1"
    assert payload["scenarios"]["export_tax"]["comparison"]["passed"] is True
