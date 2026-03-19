from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "parity" / "measure_pep_jacobian_modes.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("measure_pep_jacobian_modes", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _official_manifest_payload() -> dict[str, Any]:
    refs = {
        "base": ("base.gdx", "sim1"),
        "export_tax": ("export.gdx", "sim1"),
        "import_price_agr": ("imp_price.gdx", "sim1"),
        "import_shock": ("imp.gdx", "sim1"),
        "government_spending": ("gov.gdx", "sim1"),
    }
    return {
        "schema_version": "pep_gams_nlp_reference/v1",
        "generated_at": "2026-03-19T00:00:00+00:00",
        "model": "pep",
        "source": "gams",
        "problem_type": "nlp",
        "solver": "ipopt",
        "script_model_types": ["nlp"],
        "gms_script": {"path": "reference.gms", "sha256": "gms-sha"},
        "sam_file": {"path": "sam.gdx", "sha256": "sam-sha"},
        "scenario_slices": {name: slice_ for name, (_, slice_) in refs.items()},
        "scenario_references": {
            name: {
                "slice": slice_,
                "results_gdx": {"path": path, "sha256": f"{name}-sha"},
            }
            for name, (path, slice_) in refs.items()
        },
    }


class _FakePepSimulator:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.mode = kwargs["config"]["jacobian_mode"]

    def fit(self) -> _FakePepSimulator:
        return self

    def run_scenarios(self, **kwargs: Any) -> dict[str, Any]:
        scenarios = kwargs["scenarios"]
        if not scenarios:
            name = "base"
        else:
            name = scenarios[0].name
        is_analytic = self.mode == "analytic"
        fd = 0 if is_analytic else 100
        wall = 0.1 if is_analytic else 1.0
        entry = {
            "name": name,
            "solve": {
                "converged": True,
                "iterations": 4,
                "final_residual": 1e-9 if is_analytic else 2e-9,
                "message": f"{name}-{self.mode}-ok",
                "solver_stats": {
                    "jacobian_mode": self.mode,
                    "wall_time_seconds": wall,
                    "finite_difference_eval_count": fd,
                    "constraint_eval_count": 10,
                    "jacobian_eval_count": 5,
                    "structure_eval_count": 1,
                    "jacobian_nonzero_count": 25,
                    "hard_constraint_count": 10,
                    "variable_count": 10,
                    "objective_eval_count": 0,
                },
            },
            "comparison": {
                "gams_slice": "sim1",
                "passed": True,
                "compared": 10,
                "mismatches": 0,
                "missing": 0,
                "max_abs_diff": 0.0,
                "max_rel_diff": 0.0,
            },
            "reference_slice": "sim1",
        }
        if not scenarios:
            return {"base": entry, "scenarios": []}
        return {"base": None, "scenarios": [entry]}


class _FakePepSimulatorGateFail(_FakePepSimulator):
    def run_scenarios(self, **kwargs: Any) -> dict[str, Any]:
        payload = super().run_scenarios(**kwargs)
        entry = payload["base"] if payload["base"] is not None else payload["scenarios"][0]
        if self.mode == "analytic" and entry["name"] == "import_shock":
            entry["solve"]["solver_stats"]["finite_difference_eval_count"] = 3
        return payload


def test_cli_main_gate_passes(monkeypatch: Any, tmp_path: Path) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "PepSimulator", _FakePepSimulator)

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_official_manifest_payload()))
    out_file = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_pep_jacobian_modes.py",
            "--reference-manifest",
            str(manifest),
            "--gate",
            "--save-report",
            str(out_file),
        ],
    )

    code = module.main()
    assert code == 0
    payload = json.loads(out_file.read_text())
    assert payload["gate"]["passed"] is True
    assert payload["mode_comparison"]["export_tax"]["deltas"]["analytic_speedup_vs_numeric"] == 10.0


def test_cli_main_gate_fails_when_analytic_uses_fd(monkeypatch: Any, tmp_path: Path) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "PepSimulator", _FakePepSimulatorGateFail)

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_official_manifest_payload()))
    out_file = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_pep_jacobian_modes.py",
            "--reference-manifest",
            str(manifest),
            "--gate",
            "--save-report",
            str(out_file),
        ],
    )

    code = module.main()
    assert code == 2
    payload = json.loads(out_file.read_text())
    assert payload["gate"]["passed"] is False
    assert any("finite_difference_eval_count" in item for item in payload["gate"]["failures"])
