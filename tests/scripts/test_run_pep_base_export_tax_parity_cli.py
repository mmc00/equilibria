from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "cli" / "run_pep_base_export_tax_parity.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_pep_base_export_tax_parity", SCRIPT)
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

    def run_export_tax(self, **kwargs: Any) -> dict[str, Any]:
        _ = kwargs
        return {
            "base": {
                "solve": {
                    "converged": True,
                    "iterations": 0,
                    "final_residual": 0.0,
                },
                "comparison": {
                    "gams_slice": "base",
                    "passed": True,
                    "compared": 10,
                    "mismatches": 0,
                    "missing": 0,
                    "max_abs_diff": 0.0,
                    "max_rel_diff": 0.0,
                },
            },
            "scenarios": [
                {
                    "name": "export_tax",
                    "solve": {
                        "converged": True,
                        "iterations": 1,
                        "final_residual": 1e-8,
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
                }
            ],
        }


def test_cli_main_success_and_save_report(monkeypatch: Any, tmp_path: Path) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "PepSimulator", _FakePepSimulator)

    out_file = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pep_base_export_tax_parity.py",
            "--save-report",
            str(out_file),
        ],
    )
    code = module.main()
    assert code == 0
    payload = json.loads(out_file.read_text())
    assert payload["scenarios"]["base"]["gams_comparison"]["passed"] is True
    assert payload["scenarios"]["export_tax"]["gams_comparison"]["passed"] is True


def test_cli_main_returns_2_on_failed_comparison(monkeypatch: Any) -> None:
    module = _load_module()

    class _FailingFake(_FakePepSimulator):
        def run_export_tax(self, **kwargs: Any) -> dict[str, Any]:
            report = super().run_export_tax(**kwargs)
            report["scenarios"][0]["comparison"]["passed"] = False
            report["scenarios"][0]["comparison"]["mismatches"] = 1
            return report

    monkeypatch.setattr(module, "PepSimulator", _FailingFake)
    monkeypatch.setattr(sys, "argv", ["run_pep_base_export_tax_parity.py"])
    code = module.main()
    assert code == 2
