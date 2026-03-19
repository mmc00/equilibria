from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "parity" / "measure_simple_open_jacobian_modes.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("measure_simple_open_jacobian_modes", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_main_gate_passes(tmp_path: Path, monkeypatch: Any) -> None:
    module = _load_module()
    out_file = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_simple_open_jacobian_modes.py",
            "--gate",
            "--save-report",
            str(out_file),
        ],
    )

    code = module.main()
    assert code == 0
    payload = json.loads(out_file.read_text())
    assert payload["gate"]["passed"] is True
    assert payload["mode_comparison"]["simple_open_default"]["analytic"]["solver_stats"]["finite_difference_eval_count"] == 0
    assert payload["mode_comparison"]["flexible_external_balance"]["numeric"]["solver_stats"]["finite_difference_eval_count"] > 0
