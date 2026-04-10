from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "parity" / "compare_path_solver_parity.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("compare_path_solver_parity", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_main_gate_passes(monkeypatch: Any, tmp_path: Path) -> None:
    module = _load_module()
    out_file = tmp_path / "report.json"

    def fake_run(**_kwargs: Any) -> dict[str, Any]:
        return {
            "passed": True,
            "pathampl": {"status": "ok", "termination_condition": "optimal"},
            "path_capi": {"termination_code": 1, "residual": 1.0e-9},
            "metrics": {
                "max_abs_price_diff": 0.0,
                "max_abs_market_sum_diff": 0.0,
                "max_abs_plant_sum_diff": 0.0,
                "max_abs_arc_diff": 0.0,
            },
        }

    monkeypatch.setattr(module, "run_transmcp_parity", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_path_solver_parity.py",
            "--gate",
            "--save-report",
            str(out_file),
        ],
    )

    code = module.main()
    assert code == 0
    payload = json.loads(out_file.read_text())
    assert payload["passed"] is True


def test_cli_main_gate_fails(monkeypatch: Any) -> None:
    module = _load_module()

    def fake_run(**_kwargs: Any) -> dict[str, Any]:
        return {
            "passed": False,
            "pathampl": {"status": "ok", "termination_condition": "optimal"},
            "path_capi": {"termination_code": 4, "residual": 1.0e-2},
            "metrics": {
                "max_abs_price_diff": 1.0e-1,
                "max_abs_market_sum_diff": 1.0e-1,
                "max_abs_plant_sum_diff": 1.0e-1,
                "max_abs_arc_diff": 1.0e-1,
            },
        }

    monkeypatch.setattr(module, "run_transmcp_parity", fake_run)
    monkeypatch.setattr(sys, "argv", ["compare_path_solver_parity.py", "--gate"])

    code = module.main()
    assert code == 2
