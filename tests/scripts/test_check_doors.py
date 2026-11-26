"""check_doors.py: the pre-commit check that every template keeps behind a door."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "check_doors.py"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("check_doors", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_repo_passes(capsys: pytest.CaptureFixture[str]) -> None:
    assert _module().main() == 0


def test_a_broken_rule_fails_and_names_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _module()
    monkeypatch.setattr(mod.door, "coverage_problems", lambda: ["pkg.New has no door"])
    assert mod.main() == 1
    assert "pkg.New has no door" in capsys.readouterr().err


def test_a_broken_door_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _module()
    monkeypatch.setattr(mod.door, "check_door", lambda spec: [f"{spec.module}: bad"])
    assert mod.main() == 1
    assert "equilibria.gtap: bad" in capsys.readouterr().err
