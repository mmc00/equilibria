"""The pure bundle builder works on the model in data/gtap7/gams (ADR-0010)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from equilibria.sources import GAMS_MODEL_DIR

ROOT = Path(__file__).resolve().parents[2]


def _builder(monkeypatch: Any) -> Any:
    monkeypatch.syspath_prepend(str(ROOT / "scripts" / "gtap"))
    path = ROOT / "scripts" / "gtap" / "build_gtap7_pure_local_bundle.py"
    spec = importlib.util.spec_from_file_location("build_gtap7_pure_local_bundle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_path_opt_lands_after_the_shock_options_of_the_repository_model(
    monkeypatch: Any,
) -> None:
    # comp.gms in data/gtap7/gams sets iterlim = 50000 (see its source.yaml changes);
    # the anchor must find that line, whatever its iteration limit, and keep it
    text = (GAMS_MODEL_DIR / "comp.gms").read_text()
    out = _builder(monkeypatch)._inject_path_opt_for_shock(text)
    assert "$onecho > path.opt" in out
    assert "solprint = off, iterlim = 50000 ;\n\n* === PATH stabilisation" in out
