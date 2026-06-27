"""LAYER_SPECS: the single source of truth for cascade-tool invocation. Each
builder is PURE (dataset, period, gdx) -> argv, validated against the verified
argparse of each tool. Diagnostic order, cheap+static -> expensive+solve."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from cascade_config import ROOT, scenario_for, family, SCENARIO_BY_PERIOD

PY = str(ROOT / ".venv" / "bin" / "python")
GTAP = "scripts/gtap"
PARITY = "scripts/parity"


@dataclass
class LayerSpec:
    name: str
    tool: str
    periods: tuple[str, ...]
    seeds: bool
    build: Callable[[str, str, Optional[Path]], list[str]]


def _benchmark_period(ds: str) -> str:
    return "check" if "check" in SCENARIO_BY_PERIOD[family(ds)] else "base"


def _c_pairing(ds, period, gdx):
    return [PY, f"{GTAP}/diff_mcp_pairing.py", "--dataset", ds,
            "--period", period, "--apply-closure"]


def _c_nl(ds, period, gdx):
    return [PY, f"{GTAP}/nl_compare.py", "--dataset", ds,
            "--phase", period, "--skip-gams"]


def _c_calibration(ds, period, gdx):
    return [PY, f"{GTAP}/diff_calibration.py", "--dataset", ds,
            "--gdx", str(gdx), "--period", _benchmark_period(ds)]


def _c_validate_ref(ds, period, gdx):
    return [PY, f"{GTAP}/validate_reference.py", "--dataset", ds,
            "--period", period, "--gdx", str(gdx)]


def _c_probe_seed(ds, period, gdx):
    return [PY, f"{PARITY}/probe.py", "--template", "gtap", "--dataset", ds,
            "--scenario", scenario_for(ds, period),
            "--seed-gams", period, "--gdx-ref", str(gdx), "--residuals"]


def _c_drift(ds, period, gdx):
    return [PY, f"{GTAP}/drift_test.py", "--dataset", ds,
            "--gdx", str(gdx), "--period", period]


LAYER_SPECS: list[LayerSpec] = [
    LayerSpec("mcp_pairing",        f"{GTAP}/diff_mcp_pairing.py",   ("base", "check", "shock"), False, _c_pairing),
    LayerSpec("nl_compare",         f"{GTAP}/nl_compare.py",         ("base", "check", "shock"), False, _c_nl),
    LayerSpec("calibration",        f"{GTAP}/diff_calibration.py",   ("base", "check", "shock"), False, _c_calibration),
    LayerSpec("validate_reference", f"{GTAP}/validate_reference.py", ("base", "check", "shock"), False, _c_validate_ref),
    LayerSpec("probe_seed",         f"{PARITY}/probe.py",            ("base", "check", "shock"), True,  _c_probe_seed),
    LayerSpec("drift_test",         f"{GTAP}/drift_test.py",         ("check", "shock"),         True,  _c_drift),
]


def layer_by_name(name: str) -> LayerSpec:
    for l in LAYER_SPECS:
        if l.name == name:
            return l
    raise KeyError(name)


def build_cmd(layer: LayerSpec, dataset: str, period: str,
              gdx: Optional[Path]) -> Optional[list[str]]:
    if period not in layer.periods:
        return None
    return layer.build(dataset, period, gdx)
