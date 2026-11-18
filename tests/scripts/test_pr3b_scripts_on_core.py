"""The scripts moved by PR 3b of gtap-on-core run on the core driver (``Exercise``)
and solve to the same point as on main. Oracle:
``tests/scripts/golden/pr3b_scripts_main.json.gz`` (``scripts/dev/record_pr3b_scripts.py``,
recorded on main before the move)."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts/gtap" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_burfisher_defaults_to_the_core() -> None:
    sig = inspect.signature(_script("run_burfisher").solve_exercise)
    assert sig.parameters["route"].default == "core"


def test_compare_gempack_levels_defaults_to_the_core() -> None:
    sig = inspect.signature(_script("compare_gempack_levels").solve)
    assert sig.parameters["route"].default == "exercise"
