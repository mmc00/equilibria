"""The GDX inliner runs inside ``uv run --with gamsapi``: its interpreter must be
the bare ``python`` that uv resolves to that ephemeral env. An absolute path
(``sys.executable``) skips the env and dies with ``No module named 'gams'``.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

from equilibria.sources import GAMS_TOOLS_DIR

ROOT = Path(__file__).resolve().parents[2]
BUILDERS = [
    GAMS_TOOLS_DIR / "build_gtap7_altertax_neos_bundle.py",
    ROOT / "scripts/gtap/build_altertax_neos_bundle.py",
    ROOT / "scripts/gtap/build_9x10_altertax_neos_bundle.py",
]


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"_builder_{path.stem}", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda p: p.name)
def test_inliner_runs_in_the_uv_env(builder, monkeypatch, tmp_path):
    mod = _load(builder)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        mod.subprocess, "run",
        lambda cmd, *a, **k: calls.append(list(cmd)) or subprocess.CompletedProcess(cmd, 0),
    )  # fmt: skip
    mod.run_inliner(tmp_path / "in.gdx", tmp_path / "d.gms", tmp_path / "p.gms")
    (cmd,) = calls
    assert cmd[:2] == ["uv", "run"]
    interpreter = next(
        x for i, x in enumerate(cmd[2:], 2) if not x.startswith("--") and cmd[i - 1] != "--with"
    )  # fmt: skip
    assert interpreter == "python", cmd
