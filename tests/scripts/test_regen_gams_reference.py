"""regen_gams_reference.py solves one GAMS reference into a folder (ADR-0010)."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gtap" / "regen_gams_reference.py"
needs_gams = pytest.mark.skipif(shutil.which("gams") is None, reason="GAMS not on PATH")


def _module(monkeypatch: Any) -> Any:
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("regen_gams_reference", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(dataset: str, mode: str, form: str, ifsub: int, out: Path) -> int:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            dataset,
            "--mode",
            mode,
            "--form",
            form,
            "--ifsub",
            str(ifsub),
            "--out",
            str(out),
        ],
        cwd=ROOT,
    ).returncode


BUNDLE = """   ifMCP       "Set to 1 to solve using MCP"           / 1 /
if(ifMCP,
   solve gtap using mcp ;
else
   solve gtap using nlp maximizing walras ;
) ;
"""


def test_nlp_form_switches_off_mcp_and_uses_ipopt(monkeypatch: Any) -> None:
    out = _module(monkeypatch).to_nlp(BUNDLE)
    assert "/ 0 /" in out and "/ 1 /" not in out
    assert "   option nlp=ipopt;\n   solve gtap using nlp maximizing walras ;" in out


def test_nlp_form_refuses_a_bundle_without_the_switch(monkeypatch: Any) -> None:
    with pytest.raises(SystemExit):
        _module(monkeypatch).to_nlp("solve gtap using mcp ;\n")


@needs_gams
def test_mcp_over_demo_cap_fails_clean(tmp_path: Path) -> None:
    # gtap7_3x3 pure ifSUB=0 exceeds PATH's 1000-row demo cap (measured 2026-10-29)
    assert _run("gtap7_3x3", "gtap", "mcp", 0, tmp_path) != 0
    assert not (tmp_path / "out.gdx").exists()


@needs_gams
def test_nlp_altertax_writes_gdx(tmp_path: Path) -> None:
    assert _run("gtap7_3x3", "altertax", "nlp", 1, tmp_path) == 0
    assert {p.name for p in tmp_path.iterdir()} <= {"out.gdx", "model.gms", "path.opt"}
    assert (tmp_path / "out.gdx").is_file() and (tmp_path / "model.gms").is_file()
