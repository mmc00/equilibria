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
SCRIPT = ROOT / "data" / "gtap7" / "gams" / "tools" / "regen_gams_reference.py"
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
            "--fbep",
            "header",
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


def test_nlp_form_can_use_conopt(monkeypatch: Any) -> None:
    # 20x41: IPOPT does not fit in memory and PATH stalls; CONOPT solves it (#20x41)
    out = _module(monkeypatch).to_nlp(BUNDLE, solver="conopt")
    assert "   option nlp=conopt;\n   solve gtap using nlp maximizing walras ;" in out
    assert "ipopt" not in out


LOOP = "zcons.l(r,i,h,t) = 1 ;\nloop(tsim,\n   solve gtap using mcp ;\n) ;\n"


def test_scale_zcons_scales_by_the_calibrated_level_before_the_loop(
    monkeypatch: Any,
) -> None:
    # SUBPAR at the GTAP floor (CHE 20x41) puts zcons at ~1e-10, below the
    # solvers' tolerances; the original model has scaleopt=1 but no .scale
    out = _module(monkeypatch).scale_zcons(LOOP)
    before, after = out.split("loop(tsim,", 1)
    for var in ("zcons", "zconseq"):
        assert (
            f"{var}.scale(r,i,h,t)$(xaFlag(r,i,h) and zcons.l(r,i,h,t) gt 0)"
            " = zcons.l(r,i,h,t) ;" in before
        )
    assert ".scale" not in after
    assert before.startswith("zcons.l(r,i,h,t) = 1 ;\n")


def test_scale_zcons_refuses_a_bundle_without_the_loop(monkeypatch: Any) -> None:
    with pytest.raises(SystemExit):
        _module(monkeypatch).scale_zcons("solve gtap using mcp ;\n")
