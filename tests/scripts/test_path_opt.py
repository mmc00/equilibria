"""PATH options: the builders inline them, and every result records them (phase 2b).

The 15x10 altertax MCP shock needs the tight option set (one-shot, no ramp)
that made its July 2026 references (notes: gtap7_15x10_altertax_parity_close).
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from equilibria.sources import GAMS_TOOLS_DIR, dataset_dir

ROOT = Path(__file__).resolve().parents[2]
RESULTS = dataset_dir("gtap7_3x3")


def _tools(monkeypatch: Any, name: str) -> Any:
    monkeypatch.syspath_prepend(str(GAMS_TOOLS_DIR))
    return importlib.import_module(name)


def _altertax(tmp_path: Path, *flags: str) -> str:
    subprocess.run(
        [sys.executable, str(GAMS_TOOLS_DIR / "build_gtap7_altertax_neos_bundle.py"),
         "--dataset", "gtap7_3x3", "--ifsub", "1", "--out-dir", str(tmp_path), *flags],
        check=True, cwd=ROOT, capture_output=True,
    )  # fmt: skip
    (built,) = tmp_path.glob("comp_*.gms")
    return built.read_text()


@pytest.mark.gams  # the builders inline data through the gams binary; CI has none
def test_altertax_path_opt_is_one_shot_and_inlined(
    monkeypatch: Any, tmp_path: Path
) -> None:
    alt = _tools(monkeypatch, "build_gtap7_altertax_neos_bundle")
    text = _altertax(tmp_path, "--path-opt")
    assert "$onecho > path.opt\n" + alt.PATH_OPT + "$offecho" in text
    shock = text[text.index("if(sameas(tsim,'shock')") :]
    assert shock.index("gtap.optfile = 1 ;") < shock.index("imptx.fx")
    assert "hstep" not in text  # no homotopy ramp
    assert not (tmp_path / "path.opt").exists()  # the bundle writes its own


def test_bundle_path_opt_reads_the_last_onecho(monkeypatch: Any) -> None:
    rgr = _tools(monkeypatch, "regen_gams_reference")
    gtap = (RESULTS / "gams_mcp_gtap_ifsub1_fbep-header" / "model.gms").read_text()
    opt = rgr.bundle_path_opt(gtap)
    assert opt is not None and opt.startswith("convergence_tolerance 1e-9\n")
    alter = (RESULTS / "gams_mcp_altertax_ifsub1_fbep-header" / "model.gms").read_text()
    assert rgr.bundle_path_opt(alter) is None
