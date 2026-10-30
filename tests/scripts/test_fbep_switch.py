"""One switch, --fbep header|zero, for every GAMS result generator (#187).

``header``: the FBEP/FTRV headers of the dataset's basedata.har are injected
after getData.gms's ``fbep = 0`` (what the MCP references have used since #24).
``zero``: the official getData.gms, ``fbep = 0`` and ``ftrv = evfp - evfb``.
"""

from __future__ import annotations

import hashlib
import importlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from equilibria.sources import GAMS_TOOLS_DIR, dataset_dir

ROOT = Path(__file__).resolve().parents[2]
DS = "gtap7_3x3"
RESULTS = dataset_dir(DS)
BUILDERS = {
    "gtap": GAMS_TOOLS_DIR / "build_gtap7_pure_local_bundle.py",
    "altertax": GAMS_TOOLS_DIR / "build_gtap7_altertax_neos_bundle.py",
}
FBEP_ZERO = "fbep(fp,a0,r) = 0 ;"
FTRV_LINE = "ftrv(fp,a0,r) = evfp(fp,a0,r) - evfb(fp,a0,r) ;"
FBEP_EU_LAND = re.compile(
    r"""FBEP\(['"]Land['"],['"](a_)?Food['"],['"]EU_28['"]\) = -24050\.85"""
)


def _bundle(mode: str, out: Path, *fbep: str, dataset: str = DS) -> str:
    subprocess.run(
        [sys.executable, str(BUILDERS[mode]), "--dataset", dataset, "--ifsub", "1",
         "--out-dir", str(out), *fbep],
        check=True, cwd=ROOT, capture_output=True,
    )  # fmt: skip
    (built,) = out.glob("comp_*.gms")
    return built.read_text()


def _tools(monkeypatch: Any, name: str) -> Any:
    monkeypatch.syspath_prepend(str(GAMS_TOOLS_DIR))
    return importlib.import_module(name)


@pytest.mark.parametrize("mode", ["gtap", "altertax"])
def test_default_bundle_unchanged(mode: str, tmp_path: Path) -> None:
    default = _bundle(mode, tmp_path / "default")
    header = _bundle(mode, tmp_path / "header", "--fbep", "header")
    assert default == header
    recorded = RESULTS / f"gams_mcp_{mode}_ifsub1_fbep-header" / "model.gms"
    assert hashlib.sha256(default.encode()).hexdigest() == (
        hashlib.sha256(recorded.read_bytes()).hexdigest()
    )


# gtap7_3x4's own v7 GDX carries FBEP: the inliner writes it after fbep = 0 too
@pytest.mark.parametrize("dataset", [DS, "gtap7_3x4"])
@pytest.mark.parametrize("mode", ["gtap", "altertax"])
def test_zero_bundle_has_no_injection(mode: str, dataset: str, tmp_path: Path) -> None:
    text = _bundle(mode, tmp_path, "--fbep", "zero", dataset=dataset)
    assert FBEP_ZERO in text
    after = text[text.index(FBEP_ZERO) :]
    assert re.search(r"^\s*FBEP\(", after, flags=re.MULTILINE) is None
    assert re.search(r"^\s*FTRV\(", after, flags=re.MULTILINE) is None


def _standalone(monkeypatch: Any, **kw: Any) -> str:
    nls = _tools(monkeypatch, "nl_standalone")
    return nls._build_standalone_gms(
        "base", agg_gdx=RESULTS / "v7_consolidated.gdx", har_dir=RESULTS, ifsub=0, **kw
    )


@pytest.mark.needs_gdxdump
def test_nl_standalone_default_unchanged(monkeypatch: Any) -> None:
    recorded = RESULTS / "gams_nl_altertax_ifsub0_fbep-zero" / "gams_base.gms"
    assert _standalone(monkeypatch) == recorded.read_text()


@pytest.mark.needs_gdxdump
def test_nl_standalone_header_injects(monkeypatch: Any) -> None:
    text = _standalone(monkeypatch, fbep="header")
    hit = FBEP_EU_LAND.search(text)
    assert hit is not None
    assert text.index(FTRV_LINE) < hit.start()


def _fake_gdxdump(tmp_path: Path, fs_value: float) -> str:
    # ytax as gdxdump Format=csv prints it: region, stream, period, value
    rows = "\n".join(
        f'"USA","{s}","{t}",{v}'
        for s, v in (("ft", 1.5), ("fs", fs_value))
        for t in ("base", "check", "shock")
    )
    exe = tmp_path / "gdxdump"
    exe.write_text(f'#!/bin/sh\ncat <<\'EOF\'\n"r","ytx","t","Val"\n{rows}\nEOF\n')
    exe.chmod(0o755)
    return str(exe)


@pytest.mark.needs_gdxdump
def test_verify_zero_rejects_subsidy(monkeypatch: Any) -> None:
    rgr = _tools(monkeypatch, "regen_gams_reference")
    gdx = RESULTS / "gams_mcp_altertax_ifsub1_fbep-header" / "out.gdx"
    gdxdump = str(Path(rgr.shutil.which("gams")).with_name("gdxdump"))
    rgr._verify_subsidy_aware(gdxdump, gdx, "header")
    with pytest.raises(SystemExit, match="fs"):
        rgr._verify_subsidy_aware(gdxdump, gdx, "zero")


def test_verify_header_rejects_blind(monkeypatch: Any, tmp_path: Path) -> None:
    rgr = _tools(monkeypatch, "regen_gams_reference")
    blind = _fake_gdxdump(tmp_path, 0.0)
    rgr._verify_subsidy_aware(blind, tmp_path / "out.gdx", "zero")
    with pytest.raises(SystemExit, match="subsidy-blind"):
        rgr._verify_subsidy_aware(blind, tmp_path / "out.gdx", "header")
