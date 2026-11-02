"""regen_nus333: the nus333 GAMS reference, built from its HARs and solved locally.

GAMS is a stub here: it writes COMP.gdx where --outDir says and the listing
where o= says, with the model status the test chooses.
"""

from __future__ import annotations

import importlib
import stat
from pathlib import Path
from typing import Any

import pytest

from equilibria.sources import GAMS_TOOLS_DIR


@pytest.fixture
def regen(monkeypatch: Any) -> Any:
    monkeypatch.syspath_prepend(str(GAMS_TOOLS_DIR))
    return importlib.import_module("regen_nus333")


def _gams(tmp_path: Path, status: int) -> Path:
    stub = tmp_path / "bin" / "gams"
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text(
        "#!/bin/sh\n"
        'for a in "$@"; do case "$a" in\n'
        '  --outDir=*) out="${a#--outDir=}" ;;\n'
        '  o=*) lst="${a#o=}" ;;\n'
        "esac; done\n"
        'echo gdx > "$out/COMP.gdx"\n'
        f'printf "GAMS 53.5.0\\n**** MODEL STATUS      {status} Optimal\\n" > "$lst"\n'
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return stub


def test_writes_out_gdx(regen: Any, tmp_path: Path) -> None:
    out = tmp_path / "gams_mcp_nus333"
    assert regen.main(["--out", str(out), "--gams", str(_gams(tmp_path, 1))]) == 0
    assert (out / "out.gdx").read_text() == "gdx\n"
    assert sorted(p.name for p in out.iterdir()) == ["out.gdx"]


def test_unsolved_leaves_no_folder(regen: Any, tmp_path: Path) -> None:
    out = tmp_path / "gams_mcp_nus333"
    assert regen.main(["--out", str(out), "--gams", str(_gams(tmp_path, 5))]) == 1
    assert not out.exists()


def test_bundle_is_the_nus333_har(regen: Any, tmp_path: Path) -> None:
    # the three GDX files getData.gms reads, written from data/gtap7/nus333
    files = regen.build_bundle(tmp_path)
    assert sorted(p.name for p in files) == [
        "nus333Dat.gdx",
        "nus333Prm.gdx",
        "nus333Sets.gdx",
    ]
