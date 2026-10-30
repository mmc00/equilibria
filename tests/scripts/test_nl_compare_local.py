"""nl_compare writes the GAMS .nl locally with the ifSUB it was asked for (#23)."""

from __future__ import annotations

import importlib
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from equilibria.sources import dataset_dir

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gtap" / "nl_compare.py"
needs_gams = pytest.mark.skipif(shutil.which("gams") is None, reason="GAMS not on PATH")


def _nl_compare(monkeypatch: Any) -> Any:
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    return importlib.import_module("nl_compare")


@pytest.mark.parametrize("ifsub", [0, 1])
def test_standalone_carries_ifsub(monkeypatch: Any, ifsub: int) -> None:
    # comp.gms sets ifSUB itself; GAMS keeps the last $setGlobal, so every
    # setting in the file must say the value asked for, not just the header
    nlc = _nl_compare(monkeypatch)
    datasets = importlib.import_module("_parity_datasets").DATASETS

    gms = nlc._build_standalone_gms(
        "base",
        agg_gdx=datasets["gtap7_3x3"].agg_gdx,
        har_dir=dataset_dir("gtap7_3x3"),
        ifsub=ifsub,
    )
    values = re.findall(r"^\$setGlobal\s+ifSUB\s+(\d+)\s*$", gms, flags=re.MULTILINE)
    assert values and set(values) == {str(ifsub)}, values


@needs_gams
def test_gams_local_writes_nl(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            "gtap7_3x3",
            "--phase",
            "base",
            "--ifsub",
            "1",
            "--gams-local",
            "--skip-python",
            "--out-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    names = {p.name for p in tmp_path.iterdir()}
    assert proc.returncode == 0, (proc.stdout[-2000:], proc.stderr[-2000:])
    # exactly the files write_source records: nothing else is left behind
    assert names == {
        "gams_base.nl",
        "gams_base.row",
        "gams_base.col",
        "gams_base.gms",
    }, names
    assert "$setGlobal ifSUB     1" in (tmp_path / "gams_base.gms").read_text()
