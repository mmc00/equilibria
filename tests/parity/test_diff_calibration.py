"""diff_calibration reads Python's own calibration from the case the MCP gate
builds (gempack FBEP basis, altertax elasticities) against the data/ reference
(MAR-219). It used to build the old single-period model and read the June refs,
so ytax[fs] came out with the opposite sign."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from equilibria.sources import find_reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
_spec = importlib.util.spec_from_file_location(
    "diff_calibration", ROOT / "scripts/gtap/diff_calibration.py"
)
assert _spec is not None and _spec.loader is not None
DC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DC)

_FOLDER = find_reference(
    "gtap7_3x3", "gams", form="mcp", mode="altertax", ifsub=0, fbep="header"
)


class _Comp(dict):
    def __getitem__(self, k):
        return dict.__getitem__(self, k)


class _M:
    facty = _Comp({("USA", "check"): 2.0})
    ytax = _Comp({("USA", "fs"): 3.0})


_BLOCK = {"betap": {("USA",): 0.7}}


def test_read_takes_the_period_cell_then_the_bare_one():
    assert DC._read(_M(), "facty", ("USA",), "check") == 2.0
    assert DC._read(_M(), "ytax", ("USA", "fs"), "check") == 3.0
    assert DC._read(_M(), "facty", ("EU_28",), "check") is None


def test_read_falls_back_to_the_block_calibration():
    assert DC._read(_M(), "betap", ("USA",), "check", _BLOCK) == 0.7


def test_the_default_gdx_is_the_data_reference():
    assert _FOLDER is not None
    assert DC._default_gdx("gtap7_3x3", "altertax", 0) == _FOLDER / "out.gdx"


@pytest.mark.skipif(_FOLDER is None, reason="GAMS MCP altertax ref absent")
@pytest.mark.needs_gdxdump  # gams_levels reads the GAMS GDX through it
def test_gtap7_3x3_altertax_is_clean():
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/gtap/diff_calibration.py",
            "--dataset",
            "gtap7_3x3",
            "--period",
            "check",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["status"] == "clean", out["headline"]
