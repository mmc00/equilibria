"""har_to_consolidated: the GAMS bundles' v7_consolidated.gdx, rebuilt from the HARs.

Two oracles, neither written by the converter:
  - the v7_consolidated.gdx committed for gtap7_3x3/5x5/10x7/20x41 (GTAPAgg
    processed schema, 53 symbols): converting the same dataset's HARs must give
    every set and every parameter record back;
  - for 9x10, the GTAP Standard 7 GAMS distribution's own 9x10Dat.gdx /
    9x10Prm.gdx (upper-case names): every market and elasticity record the
    converter writes must equal the distribution's value.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from equilibria.babel.gdx.reader import (
    read_gdx,
    read_parameter_values,
    read_set_elements,
)
from equilibria.sources import GAMS_TOOLS_DIR, dataset_dir

PROCESSED = ("gtap7_3x3", "gtap7_5x5", "gtap7_10x7", "gtap7_20x41")


def _tool():
    if str(GAMS_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(GAMS_TOOLS_DIR))
    return importlib.import_module("har_to_consolidated")


def _contents(gdx: Path) -> dict[str, object]:
    g = read_gdx(gdx)
    out: dict[str, object] = {}
    for s in g["symbols"]:
        name = s["name"]
        if s["type"] == 0:
            out[name] = [tuple(e) for e in read_set_elements(g, name)]
        else:
            out[name] = dict(read_parameter_values(g, name))
    return out


@pytest.mark.parametrize("dataset", PROCESSED)
def test_converter_rebuilds_the_committed_gdx(dataset, tmp_path):
    d = dataset_dir(dataset)
    out = tmp_path / "v7_consolidated.gdx"
    _tool().write(d, out)
    want = _contents(d / "v7_consolidated.gdx")
    got = _contents(out)
    assert sorted(got) == sorted(want)
    for name, value in want.items():
        assert got[name] == value, name


def test_9x10_matches_the_gams_distribution(tmp_path):
    d = dataset_dir("9x10")
    out = tmp_path / "v7_consolidated.gdx"
    _tool().write(d, out)
    got = _contents(out)
    assert len(got) == 53
    dat = read_gdx(d / "9x10Dat.gdx")
    prm = read_gdx(d / "9x10Prm.gdx")
    checked = 0
    for name, value in got.items():
        if not isinstance(value, dict):
            continue
        src = dat if name.upper() in {s["name"] for s in dat["symbols"]} else prm
        ref = {
            k: v
            for k, v in read_parameter_values(src, name.upper()).items()
            if v != 0.0
        }
        assert value == ref, name
        checked += 1
    assert checked >= 45
