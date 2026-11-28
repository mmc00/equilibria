"""Reading MANAGE's Ghana calibration from the GAMS run, in GAMS's solve units.

GAMS calibrates and solves with GDP = 1e5 (scaleSAM) and multiplies back the
``.l`` of a listed set of symbols by ``reScale`` before writing GHA_BaU.gdx
(inc/cal/rescale.inc); bounds are never rescaled. Python works in solve units.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.babel.gdx.reader import read_gdx, read_variable_values
from equilibria.templates.manage.convert import read_dict, read_levels
from equilibria.templates.manage.data import gams_levels, load_bau

RESCALE = 3.95423752973879e-4


@pytest.fixture(scope="module")
def data(bau_path: Path):  # noqa: ANN201
    return load_bau(bau_path)


def test_scale_constants(data) -> None:  # noqa: ANN001
    assert data.scale["reScale"] == pytest.approx(RESCALE, rel=1e-14)
    assert data.scale["inScale"] == pytest.approx(2.52893255015477, rel=1e-14)


def test_base_in_solve_units(data, bau_path: Path) -> None:  # noqa: ANN001
    raw = read_variable_values(read_gdx(bau_path), "gdpmp")[("2014",)]["level"]
    assert data.base["gdpmp"][()] == pytest.approx(raw / RESCALE, rel=1e-12)
    assert data.base["pgdpmp"][()] == pytest.approx(1.0, rel=1e-12)  # a price


def test_sets_sizes(data) -> None:  # noqa: ANN001
    # keys keep the GDX's case, the case of the GAMS declaration (fppar, not fpPar)
    sizes = {k: len(data.sets[k]) for k in ("a", "i", "fp", "fppar", "kNest", "g")}
    assert sizes == {"a": 61, "i": 60, "fp": 41, "fppar": 19, "kNest": 17, "g": 16}
    assert data.sets["h"] == (("h-hhold",),)


def test_fixed_cells_agree_with_their_bounds(data, bau_path: Path) -> None:  # noqa: ANN001
    """A fixed cell's bound was never rescaled, its level was: level/factor must
    equal the bound for every fixed cell, which checks the rescale table's domains
    (atFmSup is rescaled only for inst, not for 'sumInst')."""
    gdx = read_gdx(bau_path)
    checked = 0
    for name, cells in data.fixed.items():
        records = read_variable_values(gdx, name)
        for key, value in cells.items():
            assert value == pytest.approx(
                records[key]["lower"], rel=1e-12, abs=1e-300
            ), (
                name,
                key,
            )
            checked += 1
    assert checked > 1000


def test_calibration_reads_no_free_2015_level(data, dict_path: Path) -> None:  # noqa: ANN001
    """The 9,452 columns of the 2015 solve are what Python must find itself."""
    free = {(fam, idx[:-1]) for fam, idx in read_dict(dict_path).cols if idx}
    exposed = {
        (name, key[:-1])
        for name, cells in data.fixed.items()
        for key in cells
        if key and key[-1] == "2015"
    }
    assert not (free & exposed)
    assert "2015" not in str(sorted(data.base))  # base has no year index


def test_gdx_2015_matches_convert_solution(
    bau_path: Path, dict_path: Path, run_dir: Path
) -> None:
    """Every column of the 2015 solve, read from the GDX and converted to solve
    units, equals the CONVERT solution (written in solve units by GAMS)."""
    gdx = gams_levels(bau_path, "2015")
    names = read_dict(dict_path)
    solution = read_levels(run_dir / "gha2015_solution.gms")
    worst = 0.0
    for j, (fam, idx) in enumerate(names.cols, start=1):
        key = idx[:-1] if idx else idx
        ours = gdx.get(fam, {}).get(key, 0.0)
        ref = solution.get(j, 0.0)
        worst = max(worst, abs(ours - ref) / max(1.0, abs(ref)))
    assert worst <= 1e-12
