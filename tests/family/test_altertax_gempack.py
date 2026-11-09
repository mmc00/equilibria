"""Levels gate of the ALTERTAX family solution against GEMPACK (``sl4levels_altax.har``)."""

from __future__ import annotations

import gzip
import shutil

import pytest

from equilibria.family.altertax.compare import compare_levels
from equilibria.family.altertax.gates import levels_ok
from equilibria.family.altertax.tolerances import TOL_LEVELS
from tests.family._prep import ROOT

FIXTURES = ROOT / "tests/fixtures/gtap7_gempack/burfisher_gragg"


def _worst(report):
    worst = sorted(report.cells, key=lambda c: (c.d != c.d, c.d), reverse=True)[:10]
    return [(c.variable, c.period, c.index, c.a, c.b, c.d) for c in worst]


def test_levels_match_gempack(altax_run):
    """The whole predicate (max within TOL_LEVELS, nothing missing or unmatched, pre and
    post compared, ``base`` the only family-only period, every exclusion with its reason)
    is ``gates.levels_ok``, the one the command uses."""
    report = compare_levels(altax_run, FIXTURES)
    assert report.overall["cells"] > 0
    assert report.overall["max"] <= TOL_LEVELS, _worst(report)
    assert levels_ok(report), _worst(report)


def test_levels_fall_back_to_the_sl4_when_there_is_no_levels_har(altax_run, tmp_path):
    """A GEMPACK folder with ``solution.sl4`` and no ``sl4levels.har``: the levels are read
    from the solution itself (``from_har`` reads an ``.sl4``). The burfisher altax run kept
    no levels in its ``.sl4`` (no ``PLEV``: that is why ``sl4levels_altax.har`` was
    exported), so here the fallback is shown by the reader's own clear error about it."""
    from equilibria.compare.errors import CompareError
    from equilibria.family.altertax.gates import gempack_files

    shutil.copy(FIXTURES / "altax.upd", tmp_path / "updated.har")
    with (
        gzip.open(FIXTURES / "altax.sl4.gz", "rb") as a,
        (tmp_path / "solution.sl4").open("wb") as b,
    ):
        shutil.copyfileobj(a, b)
    for name in ("summary", "decomp", "volume"):
        shutil.copy(FIXTURES / f"altax_{name}.har", tmp_path / f"{name}.har")
    assert gempack_files(tmp_path)["levels"].name == "solution.sl4"
    with pytest.raises(CompareError, match="PLEV"):
        compare_levels(altax_run, tmp_path)


def test_compare_missing_reference_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="sl4levels_altax.har"):
        compare_levels(None, tmp_path)  # type: ignore[arg-type]
