"""Levels gate of the ALTERTAX family solution against GEMPACK (``sl4levels_altax.har``)."""

from __future__ import annotations

import pytest

from equilibria.family.altertax.compare import compare_levels
from equilibria.family.altertax.tolerances import TOL_LEVELS
from tests.family._prep import ROOT

FIXTURES = ROOT / "tests/fixtures/gtap7_gempack/burfisher_gragg"


def test_levels_match_gempack(altax_run):
    report = compare_levels(altax_run, FIXTURES)
    assert report.overall["cells"] > 0
    worst = sorted(report.cells, key=lambda c: (c.d != c.d, c.d), reverse=True)[:10]
    assert report.overall["max"] <= TOL_LEVELS, [
        (c.variable, c.period, c.index, c.a, c.b, c.d) for c in worst
    ]
    assert all(e.why.strip() for e in report.exclusions)
    assert all(x.why.strip() for x in report.excluded_cells)


def test_compare_missing_reference_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="altax_sl4levels.har"):
        compare_levels(None, tmp_path)  # type: ignore[arg-type]
