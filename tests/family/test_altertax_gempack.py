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
    assert report.overall["missing_in_a"] == report.overall["missing_in_b"] == 0
    assert not report.unmatched
    assert {"pre", "post"} <= set(report.periods_compared)
    # the family side also carries its 'base' solve, which GEMPACK has no period for
    assert report.periods_only_in_a == ["base"]
    assert not report.periods_only_in_b
    assert all(e.why.strip() for e in report.exclusions)
    assert all(x.why.strip() for x in report.excluded_cells)
    assert all(why.strip() for why in report.excluded.values())


def test_compare_missing_reference_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="sl4levels_altax.har"):
        compare_levels(None, tmp_path)  # type: ignore[arg-type]
