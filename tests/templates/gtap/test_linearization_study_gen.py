"""The study page generator must always produce a page (marking absent fixtures
'—'), list the compared variables, and cite the source."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/gtap"))
sys.path.insert(0, str(ROOT / "src"))


def test_page_generates_with_no_fixtures(tmp_path):
    from gen_linearization_study import build_page

    empty = tmp_path / "fixtures"
    empty.mkdir()
    out = tmp_path / "page.md"
    md = build_page(empty, out)
    assert out.exists()
    # scope + provenance always present
    assert "linearization" in md.lower()
    assert "van der Mensbrugghe" in md
    # the shock-sweep axis is documented even with no data
    for pct in ("10", "3", "1", "0.3", "0.1"):
        assert pct in md
    # absent data is an em-dash, not a crash or a fake number
    assert "—" in md


def test_variable_list_is_shown(tmp_path):
    from gen_linearization_study import build_page

    out = tmp_path / "page.md"
    md = build_page(tmp_path, out)
    # the mapped quantity vars are named
    for gv in ("qfd", "qxs", "qgdp", "qva"):
        assert gv in md


def test_gragg_axis_is_documented(tmp_path):
    from gen_linearization_study import build_page

    out = tmp_path / "page.md"
    md = build_page(tmp_path, out)
    # the Gragg refinement axis (steps) appears
    for steps in ("4", "8", "16", "32", "64"):
        assert steps in md
