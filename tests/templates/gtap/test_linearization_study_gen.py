"""The study page generator must always produce a page (marking absent fixtures
'—'), list the compared variables, and cite the source."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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


def test_python_solves_with_the_closure_of_the_fixture_it_reads(monkeypatch, tmp_path):
    """The GEMPACK fixture is looked up by its closure and Python solves with that
    same closure, through the paper's comparator (``_solve_shock``). MAR-223: Python
    ran capFix against a capFlex fixture and the page published 52-76 %."""
    import types

    import gen_linearization_study as g

    asked, solved = {}, []

    def fake_find(dataset, tool, **kw):
        asked.update(kw)
        return None

    monkeypatch.setattr(g, "find_reference", fake_find)
    g._sl4_fixture(tmp_path, "gtap7_3x3", "10", "8 16 32")
    assert asked["closure"] == g.FIXTURE_CLOSURE

    fake = types.ModuleType("templates.gtap.test_gtap7_gempack_parity")
    fake._solve_shock = lambda *a, **k: solved.append((a, k)) or ("m", 1)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    g._solve_shock_10.cache_clear()
    assert g._solve_shock_10("gtap7_3x3") == "m"
    ((args, kw),) = solved
    assert args == ("gtap7_3x3", 1, g.FIXTURE_CLOSURE) and kw == {}


def test_a_failed_solve_is_not_measured(monkeypatch):
    import types

    import gen_linearization_study as g

    fake = types.ModuleType("templates.gtap.test_gtap7_gempack_parity")
    fake._solve_shock = lambda *a, **k: ("m", 0)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    g._solve_shock_10.cache_clear()
    with pytest.raises(RuntimeError, match="did not solve"):
        g._solve_shock_10("gtap7_3x3")
