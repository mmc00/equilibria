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
    g._solve_at.cache_clear()
    assert g._solve_at("gtap7_3x3", "10") == "m"
    assert g._solve_at("gtap7_3x3", "0.3") == "m"
    (args10, kw10), (args03, kw03) = solved
    # 10 % is the gate's run (the default tariff); the other sizes go through uniform
    assert args10 == args03 == ("gtap7_3x3", 1, g.FIXTURE_CLOSURE)
    assert kw10 == {} and kw03 == {"tariff_pct": 0.3}


def test_a_failed_solve_is_not_measured(monkeypatch):
    import types

    import gen_linearization_study as g

    fake = types.ModuleType("templates.gtap.test_gtap7_gempack_parity")
    fake._solve_shock = lambda *a, **k: ("m", 0)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    g._solve_at.cache_clear()
    with pytest.raises(RuntimeError, match="did not solve"):
        g._solve_at("gtap7_3x3", "3")


@pytest.mark.parametrize(
    "shock,tol", [("10", 0.01), ("3", 0.003), ("1", 0.001), ("0.1", 0.0001)]
)
def test_the_scaled_tolerance_shrinks_with_the_shock(shock, tol):
    """At 0.1 % almost no cell moves 1 pp, so a fixed 1 pp band would read ~100 %
    whatever Python does: the second number scales the band with the shock."""
    import gen_linearization_study as g

    assert g._scaled_tol(shock) == pytest.approx(tol, rel=1e-12)


def test_every_shock_size_is_measured(monkeypatch, tmp_path):
    """No size reads 'py-shock pending': each one is measured at its own size."""
    import gen_linearization_study as g

    seen = []
    monkeypatch.setattr(g, "_sl4_fixture", lambda *a: tmp_path / "x.har")
    monkeypatch.setattr(g, "_seed_gdx", lambda ds: tmp_path / "out.gdx")
    monkeypatch.setattr(
        g, "_measure_within_pp", lambda ds, sl4, s: seen.append(s) or ("99%", "97%")
    )
    for s in g.SWEEP_SHOCKS:
        assert g._match_cell(tmp_path, "gtap7_3x3", s, "8 16 32") == "99% / 97%"
    assert seen == g.SWEEP_SHOCKS


def test_nonlinearity_reads_the_scaled_numbers():
    import gen_linearization_study as g

    assert g._nonlinearity(["99% / 90%", "—", "—", "—", "100% / 98%"]) == "8pp"
    assert g._nonlinearity(["99% / 90%", "—", "—", "—", "—"]) == "—"
