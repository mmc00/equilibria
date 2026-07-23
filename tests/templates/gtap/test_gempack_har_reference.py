"""GTAP7 GEMPACK reference path — HAR reader/mapper unit tests.

Exercises the updated.har → Var-cells chain end-to-end against a tiny SYNTHETIC
fixture, so the whole GEMPACK path is testable without a Windows-produced ref.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests/fixtures/gtap7_gempack"))


def test_synthetic_updated_har_roundtrips(tmp_path):
    from make_synthetic_updated_har import build_synthetic_updated_har

    from equilibria.babel.har.reader import get_header_names, read_har

    out = tmp_path / "updated_synthetic.har"
    build_synthetic_updated_har(str(out))
    names = get_header_names(str(out))
    assert "VDFB" in names, f"expected VDFB header, got {names}"
    headers = read_har(str(out))
    vdfb = headers["VDFB"]
    assert vdfb.array.shape == (2, 2), f"expected 2x2, got {vdfb.shape}"
    assert vdfb.set_elements[0] == ["USA", "ROW"]


def test_gempack_levels_maps_var_to_cells(tmp_path):
    from make_synthetic_updated_har import build_synthetic_updated_har

    sys.path.insert(0, str(ROOT / "scripts/gtap"))
    import pytest as _pt
    from gempack_reference import gempack_levels

    out = tmp_path / "u.har"
    build_synthetic_updated_har(str(out))
    # 'qfd' (firm domestic demand) maps to the VDFB header in the synthetic ref
    cells = gempack_levels(str(out), "qfd")
    assert cells[("USA", "Food")] == 10.0
    assert cells[("ROW", "Mnfcs")] == 40.0
    # an aggregate-only var raises (no cell-by-cell header)
    with _pt.raises(KeyError):
        gempack_levels(str(out), "walras")


def test_gempack_qty_pct_reorders_to_python_index():
    """gempack_qty_pct reads a real sl4dump, maps a GEMPACK quantity to its Python
    Var key order, and returns %-changes as fractions."""
    import pytest as _pt

    sys.path.insert(0, str(ROOT / "scripts/gtap"))
    from gempack_reference import gempack_qty_pct

    sl4 = ROOT / "tests/fixtures/gtap7_gempack/sl4dump_gtap7_3x3_tm10.har"
    if not sl4.exists():
        _pt.skip(f"sl4dump fixture missing: {sl4}")

    # qfd [COMM,ACTS,REG] -> xda key order (r,c,a); values are fractions (~±0.25)
    qfd = gempack_qty_pct(str(sl4), "qfd")
    assert len(qfd) == 27, f"3x3 qfd should have 27 cells, got {len(qfd)}"
    # a representative cell key must be in Python (r,c,a) order
    assert ("USA", "Food", "Mnfcs") in qfd
    assert all(abs(v) < 2.0 for v in qfd.values()), (
        "%-changes should be fractions, not percents"
    )
    # qxs (bilateral trade) and qgdp (macro GDP) are in the expanded map
    assert len(gempack_qty_pct(str(sl4), "qxs")) == 27
    assert len(gempack_qty_pct(str(sl4), "qgdp")) == 3
    # qo was dropped from the map (no clean 1:1 Python Var) -> KeyError, not fabricated
    with _pt.raises(KeyError):
        gempack_qty_pct(str(sl4), "qo")
