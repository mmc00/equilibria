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
