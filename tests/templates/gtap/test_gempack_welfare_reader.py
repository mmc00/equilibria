"""The welfare reader pulls the EV decomposition from a decomp.har (WELVIEW) by
header NAME, sums into 3 canonical branches, and is a clean no-op when absent."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/gtap"))
sys.path.insert(0, str(ROOT / "src"))


def _write_synth_decomp(path: Path):
    """Minimal HAR with a named EV header (regions × components)."""
    from equilibria.babel.har.symbols import HeaderArray
    from equilibria.babel.har.writer import write_har

    # regions R1,R2 ; components alloc,tot,invsav ; EV$ values (Fortran order)
    arr = np.array([[1.0, -0.5, 0.2], [-0.3, 0.4, 0.1]], dtype=np.float32)
    ha = HeaderArray(
        name="A",
        coeff_name="EV",
        long_name="EV decomposition",
        array=arr,
        set_names=["REG", "COMP"],
        set_elements=[["R1", "R2"], ["alloc", "tot", "invsav"]],
    )
    write_har(str(path), {"A": ha})


def test_welfare_reader_absent_is_empty():
    from gempack_reference import gempack_welfare_ev

    assert gempack_welfare_ev(str(ROOT / "does_not_exist.har")) == {}


def test_welfare_reader_sums_branches(tmp_path):
    from gempack_reference import gempack_welfare_ev

    p = tmp_path / "decomp.har"
    _write_synth_decomp(p)
    ev = gempack_welfare_ev(str(p))
    # returns per-region branch dict with a 'total' == sum of branches
    assert set(ev) == {"R1", "R2"}
    assert ev["R1"]["total"] == pytest.approx(1.0 - 0.5 + 0.2, abs=1e-5)
    assert ev["R2"]["alloc"] == pytest.approx(-0.3, abs=1e-5)
    assert ev["R2"]["tot"] == pytest.approx(0.4, abs=1e-5)
