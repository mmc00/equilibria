"""GTAP7 vs RunGTAP (GEMPACK) cell-by-cell parity gate. LOCAL-only.

Measures at the SAME TOL=1e-2 as the GAMS gates; the linearized↔levels gap is
carried by the per-page floor, not the tolerance. SKIPs when a row's updated.har
fixture is absent, so it never blocks the parity stamp on a machine without the
Windows-produced ref.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/gtap"))
sys.path.insert(0, str(ROOT / "src"))
from coverage_matrix import rows_for  # noqa: E402

GEMPACK_ROWS = rows_for("gtap7", "gempack", kind="mcp")


def test_no_gempack_rows_is_a_clean_skip():
    """Until reference='gempack' rows are added (needs a real updated.har), the
    gate has nothing to run — that is a SKIP, not a failure."""
    if not GEMPACK_ROWS:
        pytest.skip("no reference='gempack' rows yet — awaiting RunGTAP updated.har")


@pytest.mark.parametrize(
    "row", GEMPACK_ROWS, ids=lambda r: f"{r.dataset}-ifsub{r.ifsub}"
)
def test_gtap7_gempack_parity(row):
    fixture = ROOT / "tests/fixtures/gtap7_gempack" / row.ref
    if not fixture.exists():
        pytest.skip(f"updated.har fixture missing: {fixture}")
    # measurement wiring mirrors test_gtap7_mcp_parity._solve_and_measure but reads
    # the reference via gempack_levels(); asserts match >= stage floor at TOL=1e-2
    # and code == 1. Implemented when the first real updated.har lands.
    pytest.skip("gempack measurement enabled once a real updated.har row exists")
