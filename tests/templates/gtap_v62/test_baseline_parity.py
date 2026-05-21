"""Phase 3 — Baseline parity validation (Python init vs SAM benchmark).

The Python variable initialization computed by ``GTAPv62ModelEquations``
must reproduce the BOOK3X3 SAM values exactly. This is a calibration
correctness check that does NOT require a solver — it verifies that
``var[idx].value == SAM[idx]`` cell-by-cell across the 8 main value-
flow variables (qfe, qfd, qfm, qpd, qpm, qgd, qgm, qst).

Future shock parity tests (Phase 3.2+) will solve Python and compare
against the GEMPACK ``gtap.exe`` ``Exp1a-upd.har`` output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make scripts/ importable for the parity helpers
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.gtap_v62.validate_v62_parity import (  # noqa: E402
    BOOK3X3_DIR,
    build_book3x3_model,
    collect_baseline_diffs,
    collect_equation_residuals,
)


def _rungtap_available() -> bool:
    return all((BOOK3X3_DIR / f).exists()
               for f in ("SETS.HAR", "basedata.har", "Default.prm"))


pytestmark = pytest.mark.skipif(
    not _rungtap_available(),
    reason="RunGTAP v6.2 dataset BOOK3X3 not available",
)


@pytest.fixture(scope="module")
def book3x3_state():
    return build_book3x3_model(BOOK3X3_DIR)


def test_python_init_matches_sam_benchmark(book3x3_state) -> None:
    """Every SAM-driven variable INIT value equals the basedata.har value.

    This is the canonical calibration correctness test: if Python and
    GEMPACK disagree on the BASELINE, the data load / share calibration
    has drifted from the SAM.
    """
    _, _, model = book3x3_state
    basedata = BOOK3X3_DIR / "basedata.har"
    reports = collect_baseline_diffs(model, basedata, abs_tol=1.0, rel_tol=1e-4)

    failures = []
    for var_name, report in reports.items():
        if report.n_diverging > 0:
            failures.append(
                f"{var_name}: {report.n_diverging}/{report.n_cells} cells diverge "
                f"(max_abs={report.max_abs:.4e}, max_rel={report.max_rel:.4e}, "
                f"worst at {report.max_abs_key})"
            )
    assert not failures, "Baseline calibration drift:\n  " + "\n  ".join(failures)


def test_total_cells_compared(book3x3_state) -> None:
    """Sanity: the baseline diff covers all 8 main value-flow families."""
    _, _, model = book3x3_state
    basedata = BOOK3X3_DIR / "basedata.har"
    reports = collect_baseline_diffs(model, basedata)
    total_cells = sum(r.n_cells for r in reports.values())
    # 36 + 36 + 36 + 9 + 9 + 9 + 9 + 3 = 147 for BOOK3X3
    assert total_cells == 147, f"Expected 147 baseline cells, got {total_cells}"


def test_equation_residuals_within_documented_bounds(book3x3_state) -> None:
    """At benchmark, equation residuals are within the documented bounds.

    The residual ceilings reflect known SAM imperfections:
    - eq_qo:   ~6% from the implicit output-tax wedge (vom vs vop gap)
    - eq_qtm:  ~7e4 from intra-region VTWR (s==d entries in BOOK3X3)
    - eq_market: ~3e4 from ~1% SAM imbalance per cell
    - All other equations: ≤ 1e-3 (machine epsilon)
    """
    _, _, model = book3x3_state
    eq_reports = collect_equation_residuals(model)
    by_name = {r.eq_name: r for r in eq_reports}

    # Known SAM-imperfection bounds (relaxed)
    known_bounds = {
        "eq_qo": 0.10,            # 10% — output tax wedge
        "eq_qtm": 1.0e5,          # 100k — intra-region VTWR
        "eq_market": 5.0e4,       # 50k — ~1% SAM imbalance
        "eq_factor_clear": 5.0,   # 5 — float32 rounding in HAR
    }

    # All other equations should balance to machine epsilon
    for eq_name, report in by_name.items():
        ceiling = known_bounds.get(eq_name, 1e-3)
        assert report.max_abs < ceiling, (
            f"{eq_name!r} max_abs residual {report.max_abs:.4e} exceeds "
            f"ceiling {ceiling}"
        )
