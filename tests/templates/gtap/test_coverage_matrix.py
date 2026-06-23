"""Tests for the GTAP7 coverage matrix (single source of truth)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/gtap"))


def test_matrix_schema_invariants():
    from coverage_matrix import ROWS, CI_STATUSES
    assert ROWS, "matrix must not be empty"
    for r in ROWS:
        # ifsub is None iff kind == "gtap"
        assert (r.ifsub is None) == (r.kind == "gtap"), r
        assert r.kind in ("gtap", "altertax"), r
        assert r.ci_status in CI_STATUSES, r
        assert r.phases, r
        # gap_min invariants do NOT apply to blocked rows (never asserted).
        if r.ci_status != "blocked":
            # gap_min is None exactly for the .nl-only gtap7_* single-period rows
            nl_only = r.kind == "gtap" and r.dataset.startswith("gtap7_")
            assert (r.gap_min is None) == nl_only, r
            # a non-None gap_min must be a sane floor (never an exact 100)
            if r.gap_min is not None:
                assert 0.0 < r.gap_min < 100.0, r


def test_matrix_helpers_partition():
    from coverage_matrix import ROWS, nl_rows, altertax_rows
    assert set(nl_rows()) | set(altertax_rows()) == set(ROWS)
    assert all(r.kind == "gtap" for r in nl_rows())
    assert all(r.kind == "altertax" for r in altertax_rows())


def test_coverage_doc_in_sync():
    """The committed doc must equal render() — regenerate + commit on drift."""
    import gen_coverage_doc
    committed = gen_coverage_doc.DOC_PATH.read_text(encoding="utf-8")
    assert committed == gen_coverage_doc.render(), (
        "docs/gtap7_coverage_matrix.md is stale — run "
        "`uv run python scripts/gtap/gen_coverage_doc.py` and commit."
    )
