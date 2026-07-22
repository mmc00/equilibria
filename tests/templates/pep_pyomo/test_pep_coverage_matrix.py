"""Tests for the PEP coverage matrix (single source of truth) — mirrors the GTAP
coverage-matrix test. Schema invariants + the committed doc stays in sync with render()."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/pep"))


def test_pep_matrix_schema_invariants():
    from pep_coverage_matrix import CI_STATUSES, FORMS, KINDS, ROWS

    assert ROWS, "matrix must not be empty"
    for r in ROWS:
        assert r.kind in KINDS, r
        assert r.form in FORMS, r
        assert r.scenario in {"base", "sim1"}, r
        assert r.ci_status in CI_STATUSES, r
        assert r.cells > 0, r
        assert r.match_note and r.gate and r.ref, r


def test_pep_matrix_helpers_partition():
    from pep_coverage_matrix import ROWS, mcp_rows, mirror_rows, nlp_rows, variant_rows

    assert (
        set(nlp_rows()) | set(mcp_rows()) | set(mirror_rows()) | set(variant_rows())
    ) == set(ROWS)
    assert all(r.kind == "nlp" for r in nlp_rows())
    assert all(r.kind == "mcp" for r in mcp_rows())
    assert all(r.kind == "mirror" for r in mirror_rows())
    assert all(r.kind == "variant" for r in variant_rows())


def test_pep_coverage_doc_in_sync():
    """The committed doc must equal render() — regenerate + commit on drift."""
    import gen_pep_coverage_doc

    committed = gen_pep_coverage_doc.DOC_PATH.read_text(encoding="utf-8")
    assert committed == gen_pep_coverage_doc.render(), (
        "docs/site/guide/pep_coverage_matrix.md is stale — run "
        "`uv run python scripts/pep/gen_pep_coverage_doc.py` and commit."
    )
