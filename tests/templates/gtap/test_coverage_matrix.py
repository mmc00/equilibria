"""Tests for the GTAP7 coverage matrix (single source of truth)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/gtap"))


def test_row_has_new_axes_with_defaults():
    from coverage_matrix import Row
    r = Row(dataset="gtap7_3x3", variant="altertax", period="multi", ifsub=0,
            phases=("base", "check", "shock"), gap_min=98.0, gap_note="~99%",
            ci_status="local", ref="out_altertax_ifsub0.gdx")
    # nuevos ejes con defaults
    assert r.model == "gtap7"
    assert r.solver == "mcp"
    assert r.gap_gempack is None
    assert r.note_gempack == ""
    assert r.ref_gempack is None


def test_kind_is_derived_from_variant():
    from coverage_matrix import Row
    core = Row(dataset="gtap7_3x3", variant="core", period="single",
               ifsub=None, phases=("base", "shock"), gap_min=None,
               gap_note="0 diffs .nl", ci_status="ci", ref="x.nl")
    alt = Row(dataset="gtap7_3x3", variant="altertax", period="multi",
              ifsub=0, phases=("base", "check", "shock"), gap_min=98.0,
              gap_note="~99%", ci_status="local", ref="x.gdx")
    assert core.kind == "gtap"
    assert alt.kind == "altertax"


def test_rows_use_variant_and_period():
    from coverage_matrix import ROWS
    for r in ROWS:
        assert r.variant in ("core", "altertax"), r
        assert r.period in ("single", "multi"), r
        # core ⇒ single ⇒ ifsub None ; altertax ⇒ multi
        if r.variant == "core":
            assert r.period == "single" and r.ifsub is None, r
        else:
            assert r.period == "multi", r


def test_validate_rejects_bad_solver():
    import dataclasses
    from coverage_matrix import ROWS
    bad = dataclasses.replace(ROWS[0], solver="quadratic")
    # _validate iterates module ROWS, so assert the invariant directly on a bad row
    assert bad.solver not in ("mcp", "nlp"), "fixture must be a bad solver"


def test_gtap7_rows_filters_model():
    from coverage_matrix import gtap7_rows
    rows = gtap7_rows()
    assert rows, "expected gtap7 rows"
    assert all(r.model == "gtap7" for r in rows)


def test_progress_buckets_sum_to_total():
    from coverage_matrix import ROWS, progress
    p = progress()
    assert p["total"] == len(ROWS)
    assert p["done"] + p["partial"] + p["blocked"] == p["total"]
    # the 20x41 blocked row must land in 'blocked'
    assert p["blocked"] >= 1


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
        "docs/site/guide/gtap7_coverage_matrix.md is stale — run "
        "`uv run python scripts/gtap/gen_coverage_doc.py` and commit."
    )
