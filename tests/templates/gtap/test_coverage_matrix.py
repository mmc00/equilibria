"""Tests for the GTAP7 coverage matrix (single source of truth)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/gtap"))


def test_matrix_schema_invariants():
    from coverage_matrix import CI_STATUSES, KINDS, ROWS

    assert ROWS, "matrix must not be empty"
    for r in ROWS:
        # ifsub is None iff kind == "gtap" (the .nl gate); SOLVE/nlp kinds carry 0/1
        assert (r.ifsub is None) == (r.kind == "gtap"), r
        assert r.kind in KINDS, r
        assert r.ci_status in CI_STATUSES, r
        assert r.phases, r
        if r.kind in ("nlp", "mcp"):
            # per-stage rows: row-level gap_min is None; the floor lives per stage
            # in stage_floors, and match is MEASURED by the test at run time (never
            # stored). mode distinguishes pure/altertax within the gate.
            assert r.gap_min is None, r
            assert r.mode in {"pure", "altertax"}, r
            assert r.stage_floors is not None, r
            floors = dict(r.stage_floors)
            assert set(floors) == set(r.phases), r
            for f in floors.values():
                assert 0.0 < f <= 100.0, r
            continue
        assert r.stage_floors is None and r.mode is None, r
        # gap_min invariants do NOT apply to blocked rows (never asserted).
        if r.ci_status != "blocked":
            # gap_min is None exactly for the .nl-only gtap7_* rows
            nl_only = r.kind == "gtap" and r.dataset.startswith("gtap7_")
            assert (r.gap_min is None) == nl_only, r
            # a non-None gap_min must be a sane floor (never an exact 100)
            if r.gap_min is not None:
                assert 0.0 < r.gap_min < 100.0, r


def test_matrix_helpers_partition():
    from coverage_matrix import (
        ROWS,
        altertax_rows,
        gtap_solve_rows,
        mcp_rows,
        nl_rows,
        nlp_rows,
    )

    assert (
        set(nl_rows())
        | set(altertax_rows())
        | set(gtap_solve_rows())
        | set(nlp_rows())
        | set(mcp_rows())
    ) == set(ROWS)
    assert all(r.kind == "gtap" for r in nl_rows())
    assert all(r.kind == "altertax" for r in altertax_rows())
    assert all(r.kind == "gtap_solve" for r in gtap_solve_rows())
    assert all(r.kind == "nlp" for r in nlp_rows())
    assert all(r.kind == "mcp" for r in mcp_rows())


def test_coverage_doc_in_sync():
    """The committed doc must equal render() — regenerate + commit on drift."""
    import gen_coverage_doc

    committed = gen_coverage_doc.DOC_PATH.read_text(encoding="utf-8")
    assert committed == gen_coverage_doc.render(), (
        "docs/site/guide/gtap7_coverage_matrix.md is stale — run "
        "`uv run python scripts/gtap/gen_coverage_doc.py` and commit."
    )


def test_new_axes_default_and_validate():
    from coverage_matrix import MODELS, REFERENCES, ROWS

    # all rows are gtap7 today; reference is gams (default) or gempack (F5 rows)
    assert all(r.model == "gtap7" for r in ROWS), "all rows must be model=gtap7 today"
    assert all(r.reference in REFERENCES for r in ROWS), (
        "reference must be in REFERENCES"
    )
    assert any(r.reference == "gams" for r in ROWS), "gams rows must exist"
    # the invariant sets exist and contain the axis values
    assert "gtap6" in MODELS and "gtap7" in MODELS
    assert "gams" in REFERENCES and "gempack" in REFERENCES
    # _validate rejects an out-of-domain model/reference
    import dataclasses

    import coverage_matrix as cm
    import pytest as _pt

    bad = dataclasses.replace(ROWS[0], model="gtap9")
    saved = cm.ROWS
    try:
        cm.ROWS = [*saved, bad]
        with _pt.raises(AssertionError):
            cm._validate()
    finally:
        cm.ROWS = saved


def test_rows_for_filters_by_model_and_reference():
    from coverage_matrix import mcp_rows, rows_for

    # mcp_rows() is gams + gempack; rows_for narrows to one reference
    gams_mcp = rows_for("gtap7", "gams", kind="mcp")
    gempack_mcp = rows_for("gtap7", "gempack", kind="mcp")
    assert set(gams_mcp) | set(gempack_mcp) == set(mcp_rows())
    assert all(r.reference == "gams" for r in gams_mcp)
    assert gempack_mcp and all(r.reference == "gempack" for r in gempack_mcp)
    # no gtap6 rows yet → empty
    assert rows_for("gtap6", "gams") == []
