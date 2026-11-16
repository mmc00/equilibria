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


def test_gempack_doc_in_sync():
    """The committed against-GEMPACK doc must equal render() — regen + commit on drift."""
    import gen_gempack_doc

    committed = gen_gempack_doc.DOC_PATH.read_text(encoding="utf-8")
    assert committed == gen_gempack_doc.render(), (
        "docs/site/guide/gtap7_coverage_matrix_gempack.md is stale — run "
        "`uv run python scripts/gtap/gen_gempack_doc.py` and commit."
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


def test_no_floor_below_95():
    """No row may ask for less than 95% match (Marlon, 2026-11-14): every stage floor
    and every gap_min. Lowering one below needs his OK and this test changed with it."""
    from coverage_matrix import ROWS

    low = []
    for r in ROWS:
        floors = dict(r.stage_floors or ())
        if r.gap_min is not None:
            floors["gap_min"] = r.gap_min
        low += [
            (r.dataset, r.kind, r.ifsub, k, f) for k, f in floors.items() if f < 95.0
        ]
    assert low == []


def _param_cases(fn):
    """``[(values, marks)]`` of ``fn``'s parametrize mark, plain tuples or params."""
    (mark,) = [m for m in fn.pytestmark if m.name == "parametrize"]
    out = []
    for v in mark.args[1]:
        if hasattr(v, "values"):
            out.append((tuple(v.values), [m.name for m in v.marks]))
        else:
            out.append((tuple(v) if isinstance(v, tuple) else (v,), []))
    return out


def test_every_pyomo_solve_case_has_a_family_twin():
    """A Pyomo solve case is marked ``pyomo_solve`` (it leaves PR CI) exactly when a
    non-xfail family case covers its variant and reference (Marlon, 2026-11-14: one
    the family does not cover yet keeps running in PR CI)."""
    import family_twins as _twins
    from tests.family._parity import gams_cases, gempack_cases
    from tests.templates.gtap import test_gtap7_gempack_parity as gem
    from tests.templates.gtap import test_gtap7_mcp_parity as mcp
    from tests.templates.gtap import test_gtap7_nlp_parity as nlp
    from tests.templates.gtap import test_ifsub_equivalence as eqv

    family = {
        (c.dataset, c.mode, c.ifsub, "gams") for c in gams_cases() if c.mode == "pure"
    } | {(c.dataset, c.mode, c.ifsub, "gempack") for c in gempack_cases()}
    assert _twins.twins() == family

    def key_mcp(v):
        return (v[0], v[2], v[1], "gams")

    def key_gem(v):
        return (v[0].dataset, v[0].mode, v[0].ifsub, "gempack")

    wrong = []
    for fn, key in (
        (mcp.test_gtap7_mcp_parity, key_mcp),
        (nlp.test_gtap7_nlp_parity, key_mcp),
        (gem.test_gtap7_gempack_parity, key_gem),
    ):
        for values, marks in _param_cases(fn):
            k = key(values)
            if ("pyomo_solve" in marks) != (k in family):
                wrong.append((fn.__name__, k, "pyomo_solve" in marks))
    for values, marks in _param_cases(eqv.test_ifsub_primary_block_consistent):
        d = values[0]
        pair = all((d, "pure", i, "gams") in family for i in (0, 1))
        if ("pyomo_solve" in marks) != pair:
            wrong.append(("test_ifsub_primary_block_consistent", d, not pair))
    for values, marks in _param_cases(mcp.test_basis_reproduces_its_gams_variant):
        if "pyomo_solve" in marks:  # other bases: the family runs the default one
            wrong.append(("test_basis_reproduces_its_gams_variant", values, True))
    assert wrong == [], (
        f"(test, case, marked) whose mark disagrees with the twins: {wrong}"
    )


def test_pyomo_gates_import_as_scripts_do(tmp_path):
    """gen_nlp_matrix_page.py imports the gate modules by bare name, with only
    scripts/gtap and tests/templates/gtap on sys.path (no repo root): the twin marks
    must not need the ``tests`` package."""
    import subprocess

    root = Path(__file__).resolve().parents[3]
    code = (
        "import sys;"
        f"sys.path[:0] = [{str(root / 'tests/templates/gtap')!r}, "
        f"{str(root / 'scripts/gtap')!r}, {str(root / 'src')!r}];"
        "import test_gtap7_mcp_parity, test_gtap7_nlp_parity, "
        "test_gtap7_gempack_parity, test_ifsub_equivalence"
    )
    p = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1"},
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr[-2000:]
