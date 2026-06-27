import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from cascade_classify import classify


def _p(status, kind=None, headline="h"):
    meta = {"error_kind": kind} if kind else {}
    return {"status": status, "headline": headline, "meta": meta, "violations": []}


def test_clean_continues():
    r = classify("nl_compare", _p("clean"), 0)
    assert r.action == "continue"


def test_dirty_explains_and_stops():
    r = classify("drift_test", _p("dirty"), 1)
    assert r.action == "explain_stop"


def test_nl_vacuity_is_visible_continue_not_clean():
    r = classify("nl_compare", _p("error", "no_common_constraints"), 2)
    assert r.action == "vacuous_continue"


def test_no_convergence_stops_the_period():
    r = classify("probe_seed", _p("error", "no_convergence"), 2)
    assert r.action == "upstream_stop"


def test_exception_records_and_continues():
    r = classify("calibration", _p("error", "exception"), 2)
    assert r.action == "tool_broken_continue"


def test_unknown_error_kind_is_blocking_never_clean():
    r = classify("validate_reference", _p("error", "some_future_kind"), 2)
    assert r.action == "blocking_stop"


def test_gdx_not_found_is_blocking():
    r = classify("calibration", _p("error", "gdx_not_found"), 2)
    assert r.action == "blocking_stop"
