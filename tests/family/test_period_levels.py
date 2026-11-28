"""The family-vs-Pyomo cell comparison sees a cell that either side drops."""

from tests.family._period_levels import mismatches


def test_a_cell_missing_from_the_run_is_a_mismatch():
    want = {("x", ("USA",), "shock"): 1.0, ("y", ("USA",), "shock"): 2.0}
    got = {("x", ("USA",), "shock"): 1.0}
    assert mismatches(got, want, 1e-9) == ["('y', ('USA',), 'shock'): missing"]


def test_a_cell_the_reference_lacks_is_a_mismatch():
    want = {("x", ("USA",), "shock"): 1.0}
    got = {("x", ("USA",), "shock"): 1.0, ("y", ("USA",), "shock"): 2.0}
    assert mismatches(got, want, 1e-9) == ["('y', ('USA',), 'shock'): extra"]


def test_equal_cells_within_the_bound_match():
    want = {("x", ("USA",), "shock"): 1.0}
    assert mismatches({("x", ("USA",), "shock"): 1.0 + 1e-12}, want, 1e-9) == []
