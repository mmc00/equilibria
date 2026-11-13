"""Multiperiod state on arrays, and the squared-system comparison."""

import copy

import numpy as np
import pytest

from tests.family._prep import capture_dir


@pytest.fixture(scope="module")
def photos():
    from equilibria.family.prep_capture import pairs

    return pairs(capture_dir())


def test_photo_roundtrip(photos):
    from equilibria.family.mp_arrays import MPArrays

    a, _ = photos[0]
    mp = MPArrays.from_photo(a)
    for name, cells in a.vars.items():
        v = mp.vars[name]
        np.testing.assert_array_equal(v.value, cells.value)
        np.testing.assert_array_equal(v.fixed, cells.fixed)
        np.testing.assert_array_equal(v.lb, cells.lb)
        np.testing.assert_array_equal(v.ub, cells.ub)
        k = cells.keys[0]
        assert mp.name(name, 0) == (f"{name}[{','.join(k)}]" if k else name)
        assert mp.cells(name, k[-1]).size > 0
    for name, rows in a.rows.items():
        np.testing.assert_array_equal(mp.rows[name].active, rows.active)


def _inputs(b):
    from equilibria.family.mp_arrays import squared_from_photo

    s = squared_from_photo(b)
    inc = {r: set(c) for r, c in zip(b.con_names, b.incidence, strict=True)}
    return s, inc, {(e, v) for e, v, hard in b.forced if hard}


def test_squared_from_photo_compares_equal(photos):
    from equilibria.family.mp_arrays import compare

    _, b = photos[0]
    s, inc, forced = _inputs(b)
    assert compare(s, copy.deepcopy(s), inc, forced) == []


def test_compare_detects(photos):
    from equilibria.family.mp_arrays import compare

    _, b = photos[0]
    ref, inc, forced = _inputs(b)

    lb = copy.deepcopy(ref)
    cell = next(c for c in lb.col_names if np.isfinite(lb.cells[c][2]))
    v, f, lo, up = lb.cells[cell]
    lb.cells[cell] = (v, f, lo + 0.5, up)
    msgs = compare(lb, ref, inc, forced)
    assert len(msgs) == 1 and "lb" in msgs[0] and cell in msgs[0]

    rows = copy.deepcopy(ref)
    del rows.row_names[-1], rows.col_names[-1]
    msgs = compare(rows, ref, inc, forced)
    assert any("rows" in m for m in msgs)

    bad = copy.deepcopy(ref)
    k = next(
        i
        for i, r in enumerate(bad.row_names)
        if (r, bad.col_names[i]) not in forced
        and any(c not in inc[r] for c in bad.col_names)
    )
    j = next(
        i
        for i, c in enumerate(bad.col_names)
        if c not in inc[bad.row_names[k]] and (bad.row_names[i], c) not in forced
    )
    bad.col_names[k], bad.col_names[j] = bad.col_names[j], bad.col_names[k]
    msgs = compare(bad, ref, inc, forced)
    assert any("pair" in m for m in msgs)


def test_compare_accepts_other_valid_matching(photos):
    """Pyomo assigns leftover columns (outside the incidence) to the rows the locked
    hard pairs leave unmatched; any other leftover assignment is as valid."""
    from equilibria.family.mp_arrays import compare

    _, b = photos[0]
    ref, inc, hard = _inputs(b)
    alt = copy.deepcopy(ref)
    rows, cols = alt.row_names, alt.col_names
    left = [
        k
        for k, (r, c) in enumerate(zip(rows, cols, strict=True))
        if c not in inc[r] and (r, c) not in hard
    ]
    assert len(left) >= 2
    i, j = left[:2]
    cols[i], cols[j] = cols[j], cols[i]
    assert compare(alt, ref, inc, hard) == []

    # Giving a hard pair's column away is not.
    k = next(k for k, p in enumerate(zip(rows, cols, strict=True)) if p in hard)
    cols[k], cols[i] = cols[i], cols[k]
    msgs = compare(alt, ref, inc, hard)
    assert any("hard pairs not kept" in m for m in msgs)
