"""Family ``fscale`` (leaves of the recorded rows, before zero folding) against Pyomo's."""

from __future__ import annotations

import numpy as np
import pytest

from equilibria.family.fscale import pyomo_fscale_from_photo, record_fscale
from equilibria.family.recorder import Node, Residual
from tests.family._prep import capture_dir
from tests.family.test_squared_system import (  # noqa: F401  (fixtures)
    DATASET,
    STRUCTURAL_ZERO_FAMILIES,
    _calls,
    cross,
    inc,
    real_systems,
)


def _leaf(var, flat, level):
    return Node("v", (var, flat, level, False))


def test_record_fscale_counts_free_leaves_of_both_sides_before_folding():
    # row 0: x[0] (free, 5) * y[1] (free, 7) == z[0] (fixed, 100)
    # row 1: x[1] (free, -9) * y[0] (fixed at 0, so compile_family folds it away) == 0
    x0, y1, z0 = _leaf("x", 0, 5.0), _leaf("y", 1, 7.0), _leaf("z", 0, 100.0)
    x1, y0 = _leaf("x", 1, -9.0), _leaf("y", 0, 0.0)
    records = {"eq": [((), Residual(x0 * y1, z0)), (("b",), Residual(x1 * y0, 0.0))]}
    values = {
        "x": np.array([3.0, -9.0]),
        "y": np.array([0.0, 7.0]),
        "z": np.array([100.0]),
    }
    free = {
        "x": np.array([True, True]),
        "y": np.array([False, True]),
        "z": np.array([False]),
    }
    got = record_fscale(records, values, free)
    # row 0: max(1, 3, 7); z is fixed. row 1: the free leaf x[1]; the folded y[0] is
    # fixed anyway and a NaN level is skipped
    assert got == {"eq": 7.0, "eq[b]": 9.0}
    values["x"][1] = np.nan
    assert record_fscale(records, values, free)["eq[b]"] == 1.0
    assert record_fscale(records, values, free, keep=lambda n: n == "eq") == {"eq": 7.0}


def test_pyomo_fscale_from_photo_reads_incidence_flags_and_levels():
    from equilibria.family.prep_capture import Photo, VarCells

    def cells(keys, val, fixed):
        n = len(keys)
        return VarCells(
            keys, np.array(val), np.array(fixed, bool), np.zeros(n), np.full(n, np.inf)
        )

    b = Photo(
        "B",
        1,
        "t",
        {"x": cells([("a", "t")], [4.0], [False])},
        {},
        con_names=["e1", "e2"],
        incidence=[["x[a,t]", "w[o]"], ["w[o]"]],
    )
    a2 = Photo("A2", 1, "t", {"w": cells([("o",)], [50.0], [True])}, {})
    # w[o] belongs to another period and is fixed there: it does not count
    assert pyomo_fscale_from_photo(b, a2) == {"e1": 4.0, "e2": 1.0}
    a2.vars["w"].fixed[0] = False
    assert pyomo_fscale_from_photo(b, a2) == {"e1": 50.0, "e2": 50.0}


@pytest.mark.slow
def test_fscale_matches_pyomo_3x3_all_calls(real_systems):  # noqa: F811
    from equilibria.family.prep_capture import load_a2

    d = capture_dir(DATASET)
    report, bad_rows = [], []
    for (a, _), (b, ss, _, sq) in zip(_calls(), real_systems, strict=True):
        got = ss.fscale()
        want_by_row = pyomo_fscale_from_photo(b, load_a2(d, a))
        want = np.array([want_by_row[r] for r in sq.row_names])
        assert got.shape == want.shape
        bad = np.flatnonzero(got.view(np.int64) != want.view(np.int64))
        report.append((b.n, b.period, len(want), len(bad)))
        bad_rows += [(b.n, sq.row_names[i], got[i], want[i]) for i in bad]
    print("fscale (call, period, rows, differ):", report)
    # Every difference is a structural zero the recorder drops while recording
    # (``ratio**0``, ``0 * x``: the leaf is not in the recorded tree at all, while
    # Pyomo's incidence still lists it): same families as the J-pattern gap of
    # test_squared_system. The family scale is then smaller, i.e. the relative test is
    # stricter there, never looser (spec answer Q4).
    for n, row, got, want in bad_rows:
        assert row.partition("[")[0] in STRUCTURAL_ZERO_FAMILIES, (n, row, got, want)
        assert got < want, (n, row, got, want)
    assert len(bad_rows) <= 9, len(bad_rows)  # 9 of 3 x 894 rows today
