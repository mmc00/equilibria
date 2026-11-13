"""The row-column pairing of ``square`` kept on disk next to the records.

``ArrayIncidence(store=dir)`` writes the pairing of each (period, rows, columns,
forced pairs) once; a new run with the same store reads it instead of running
``match_rows``. The squared system must be ``==`` to the one paired in memory.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.family._prep import capture_dir
from tests.family.test_prep_recipe import ROUTE, _closure
from tests.family.test_squared_system import _backends

DS = "gtap7_3x3"


@pytest.fixture(scope="module")
def route():
    from equilibria.family.mp_arrays import MPArrays
    from equilibria.family.prep_capture import pairs
    from equilibria.family.prep_recipe import run_recipe, sp_reference_arrays
    from equilibria.family.prep_square import ConditionalMasks

    p, backend, state, cross = _backends(DS)
    ref = sp_reference_arrays(p, _closure(), list(p.sets.r)[-1])
    a, _ = pairs(capture_dir(DS))[0]
    masks = ConditionalMasks.from_params(p, MPArrays.from_photo(a))
    mp = run_recipe(
        MPArrays.from_photo(a),
        a.period,
        params=p,
        closure=_closure(),
        sp_ref=ref,
        mode="gtap",
        cross=cross,
        backend=backend,
        **ROUTE,
    )
    return p, backend, state, cross, ref, masks, mp, a.period


def _square(route, store=None):
    from equilibria.family.prep_square import square
    from equilibria.family.squared_system import ArrayIncidence

    p, backend, state, cross, ref, masks, mp, period = route
    inc = ArrayIncidence(backend, state, period, cross, store=store)
    return square(
        mp,
        period,
        params=p,
        closure=_closure(),
        incidence=inc,
        exogenous=ref.exogenous,
        masks=masks,
    )


def _same(a, b):
    assert a.row_names == b.row_names
    assert a.col_names == b.col_names
    assert a.cells == b.cells


def test_first_run_writes_one_pairing_file(route, tmp_path):
    got = _square(route, store=tmp_path)
    _same(got, _square(route))
    assert len(list(tmp_path.glob("pairing-*.npy"))) == 1


def test_second_run_reads_it_without_matching(route, tmp_path, monkeypatch):
    from equilibria.family import prep_square

    want = _square(route, store=tmp_path)

    def no_matching(*a, **k):
        raise AssertionError("match_rows ran although the pairing was on disk")

    monkeypatch.setattr(prep_square, "match_rows", no_matching)
    _same(_square(route, store=tmp_path), want)


def test_a_bad_file_is_recomputed(route, tmp_path):
    want = _square(route, store=tmp_path)
    (f,) = tmp_path.glob("pairing-*.npy")
    np.save(f, np.zeros(3, dtype=np.int64))  # wrong length, not a permutation
    _same(_square(route, store=tmp_path), want)
    assert len(np.load(f)) == len(want.row_names)  # rewritten


def test_no_store_writes_nothing(route, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _square(route)
    assert not list(tmp_path.rglob("pairing-*"))
