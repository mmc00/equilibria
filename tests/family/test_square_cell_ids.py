"""``square`` on integer cell ids equals the string version it replaced.

``_legacy_square`` is the body of ``square`` before the change (names built, sorted
with ``list.sort``, adjacency and pairs by name dicts). Row order, column order, the
pairing and the cells must be ``==`` on every captured call, with the name incidence
and with ``ArrayIncidence`` on the J pattern.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.family._prep import capture_dir
from tests.family.test_prep_recipe import ROUTE, _closure
from tests.family.test_squared_system import _backends

DATASETS = ["gtap7_3x3", "gtap7_10x7"]


def _legacy_square(mp, period, *, params, closure, incidence, exogenous, masks):
    from equilibria.family import prep_square as ps
    from equilibria.family.mp_arrays import Squared
    from equilibria.solver._matching import match_rows

    mp = mp.copy()
    for name in ps._REQUIRED_VARS:
        mp.var(name)
    for name in ps._REQUIRED_ROWS:
        mp.row(name)
    ps._initialize_benchmark(mp)
    if closure is not None:
        ps._apply_closure(mp, closure, exogenous)
    ps._conditional_fixing(mp, masks)
    ps._check_gap(mp)
    ps._fix_sluggish_pft(mp, params)
    ps._deactivate_fixed_quantity_eqs(mp)
    ps._check_overdetermined(mp)
    ps._relative_floors(mp)
    rnames, cnames = [], []
    for name, row in mp.rows.items():
        rnames += ps._names(name, ps._active_keys(row))
    for name, v in mp.vars.items():
        cnames += ps._names(name, [v.keys[i] for i in np.flatnonzero(~v.fixed)])
    rnames.sort()
    cnames.sort()
    adjacency = ps._adjacency(rnames, cnames, incidence)
    r_index = {r: i for i, r in enumerate(rnames)}
    c_index = {c: i for i, c in enumerate(cnames)}
    pairs = [
        (r_index[e], c_index[v], h)
        for e, v, h in ps.gams_pairs(mp, params=params, mode="gtap")
        if e in r_index and v in c_index
    ]
    paired = match_rows(len(rnames), adjacency, pairs)
    return Squared(
        rnames, [cnames[j] for j in paired], ps._cells(mp, period)
    ), adjacency


@pytest.fixture(scope="module", params=DATASETS)
def route(request):
    from equilibria.family.mp_arrays import MPArrays
    from equilibria.family.prep_capture import pairs
    from equilibria.family.prep_recipe import run_recipe, sp_reference_arrays
    from equilibria.family.prep_square import ConditionalMasks

    ds = request.param
    p, backend, state, cross = _backends(ds)
    ref = sp_reference_arrays(p, _closure(), list(p.sets.r)[-1])
    calls = pairs(capture_dir(ds))
    masks = ConditionalMasks.from_params(p, MPArrays.from_photo(calls[0][0]))
    mps = [
        run_recipe(
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
        for a, _ in calls
    ]
    periods = [a.period for a, _ in calls]
    return p, backend, state, cross, ref, masks, mps, periods


def _kw(p, ref, masks) -> dict[str, Any]:
    return {
        "params": p,
        "closure": _closure(),
        "exogenous": ref.exogenous,
        "masks": masks,
    }


def test_three_calls_at_least(route):
    assert len(route[6]) >= 3


@pytest.mark.parametrize("k", range(3))
def test_square_equals_the_string_version(route, k):
    from equilibria.family.prep_square import square
    from equilibria.family.squared_system import ArrayIncidence, period_incidence

    p, backend, state, cross, ref, masks, mps, periods = route
    mp, period = mps[k], periods[k]
    inc = period_incidence(backend, state, period, cross)
    want, want_adj = _legacy_square(mp, period, incidence=inc, **_kw(p, ref, masks))
    for incidence in (inc, ArrayIncidence(backend, state, period, cross)):
        got = square(mp, period, incidence=incidence, **_kw(p, ref, masks))
        assert got.row_names == want.row_names
        assert got.col_names == want.col_names
        assert list(got.cells) == list(want.cells)
        assert got.cells == want.cells
        assert got == want


def test_adjacency_equals_the_name_one(route):
    """Same ``list[list[int]]`` goes into ``match_rows``."""
    from equilibria.family.cell_index import CellIndex
    from equilibria.family.squared_system import ArrayIncidence, period_incidence

    p, backend, state, cross, ref, masks, mps, periods = route
    mp, period = mps[0], periods[0]
    sq = _legacy_square(
        mp,
        period,
        incidence=period_incidence(backend, state, period, cross),
        **_kw(p, ref, masks),
    )[0]
    del sq
    # the active rows/columns after the closure: take them from the squared mp
    from equilibria.family.prep_square import _adjacency, square

    got_sq, mp_sq = square(
        mp,
        period,
        incidence=period_incidence(backend, state, period, cross),
        return_mp=True,
        **_kw(p, ref, masks),
    )
    index = CellIndex(mp_sq)
    rid = np.flatnonzero(np.concatenate([r.active for r in mp_sq.rows.values()]))
    cid = np.flatnonzero(~np.concatenate([v.fixed for v in mp_sq.vars.values()]))
    rid = rid[np.argsort(index.row_rank[rid], kind="stable")]
    cid = cid[np.argsort(index.col_rank[cid], kind="stable")]
    col_pos = np.full(len(index.col_names), -1, dtype=np.int64)
    col_pos[cid] = np.arange(len(cid))
    rn = [index.row_names[i] for i in rid]
    cn = [index.col_names[i] for i in cid]
    want = _adjacency(rn, cn, period_incidence(backend, state, period, cross))
    got = ArrayIncidence(backend, state, period, cross).adjacency(
        mp_sq, index, rid, col_pos
    )
    assert got == want


def test_cell_index_is_reused_across_copies(route):
    from equilibria.family.cell_index import CellIndex
    from equilibria.family.prep_square import square
    from equilibria.family.squared_system import ArrayIncidence

    p, backend, state, cross, ref, masks, mps, periods = route
    mp, period = mps[0], periods[0]
    index = CellIndex(mp)
    assert index.fits(mp) and index.fits(mp.copy())
    inc = ArrayIncidence(backend, state, period, cross)
    a = square(mp, period, incidence=inc, index=index, **_kw(p, ref, masks))
    b = square(mp, period, incidence=inc, index=index, **_kw(p, ref, masks))
    assert a == b
    assert not index.fits(mps[1])  # other key lists: the driver rebuilds


def test_cells_are_the_values_when_squared(route):
    """Lazy cells do not follow later writes to the returned arrays."""
    from equilibria.family.prep_square import square
    from equilibria.family.squared_system import ArrayIncidence

    p, backend, state, cross, ref, masks, mps, periods = route
    mp, period = mps[0], periods[0]
    inc = ArrayIncidence(backend, state, period, cross)
    sq, mp_sq = square(mp, period, incidence=inc, return_mp=True, **_kw(p, ref, masks))
    for v in mp_sq.vars.values():
        v.value[:] = -7.0
    want, _ = _legacy_square(
        mp, period, incidence=inc_names(route), **_kw(p, ref, masks)
    )
    assert sq.cells == want.cells


def inc_names(route):
    from equilibria.family.squared_system import period_incidence

    p, backend, state, cross, ref, masks, mps, periods = route
    return period_incidence(backend, state, periods[0], cross)
