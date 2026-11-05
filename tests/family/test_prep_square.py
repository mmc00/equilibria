"""Closure, squaring and matching on arrays give the system Pyomo hands to the solver.

Oracle: photo B, the squared system right before PyomoNLP is built (active rows, free
columns in paired order, every cell of the active period). The input is the recipe's
output on photo A (Task 3), so the two ports are checked end to end.
"""

from __future__ import annotations

import pytest

from tests.family._data import load_params
from tests.family._prep import capture_dir
from tests.family.test_prep_recipe import ROUTE, _closure, _params

DATASETS = (
    "gtap7_3x3",
    "gtap7_10x7",
    pytest.param("gtap7_15x10", marks=pytest.mark.slow),
)


@pytest.fixture(scope="module", params=DATASETS)
def case(request):
    from equilibria.family.prep_capture import pairs
    from equilibria.family.prep_recipe import sp_reference_arrays

    ds = request.param
    p = _params(ds)
    ref = sp_reference_arrays(p, _closure(), list(p.sets.r)[-1])
    calls = pairs(capture_dir(ds))
    return ds, p, ref, calls, _masks(p, calls)


def _masks(p, calls):
    """Built once per dataset and reused by every call (keyed by component)."""
    from equilibria.family.mp_arrays import MPArrays
    from equilibria.family.prep_square import ConditionalMasks

    return ConditionalMasks.from_params(p, MPArrays.from_photo(calls[0][0]))


def _recipe(a, p, ref):
    from equilibria.family.mp_arrays import MPArrays
    from equilibria.family.prep_recipe import run_recipe

    return run_recipe(
        MPArrays.from_photo(a),
        a.period,
        params=p,
        closure=_closure(),
        sp_ref=ref,
        mode="gtap",
        **ROUTE,
    )


def _square(mp, a, p, b, ref, masks):
    from equilibria.family.prep_square import square

    inc = {r: list(c) for r, c in zip(b.con_names, b.incidence, strict=True)}
    return square(
        mp,
        a.period,
        params=p,
        closure=_closure(),
        incidence=inc,
        exogenous=ref.exogenous,
        masks=masks,
    )


def _hard(b):
    return {(e, v) for e, v, h in b.forced if h}


@pytest.mark.parametrize("k", range(3))
def test_square_matches_pyomo(case, k):
    from equilibria.family.mp_arrays import compare, squared_from_photo

    _, p, ref, calls, masks = case
    if k >= len(calls):
        pytest.skip("fewer solve calls")
    a, b = calls[k]
    got = _square(_recipe(a, p, ref), a, p, b, ref, masks)
    inc = {r: set(c) for r, c in zip(b.con_names, b.incidence, strict=True)}
    assert compare(got, squared_from_photo(b), inc, _hard(b)) == []


def test_hard_pairs_kept(case):
    _, p, ref, calls, masks = case
    for a, b in calls:
        got = _square(_recipe(a, p, ref), a, p, b, ref, masks)
        pairs = set(zip(got.row_names, got.col_names, strict=True))
        assert _hard(b)
        assert _hard(b) <= pairs


def test_gams_pairs_equal_pyomo(case):
    """The (eq, var, hard) pairs handed to the matching are the ones Pyomo built."""
    from equilibria.family.prep_square import gams_pairs

    _, p, ref, calls, masks = case
    for a, b in calls:
        got = gams_pairs(_recipe(a, p, ref), params=p, mode="gtap")
        assert set(got) == set(b.forced)


def test_overdetermined_raises(case):
    """One free column fewer than rows: Pyomo would drop an equation by matching."""
    _, p, ref, calls, masks = case
    a, b = calls[-1]
    mp = _recipe(a, p, ref)
    sq = _square(mp, a, p, b, ref, masks)
    name = next(c for c in sq.col_names if c.startswith("xp["))
    comp, key = name[:-1].split("[")
    v = mp.vars[comp]
    v.fixed[v.pos(tuple(key.split(",")))] = True
    with pytest.raises(ValueError, match="over-determined"):
        _square(mp, a, p, b, ref, masks)


def test_gap_raises(case):
    """More free columns than rows: Pyomo would fix variables by matching."""
    _, p, ref, calls, masks = case
    a, b = calls[-1]
    mp = _recipe(a, p, ref)
    r = (
        mp.rows["eq_xp"]
        if "eq_xp" in mp.rows
        else next(r for r in mp.rows.values() if r.active.any())
    )
    r.active[int(r.active.nonzero()[0][0])] = False
    with pytest.raises(ValueError, match="gap"):
        _square(mp, a, p, b, ref, masks)


def test_missing_component_raises(case):
    _, p, ref, calls, masks = case
    a, b = calls[-1]
    mp = _recipe(a, p, ref)
    del mp.vars["pmt"]
    with pytest.raises(KeyError, match="pmt"):
        _square(mp, a, p, b, ref, masks)


def test_input_not_mutated(case):
    import numpy as np

    _, p, ref, calls, masks = case
    a, b = calls[-1]
    mp = _recipe(a, p, ref)
    before = {n: (v.fixed.copy(), v.lb.copy()) for n, v in mp.vars.items()}
    _square(mp, a, p, b, ref, masks)
    for n, (f, lb) in before.items():
        np.testing.assert_array_equal(mp.vars[n].fixed, f)
        np.testing.assert_array_equal(mp.vars[n].lb, lb)


def test_exogenous_is_required(case):
    """The registered instruments come from the model (``SPRef.exogenous``)."""
    from equilibria.family.prep_square import square

    _, p, ref, calls, masks = case
    a, b = calls[-1]
    inc = {r: list(c) for r, c in zip(b.con_names, b.incidence, strict=True)}
    with pytest.raises(TypeError):
        square(  # ty: ignore[missing-argument]
            _recipe(a, p, ref),
            a.period,
            params=p,
            closure=_closure(),
            incidence=inc,
            masks=masks,
        )


def test_masks_required(case):
    """The data masks are built once per run: ``masks`` has no default."""
    from equilibria.family.prep_square import square

    _, p, ref, calls, _ = case
    a, b = calls[-1]
    inc = {r: list(c) for r, c in zip(b.con_names, b.incidence, strict=True)}
    with pytest.raises(TypeError):
        square(  # ty: ignore[missing-argument]
            _recipe(a, p, ref),
            a.period,
            params=p,
            closure=_closure(),
            incidence=inc,
            exogenous=ref.exogenous,
        )


def test_masks_reject_other_keys(case):
    """Masks built for one key layout raise on a permuted or a different-size mp."""
    import copy

    _, _, _, calls, masks = case
    mp = _mp_of(calls)
    masks.get(mp, "xp")  # same keys: fine (equal list, other object too)
    assert masks.get(mp.copy(), "xp") is masks.outside["xp"]
    permuted = copy.copy(mp)
    permuted.vars = dict(mp.vars)
    v = copy.copy(mp.vars["xp"])
    v.keys = list(reversed(v.keys))
    permuted.vars["xp"] = v
    assert v.keys != masks._keys["xp"]
    with pytest.raises(ValueError, match="other keys"):
        masks.get(permuted, "xp")
    short = copy.copy(mp)
    short.vars = dict(mp.vars)
    w = copy.copy(mp.vars["xp"])
    w.keys = w.keys[:-1]
    short.vars["xp"] = w
    with pytest.raises(ValueError, match="other keys"):
        masks.get(short, "xp")


def _mp_of(calls):
    from equilibria.family.mp_arrays import MPArrays

    return MPArrays.from_photo(calls[0][0])
