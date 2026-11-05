"""The J pattern gives the incidence Pyomo hands to the squaring (photo B oracle)."""

from __future__ import annotations

import time

import pytest

from tests.family._data import load_params
from tests.family._oracle import comp_stat_closure
from tests.family._prep import capture_dir
from tests.family.test_prep_recipe import _closure, _params
from tests.family.test_prep_square import _recipe

DATASET = "gtap7_3x3"

# FINDING (not papered over): for these families photo B lists free columns that the
# family J pattern lacks. The backend records the single-period form of the equation;
# the multiperiod model's row differs (eq_nd/eq_va/eq_xi/eq_xfteq/eq_paa/... carry more
# variables) and the cross-period Fisher rows (eq_chisave, eq_pabs, eq_pfact,
# eq_pwfact, eq_rgdpmp) are left out of period_incidence (CROSS_PERIOD_FAMILIES).
KNOWN_GAPS = {
    "eq_gdpmp", "eq_nd", "eq_paa", "eq_pmteq", "eq_va", "eq_xfteq", "eq_xi",
    "eq_xtmg", "eq_xweq", "eq_ytax",
}  # fmt: skip
GAP = pytest.mark.xfail(
    strict=True, reason="J pattern lacks columns of B in KNOWN_GAPS (task-1 finding)"
)


@pytest.fixture(scope="module")
def inc():
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.family.squared_system import period_incidence
    from equilibria.templates.gtap.gtap_block_model import compose_block_state

    p = load_params(DATASET)
    model, state = compose_block_state(
        p, p.sets, comp_stat_closure(True), list(p.sets.r)[-1]
    )
    backend = FamilyBackend()
    backend.build(model)
    calls = _calls()
    t = time.perf_counter()
    out = period_incidence(backend, state, calls[0][1].period)
    return out, time.perf_counter() - t, state


def _calls():
    from equilibria.family.prep_capture import pairs

    return pairs(capture_dir(DATASET))


def _free(b):
    return {
        f"{c}[{','.join(k)}]" if k else c
        for c, v in b.vars.items()
        for k, fx in zip(v.keys, v.fixed, strict=True)
        if not fx
    }


def test_incidence_gaps_are_the_known_ones(inc):
    from equilibria.family.squared_system import is_cross_period

    out, _, _ = inc
    gaps, seen = set(), set()
    for _, b in _calls():
        free = _free(b)
        for r, cols in zip(b.con_names, b.incidence, strict=True):
            if r in out:
                seen.add(r.split("[")[0])
                if (set(cols) & free) - set(out[r]):
                    gaps.add(r.split("[")[0])
    assert len(seen) > 60
    assert gaps == KNOWN_GAPS
    assert not any(is_cross_period(f) for f in seen)
    # Everything else is exact: no extra column either.
    extra = {
        r.split("[")[0]
        for _, b in _calls()
        for r, cols in zip(b.con_names, b.incidence, strict=True)
        if r in out and (set(out[r]) & _free(b)) - set(cols)
    }
    assert extra == set()


@GAP
def test_incidence_matches_photo_b(inc):
    out, _, _ = inc
    bad = []
    for _, b in _calls():
        free = _free(b)
        for r, cols in zip(b.con_names, b.incidence, strict=True):
            if r not in out:
                continue
            if set(out[r]) & free != set(cols) & free:
                bad.append((r, sorted((set(out[r]) ^ set(cols)) & free)))
    assert bad == []


@GAP
def test_incidence_keeps_structural_zeros(inc):
    out, _, _ = inc
    zeros = 0
    for _, b in _calls():
        zero = {
            f"{c}[{','.join(k)}]" if k else c
            for c, v in b.vars.items()
            for k, x in zip(v.keys, v.value, strict=True)
            if x == 0
        }
        for r, cols in zip(b.con_names, b.incidence, strict=True):
            if r in out:
                for c in set(cols) & zero:
                    zeros += 1
                    assert c in out[r], (r, c)
    assert zeros > 0


@GAP
def test_square_with_pattern_incidence(inc):
    from equilibria.family.mp_arrays import compare, squared_from_photo
    from equilibria.family.prep_recipe import sp_reference_arrays
    from equilibria.family.prep_square import square

    out, _, _ = inc
    p = _params(DATASET)
    ref = sp_reference_arrays(p, _closure(), list(p.sets.r)[-1])
    calls = _calls()
    assert calls
    for a, b in calls:
        merged = {r: list(c) for r, c in zip(b.con_names, b.incidence, strict=True)}
        merged.update(out)
        got = square(
            _recipe(a, p, ref),
            a.period,
            params=p,
            closure=_closure(),
            incidence=merged,
            exogenous=ref.exogenous,
        )
        binc = {r: set(c) for r, c in zip(b.con_names, b.incidence, strict=True)}
        hard = {(e, v) for e, v, h in b.forced if h}
        assert compare(got, squared_from_photo(b), binc, hard) == []
