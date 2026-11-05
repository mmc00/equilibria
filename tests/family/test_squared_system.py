"""The J pattern gives the incidence Pyomo hands to the squaring (photo B oracle)."""

from __future__ import annotations

import time

import pytest

from tests.family._oracle import comp_stat_closure
from tests.family._prep import capture_dir
from tests.family.test_prep_recipe import _closure, _params
from tests.family.test_prep_square import _recipe

DATASET = "gtap7_3x3"

# Pyomo's ``identify_variables`` lists every variable a row mentions, also those whose
# derivative is exactly 0 (checked with Pyomo's derivatives at photo B's point and at a
# perturbed point). The J pattern is the true derivative structure, so it may lack
# columns B lists, but only in these families:
STRUCTURAL_ZERO_FAMILIES = {
    "eq_nd": "sigma=0: ratio**0",
    "eq_va": "sigma=0: ratio**0",
    "eq_xi": "sigma=0: ratio**0",
    "eq_xfteq": "etaf=0: pft enters multiplied by 0",
    "eq_gdpmp": "zero margin/tax coefficient",
    "eq_paa": "zero margin/tax coefficient",
    "eq_pmteq": "zero margin/tax coefficient",
    "eq_xtmg": "zero margin/tax coefficient",
    "eq_xweq": "zero margin/tax coefficient",
    "eq_ytax": "zero margin/tax coefficient",
}


@pytest.fixture(scope="module")
def inc():
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.family.squared_system import period_incidence
    from equilibria.templates.gtap.gtap_block_model import compose_block_state

    p = _params(DATASET)  # the run's params (gempack basis)
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


def _fam(r):
    return r.split("[")[0]


def test_pattern_is_the_true_derivative_subset_of_b(inc):
    from equilibria.family.squared_system import is_cross_period

    out, _, _ = inc
    seen, extra, lacking = set(), [], set()
    for _, b in _calls():
        free = _free(b)
        for r, cols in zip(b.con_names, b.incidence, strict=True):
            if r not in out:
                continue
            seen.add(_fam(r))
            if (set(out[r]) & free) - set(cols):
                extra.append(r)
            if (set(cols) & free) - set(out[r]):
                lacking.add(_fam(r))
    assert len(seen) > 60
    assert not any(is_cross_period(f) for f in seen)
    assert extra == []  # (a) no column B does not have
    assert lacking <= set(STRUCTURAL_ZERO_FAMILIES)  # (b)
    assert lacking == set(STRUCTURAL_ZERO_FAMILIES)  # the constant is not stale


def test_incidence_keeps_structural_zeros(inc):
    """A zero-valued variable with a nonzero derivative stays in the pattern."""
    out, _, _ = inc
    hits = []
    for _, b in _calls():
        free = _free(b)
        zero = {
            f"{c}[{','.join(k)}]" if k else c
            for c, v in b.vars.items()
            for k, x, fx in zip(v.keys, v.value, v.fixed, strict=True)
            if x == 0 and not fx
        }
        for r, cols in zip(b.con_names, b.incidence, strict=True):
            if r in out and _fam(r) not in STRUCTURAL_ZERO_FAMILIES:
                hits += [(r, c) for c in set(cols) & zero & free]
    assert hits, "no zero-valued free cell in a B incidence on 3x3"
    for r, c in hits:
        assert c in out[r], (r, c)


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
        # B's incidence only for the rows the pattern does not give (cross-period).
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
        inc_sets = {r: set(c) for r, c in merged.items()}
        hard = {(e, v) for e, v, h in b.forced if h}
        outside = sorted(
            (r, c) for r, c in hard if c not in inc_sets.get(r, ()) and r in inc_sets
        )
        print(f"call {a.n}: hard pairs {len(hard)}, outside the pattern {len(outside)}")
        assert compare(got, squared_from_photo(b), inc_sets, hard) == []
