"""The J pattern gives the incidence Pyomo hands to the squaring (photo B oracle)."""

from __future__ import annotations

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


class _Incidence:
    """``period_incidence`` of the 3x3 backend, computed once per period."""

    def __init__(self, backend, state):
        self.backend, self.state, self._cache = backend, state, {}

    def __call__(self, period: str) -> dict[str, list[str]]:
        from equilibria.family.squared_system import period_incidence

        if period not in self._cache:
            self._cache[period] = period_incidence(self.backend, self.state, period)
        return self._cache[period]


@pytest.fixture(scope="module")
def inc():
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.templates.gtap.gtap_block_model import compose_block_state

    p = _params(DATASET)  # the run's params (gempack basis)
    model, state = compose_block_state(
        p, p.sets, comp_stat_closure(True), list(p.sets.r)[-1]
    )
    backend = FamilyBackend()
    backend.build(model)
    return _Incidence(backend, state)


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


def _assert_all_rows_found(out, b):
    """Every active non-cross-period row of B is in ``out`` (no silent skip)."""
    from equilibria.family.squared_system import is_cross_period

    want = [r for r in b.con_names if not is_cross_period(_fam(r))]
    found = [r for r in want if r in out]
    assert found, f"call {b.n} ({b.period}): no B row is in the incidence"
    assert len(found) == len(want), (
        f"call {b.n} ({b.period}): rows of B missing from the incidence: "
        f"{[r for r in want if r not in out][:5]}"
    )


def _assert_cross_period_names(inc, calls):
    """Cross-period families are absent from the incidence, and every name in
    CROSS_PERIOD_FAMILIES exists in the backend records or in B (no misspelling)."""
    from equilibria.family.squared_system import (
        CROSS_PERIOD_FAMILIES,
        is_cross_period,
    )

    recorded = set(inc.backend.records)
    in_b = {name for _, b in calls for name in b.rows}
    for name in CROSS_PERIOD_FAMILIES:
        assert name in recorded | in_b, f"{name}: not a family of the backend or of B"
    assert any(is_cross_period(f) for f in recorded), "no cross-period family recorded"
    for _, b in calls:
        assert not {_fam(r) for r in inc(b.period) if is_cross_period(_fam(r))}


def test_pattern_is_the_true_derivative_subset_of_b(inc):
    from equilibria.family.squared_system import is_cross_period

    seen, extra, lacking = set(), [], set()
    for _, b in _calls():
        out = inc(b.period)
        _assert_all_rows_found(out, b)
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
    _assert_cross_period_names(inc, _calls())
    assert extra == []  # (a) no column B does not have
    assert lacking <= set(STRUCTURAL_ZERO_FAMILIES)  # (b)
    assert lacking == set(STRUCTURAL_ZERO_FAMILIES)  # the constant is not stale


def test_incidence_keeps_structural_zeros(inc):
    """A zero-valued variable with a nonzero derivative stays in the pattern."""
    hits = []
    for _, b in _calls():
        out = inc(b.period)
        free = _free(b)
        zero = {
            f"{c}[{','.join(k)}]" if k else c
            for c, v in b.vars.items()
            for k, x, fx in zip(v.keys, v.value, v.fixed, strict=True)
            if x == 0 and not fx
        }
        for r, cols in zip(b.con_names, b.incidence, strict=True):
            if r in out and _fam(r) not in STRUCTURAL_ZERO_FAMILIES:
                hits += [(out, r, c) for c in set(cols) & zero & free]
    assert hits, "no zero-valued free cell in a B incidence on 3x3"
    for o, r, c in hits:
        assert c in o[r], (r, c)


def test_square_with_pattern_incidence(inc):
    from equilibria.family.mp_arrays import compare, squared_from_photo
    from equilibria.family.prep_recipe import sp_reference_arrays
    from equilibria.family.prep_square import square

    p = _params(DATASET)
    ref = sp_reference_arrays(p, _closure(), list(p.sets.r)[-1])
    calls = _calls()
    assert calls
    for a, b in calls:
        out = inc(b.period)
        _assert_all_rows_found(out, b)
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


# ---------------------------------------------------------------------------
# SquaredSystem: the squared system of one call on the backend
# ---------------------------------------------------------------------------

# Families whose rows lose J entries when a fixed-at-0 cell zeroes a term (3x3 calls).
FOLDED_BY_FIXED_ZERO = {"eq_ytax"}


@pytest.fixture(scope="module")
def systems(inc):
    """(call, SquaredSystem, incidence used, sq) for every 3x3 solve call."""
    from equilibria.family.mp_arrays import MPArrays
    from equilibria.family.prep_recipe import sp_reference_arrays
    from equilibria.family.prep_square import square
    from equilibria.family.squared_system import SquaredSystem

    p = _params(DATASET)
    ref = sp_reference_arrays(p, _closure(), list(p.sets.r)[-1])
    out = []
    for a, b in _calls():
        mp = _recipe(a, p, ref)
        merged = {r: list(c) for r, c in zip(b.con_names, b.incidence, strict=True)}
        merged.update(inc(b.period))
        sq = square(
            mp,
            a.period,
            params=p,
            closure=_closure(),
            incidence=merged,
            exogenous=ref.exogenous,
        )
        ss = SquaredSystem.build(inc.backend, inc.state, mp, a.period, sq)
        out.append((b, ss, merged, sq))
    assert out
    return out


def test_squared_system_shapes_3x3(inc, systems):
    from equilibria.family.squared_system import is_cross_period

    for b, ss, merged, sq in systems:
        n = len(sq.row_names)
        assert n == len(sq.col_names)
        assert ss.x0.shape == (n,)
        assert ss.F(ss.x0).shape == (n,)
        J = ss.J(ss.x0)
        assert J.shape == (n, n)
        assert ss.proxy_families
        assert sum(ss.proxy_families.values()) == len(ss.proxy_rows)
        assert all(is_cross_period(f) for f in ss.proxy_families)
        assert len(ss.proxy_rows) > 0
        # J pattern of the non-proxy rows is the incidence of Task 1 on sq's columns,
        # except where the recorder folds a term that touches a cell FIXED at 0 in
        # this call's layout (the all-free incidence keeps it): never a new column.
        col_pos = {c: k for k, c in enumerate(sq.col_names)}
        proxy, folded, off_diagonal = set(ss.proxy_rows), set(), []
        pat = inc(b.period)
        Jc = J.tocsr()
        for k, r in enumerate(sq.row_names):
            got = set(Jc.indices[Jc.indptr[k] : Jc.indptr[k + 1]])
            if r in proxy:
                assert got == {k}, r
                continue
            want = {col_pos[c] for c in pat[r] if c in col_pos}
            if k not in got:
                # The pairing may put a structural zero on the diagonal (a hard GAMS
                # pair outside the true pattern), never a pair the pattern has.
                assert k not in want or _fam(r) in FOLDED_BY_FIXED_ZERO, r
                off_diagonal.append(r)
            assert got <= want, r
            if got != want:
                folded.add(_fam(r))
        assert folded <= FOLDED_BY_FIXED_ZERO
        print(f"call {b.n}: pairs off the J pattern {len(off_diagonal)}")


def test_jacobian_matches_finite_difference(systems):
    import numpy as np

    rng = np.random.default_rng(1)
    eps = 1e-7
    for _, ss, _, _ in systems:
        d = rng.standard_normal(ss.x0.size)
        Jd = ss.J(ss.x0) @ d
        fd = (ss.F(ss.x0 + eps * d) - ss.F(ss.x0 - eps * d)) / (2 * eps)
        assert np.abs(Jd - fd).max() <= 1e-6 * max(1.0, np.abs(Jd).max())


def test_proxy_rows_residual(systems):
    import numpy as np

    for _, ss, _, sq in systems:
        x = ss.x0 + 0.5
        F = ss.F(x)
        for r in ss.proxy_rows:
            k = sq.row_names.index(r)
            assert F[k] == pytest.approx(0.5)
        assert np.isfinite(F).all()


def test_unknown_family_row_raises(inc, systems):
    """An active row of no family and not cross-period is not silently proxied."""
    import copy

    from equilibria.family.squared_system import SquaredSystem

    _, _, _, sq = systems[0]
    a = next(a for a, b in _calls() if b.period == "check")
    bad = copy.deepcopy(sq)
    bad.row_names.append("eq_unknown[x,check]")
    bad.col_names.append("pf[unknown,check]")
    bad.cells["pf[unknown,check]"] = (1.0, False, 0.0, 1.0)
    from equilibria.family.mp_arrays import MPArrays

    with pytest.raises(ValueError, match="eq_unknown"):
        SquaredSystem.build(
            inc.backend, inc.state, MPArrays.from_photo(a), "check", bad
        )


def test_variable_missing_from_mp_raises(inc, systems):
    """Only the SP-only allow-list may be absent from mp (it takes state values)."""
    from equilibria.family.mp_arrays import MPArrays
    from equilibria.family.squared_system import SP_ONLY_VARS, SquaredSystem

    a, _ = _calls()[0]
    _, ss, _, sq = systems[0]
    # sq with the family variable's cells removed, so only mp could supply them.
    drop = "pf"
    assert drop not in SP_ONLY_VARS
    sq2 = type(sq)(
        sq.row_names,
        sq.col_names,
        {k: v for k, v in sq.cells.items() if k.partition("[")[0] != drop},
    )
    mp = MPArrays.from_photo(a)
    del mp.vars[drop]
    with pytest.raises(KeyError, match=drop):
        SquaredSystem.build(inc.backend, inc.state, mp, a.period, sq2)
