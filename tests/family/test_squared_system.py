"""The J pattern gives the incidence Pyomo hands to the squaring (photo B oracle)."""

from __future__ import annotations

import pytest

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


def _script(name: str):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "family" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _route_closure():
    """The capture route's closure (``_solve_shock``), as newton_cost_20x41 uses."""
    return _script("budget_20x41")._closure()


@pytest.fixture(scope="module")
def inc():
    """The backend as scripts/family/newton_cost_20x41.py builds it: the run's params
    (gempack basis), the route's closure, ifSUB-skipped families left out."""
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.templates.gtap.gtap_block_model import compose_block_state

    p = _params(DATASET)
    model, state = compose_block_state(p, p.sets, _route_closure(), list(p.sets.r)[-1])
    backend = FamilyBackend()
    backend.build(model, skip=state.ifsub_skipped)
    assert state.ifsub_skipped and set(backend.records).isdisjoint(state.ifsub_skipped)
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
    """Cross-period families are absent from the incidence, every name in
    CROSS_PERIOD_FAMILIES exists in the backend records or in B (no misspelling),
    and the ``eq_mq_`` prefix matches some row component of B."""
    from equilibria.family.squared_system import (
        _CROSS_PERIOD_PREFIX,
        CROSS_PERIOD_FAMILIES,
        is_cross_period,
    )

    recorded = set(inc.backend.records)
    in_b = {name for _, b in calls for name in b.rows}
    for name in CROSS_PERIOD_FAMILIES:
        assert name in recorded | in_b, f"{name}: not a family of the backend or of B"
    assert any(n.startswith(_CROSS_PERIOD_PREFIX) for n in in_b), (
        f"{_CROSS_PERIOD_PREFIX}*: no row component of B has the prefix"
    )
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
    from equilibria.family.mp_arrays import MPArrays, compare, squared_from_photo
    from equilibria.family.prep_recipe import sp_reference_arrays
    from equilibria.family.prep_square import ConditionalMasks, square

    p = _params(DATASET)
    ref = sp_reference_arrays(p, _closure(), list(p.sets.r)[-1])
    calls = _calls()
    assert calls
    masks = ConditionalMasks.from_params(p, MPArrays.from_photo(calls[0][0]))
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
            masks=masks,
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
    from equilibria.family.prep_square import ConditionalMasks, square
    from equilibria.family.squared_system import SquaredSystem

    p = _params(DATASET)
    ref = sp_reference_arrays(p, _closure(), list(p.sets.r)[-1])
    masks = ConditionalMasks.from_params(p, MPArrays.from_photo(_calls()[0][0]))
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
            masks=masks,
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


# ---------------------------------------------------------------------------
# SquaredSystem against Pyomo's squared system at photo B's point
# ---------------------------------------------------------------------------

F_RTOL = J_RTOL = 1e-9
J_ZERO = 1e-12


@pytest.fixture(scope="module")
def pyomo_mp():
    """The capture route's Pyomo multiperiod model, built as ``_solve_shock`` builds
    it (route closure, ``base_calibrated=True``, reference GDX, NLP mode env), and
    never solved: each call's state is loaded from the photos."""
    import os

    from equilibria.sources import find_reference
    from equilibria.templates.gtap.gtap_block_model import build_block_model

    p = _script("prep_cost_20x41")._params(DATASET)
    ref = find_reference(
        DATASET, "gams", form="mcp", mode="gtap", ifsub=1, fbep="header"
    ) or find_reference(
        DATASET, "gams", form="nlp", mode="gtap", ifsub=1, fbep="header"
    )
    prev = os.environ.get("EQUILIBRIA_GTAP_SOLVE_NLP")
    os.environ["EQUILIBRIA_GTAP_SOLVE_NLP"] = "1"  # the capture runs solve_nlp=True
    try:
        m, _ = build_block_model(
            p,
            p.sets,
            _route_closure(),
            list(p.sets.r)[-1],
            base_calibrated=True,
            ref_gdx=None if ref is None else ref / "out.gdx",
        )
    finally:
        if prev is None:
            os.environ.pop("EQUILIBRIA_GTAP_SOLVE_NLP", None)
        else:
            os.environ["EQUILIBRIA_GTAP_SOLVE_NLP"] = prev
    return m


def _pyomo_at_b(m, a, b):
    """PyomoNLP of ``m`` in photo B's state: every cell at A2's level (A if no A2),
    B's active period on top; free cells exactly B's, active rows exactly B's."""
    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
    from pyomo.environ import Constraint, Objective, Var

    from equilibria.family.prep_capture import _key, load_a2

    base = load_a2(capture_dir(DATASET), a) or a
    n_free = 0
    for comp in m.component_objects(Var, descend_into=True):
        cells = {}
        for photo in (base, b):
            vc = photo.vars.get(comp.name)
            if vc is not None:
                cells.update(
                    zip(vc.keys, zip(vc.value, vc.fixed, strict=True), strict=True)
                )
        for idx in comp:
            vd = comp[idx]
            val, fixed = cells.get(_key(idx), (None, True))
            if val is not None:
                vd.set_value(None if val != val else float(val), skip_validation=True)
            free = (not fixed) and _key(idx)[-1:] == (b.period,)
            if free:
                vd.unfix()
                n_free += 1
            else:
                vd.fix()
    assert n_free == len(b.var_names), (n_free, len(b.var_names))
    active = set(b.con_names)
    for comp in m.component_objects(Constraint, descend_into=True):
        comp.activate()
        for idx in comp:
            cd = comp[idx]
            (cd.activate if cd.name in active else cd.deactivate)()
    for o in m.component_objects(Objective, descend_into=True):
        o.deactivate()
    if hasattr(m, "_family_cmp_obj"):
        m.del_component(m._family_cmp_obj)
    m._family_cmp_obj = Objective(expr=0.0)
    nlp = PyomoNLP(m)
    cons = [c.name for c in nlp.get_pyomo_equality_constraints()]
    assert sorted(cons) == sorted(b.con_names)
    return nlp


def _pairs_not_b(sq, b) -> int:
    """Rows whose paired column differs from Pyomo's pairing (photo B's order)."""
    pb = dict(zip(b.con_names, b.var_names, strict=True))
    return sum(pb[r] != c for r, c in zip(sq.row_names, sq.col_names, strict=True))


@pytest.mark.slow
def test_squared_system_matches_pyomo_3x3(systems, pyomo_mp):
    """F and J of ``SquaredSystem`` equal Pyomo's at photo B's point, on every
    non-proxy row (PyomoNLP of the capture route's model in B's state: B's values,
    fixed flags and active rows)."""
    import numpy as np

    report, bad = [], {}
    for (a, _), (b, ss, _, sq) in zip(_calls(), systems, strict=True):
        assert a.n == b.n
        nlp = _pyomo_at_b(pyomo_mp, a, b)
        rows = [c.name for c in nlp.get_pyomo_equality_constraints()]
        cols = [v.name for v in nlp.get_pyomo_variables()]
        Fp = nlp.evaluate_eq_constraints()
        Jp = nlp.evaluate_jacobian_eq().tocoo()
        # Family side at the same point: B's level for every squared column.
        bval = dict(zip(cols, nlp.get_primals(), strict=True))
        x = np.array([bval.get(c, ss.x0[k]) for k, c in enumerate(sq.col_names)])
        Ff, Jf = ss.F(x), ss.J(x).tocoo()
        proxy = set(ss.proxy_rows)
        rpos = {r: k for k, r in enumerate(sq.row_names)}
        cpos = {c: k for k, c in enumerate(sq.col_names)}
        fp = {r: float(Fp[i]) for i, r in enumerate(rows) if r not in proxy}
        jp: dict[tuple[int, int], float] = {}
        for i, j, v in zip(Jp.row, Jp.col, Jp.data, strict=True):
            if rows[i] not in proxy:
                key = (rpos[rows[i]], cpos[cols[j]])
                jp[key] = jp.get(key, 0.0) + float(v)
        jf: dict[tuple[int, int], float] = {}
        for i, j, v in zip(Jf.row, Jf.col, Jf.data, strict=True):
            if sq.row_names[i] not in proxy:
                jf[(int(i), int(j))] = jf.get((int(i), int(j)), 0.0) + float(v)

        f_rel = {r: abs(Ff[rpos[r]] - v) / max(1.0, abs(v)) for r, v in fp.items()}
        j_rel = {k: abs(jf.get(k, 0.0) - v) / max(1.0, abs(v)) for k, v in jp.items()}
        new = set(jf) - set(jp)  # a family entry Pyomo does not have
        for r, d in f_rel.items():
            if d > F_RTOL:
                bad.setdefault((b.n, "F", _fam(r)), []).append((r, d))
        for (i, j), d in j_rel.items():
            tol = J_ZERO if (i, j) not in jf else J_RTOL
            if (d if (i, j) in jf else abs(jp[(i, j)])) > tol:
                r = sq.row_names[i]
                bad.setdefault((b.n, "J", _fam(r)), []).append((r, sq.col_names[j], d))
        for i, j in new:
            r = sq.row_names[i]
            bad.setdefault((b.n, "J-new", _fam(r)), []).append((r, sq.col_names[j]))
        report.append(
            f"call {b.n} ({b.period}): rows {len(fp)}, J entries {len(jp)}, "
            f"max rel F {max(f_rel.values()):.3e}, max rel J {max(j_rel.values()):.3e}, "
            f"Pyomo-only entries {len(set(jp) - set(jf))} "
            f"(on the diagonal {sum(i == j for i, j in set(jp) - set(jf))}), "
            f"pairs not B's {_pairs_not_b(sq, b)}"
        )
    print("\n".join(report))
    summary = {
        k: (len(v), max((e[-1] for e in v if isinstance(e[-1], float)), default=0.0))
        for k, v in sorted(bad.items())
    }
    for k, v in sorted(bad.items()):
        print(k, len(v), v[:3])
    assert not bad, f"(call, kind, family) -> (rows, max rel diff): {summary}"


def _with_outside_column(sq, proxy, swap: bool):
    """``sq`` plus a cross-period proxy row and a column with no backend cell. With
    ``swap`` the new column is paired with a family row instead of the proxy row."""
    import copy

    out = copy.deepcopy(sq)
    k = next(i for i, r in enumerate(out.row_names) if r not in proxy)
    out.row_names.append("eq_mq_fake[x,check]")
    out.col_names.append("nocell[x,check]")
    out.cells["nocell[x,check]"] = (1.0, False, 0.0, 1.0)
    if swap:
        out.col_names[k], out.col_names[-1] = out.col_names[-1], out.col_names[k]
    return out, out.row_names[k]


def test_column_outside_backend_needs_a_proxy_row(inc, systems):
    """A squared column the backend has no cell for must be paired with a proxy row."""
    from equilibria.family.mp_arrays import MPArrays
    from equilibria.family.squared_system import SquaredSystem

    a = next(a for a, b in _calls() if b.period == "check")
    b, ss, _, sq = next(s for s in systems if s[0].period == "check")
    base = ss.proxy_cols_outside_backend
    print(f"call {b.n}: proxy columns outside the backend {base}")
    ok, _ = _with_outside_column(sq, set(ss.proxy_rows), swap=False)
    got = SquaredSystem.build(
        inc.backend, inc.state, MPArrays.from_photo(a), "check", ok
    )
    assert got.proxy_cols_outside_backend == base + 1
    bad, row = _with_outside_column(sq, set(ss.proxy_rows), swap=True)
    with pytest.raises(ValueError, match=r"nocell\[x,check\]"):
        SquaredSystem.build(
            inc.backend, inc.state, MPArrays.from_photo(a), "check", bad
        )
    print(f"paired with family row {row}: raises")


@pytest.mark.xfail(
    strict=True,
    reason="period_incidence lists the J pattern in layout-column order and lacks "
    "structural-zero columns (sigma=0, zero margins) that Pyomo's identify_variables "
    "keeps; 81 rows/call pair differently (58 with B's order on the pattern's set).",
)
def test_pairs_equal_pyomo_with_period_incidence(inc):
    """The real path's incidence (``period_incidence`` over B's for cross-period rows)
    gives Pyomo's pairs: 0 differing rows per call."""
    from equilibria.family.mp_arrays import MPArrays
    from equilibria.family.prep_recipe import run_recipe, sp_reference_arrays
    from equilibria.family.prep_square import ConditionalMasks, square
    from tests.family.test_prep_recipe import ROUTE

    p = _params(DATASET)
    closure = _closure()
    ref = sp_reference_arrays(p, closure, list(p.sets.r)[-1])
    masks = None
    for a, b in _calls():
        mp = run_recipe(
            MPArrays.from_photo(a),
            a.period,
            params=p,
            closure=closure,
            sp_ref=ref,
            mode="gtap",
            **ROUTE,
        )
        masks = masks or ConditionalMasks.from_params(p, mp)
        merged = {r: list(c) for r, c in zip(b.con_names, b.incidence, strict=True)}
        merged.update(inc(a.period))
        got = square(
            mp,
            a.period,
            params=p,
            closure=closure,
            incidence=merged,
            exogenous=ref.exogenous,
            masks=masks,
        )
        want = dict(zip(b.con_names, b.var_names, strict=True))
        differ = [
            r for r, c in zip(got.row_names, got.col_names, strict=True) if want[r] != c
        ]
        assert differ == [], f"call {b.n}: {len(differ)} rows differ, e.g. {differ[:5]}"
