"""``SquaredSystem.build`` on integer maps equals the per-cell version it replaced.

The functions named ``_legacy_*`` are the code before the change: ``compile_family``
(terms regrouped per layout, one Python pass over every term), ``_single_period_part``
and ``build`` (names, dicts, a loop over every cell). The groups must be
``np.array_equal`` (same bucket order, same arrays) and ``F`` / ``J`` bitwise equal on
every captured call of the 3x3 and 10x7 route.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pytest

from equilibria.family.recorder import (
    Residual,
    fold_zeros,
    residual_terms,
    signature,
    template,
)
from tests.family import test_square_cell_ids as _cells
from tests.family.test_square_cell_ids import _kw

route = _cells.route  # the module's fixture: the route's recipe output per dataset


def _legacy_compile_family(name: str, records, layout) -> Any:
    """Recorded cells of one equation -> ``CompiledFamily`` (rows + term groups by shape).

    ``records`` is ``[(indices, Residual), ...]`` for the cells the rule did not skip.
    """
    from equilibria.family.core import CompiledFamily, TermGroup

    flat_slots = {v: np.asarray(a).ravel() for v, a in layout.slots.items()}
    n_cols = layout.n_cols
    fixed = np.asarray(layout.fixed_values, dtype=float)

    def is_fixed_zero(leaf) -> bool:
        slot = flat_slots[leaf[0]][leaf[1]]
        return bool(slot >= n_cols and fixed[slot - n_cols] == 0.0)

    def signed_terms(res):
        """Terms of the residual; a term touching a fixed cell at 0 is re-simplified
        with that 0 in place (it may vanish or split), as the ``.nl`` writer does."""
        for sgn, term in residual_terms(res):
            key, leaves, consts = signature(term)
            if any(is_fixed_zero(lf) for lf in leaves):
                folded = fold_zeros(term, is_fixed_zero)
                for s2, t2 in residual_terms(Residual(folded, 0.0)):
                    yield (sgn * s2, *signature(t2))
            else:
                yield sgn, key, leaves, consts

    buckets: dict[tuple, dict[str, list]] = {}
    labels = []
    for row, (idx, res) in enumerate(records):
        labels.append(",".join(str(i) for i in idx))
        for sgn, key, leaves, consts in signed_terms(res):
            b = buckets.get(key)
            if b is None:
                b = buckets[key] = {"rows": [], "leaves": [], "c": [], "s": []}
            b["rows"].append(row)
            b["leaves"].append(leaves)
            b["c"].append(consts)
            b["s"].append(sgn)

    groups = []
    for key, b in buckets.items():
        n = len(b["rows"])
        k = len(b["leaves"][0])
        cols = np.empty((n, k), dtype=np.int64)
        for t, leaves in enumerate(b["leaves"]):
            for j, (var, flat) in enumerate(leaves):
                cols[t, j] = flat_slots[var][flat]
        groups.append(
            TermGroup(
                rows=np.asarray(b["rows"], dtype=np.int64),
                fn=template(key),
                cols=cols,
                args={
                    "c": np.asarray(b["c"], dtype=float).reshape(n, -1),
                    "s": np.asarray(b["s"], dtype=float),
                },
            )
        )
    return CompiledFamily(name=name, row_labels=labels, groups=groups)


def _legacy_single_period_part(backend, state, mp, period, sq, col_pos):
    from equilibria.family.layout import VarLayout
    from equilibria.family.squared_system import (
        SP_ONLY_VARS,
        _cell_name,
        _with_period,
    )

    values, free = {}, {}
    for v, doms in backend.var_domains.items():
        shape = tuple(len(state.sets[d]) for d in doms)
        vals = np.empty(shape)
        mask = np.zeros(shape, dtype=bool)
        var = mp.vars.get(v)
        for k, idx in enumerate(itertools.product(*(state.sets[d] for d in doms))):
            name = _cell_name(v, idx, period)
            if name in sq.cells:
                vals.flat[k] = sq.cells[name][0]
            elif var is not None:
                vals.flat[k] = var.value[var.pos((*idx, period))]
            elif v in SP_ONLY_VARS:
                vals.flat[k] = np.asarray(state.values[v], dtype=float).flat[k]
            else:
                raise KeyError(f"variable component {v!r} missing from mp")
            mask.flat[k] = name in col_pos
        values[v], free[v] = vals, mask
    for p in backend.param_domains:
        values[p] = state.find_component(p).data
        free[p] = np.zeros(values[p].shape, dtype=bool)
    layout = VarLayout.from_state(
        state.sets, {**backend.var_domains, **backend.param_domains}, values, free
    )
    names = [_with_period(n, period) for n in layout.col_names]
    ctx = (backend.records, values, free)
    return backend.system(layout), names, lambda n: _with_period(n, period), ctx


def _legacy_cross_period_part(cross, mp):
    from equilibria.family.layout import VarLayout
    from equilibria.family.squared_system import _flat_index

    sets = cross.sets
    values, free = {}, {}
    for v, doms in cross.var_domains.items():
        shape = tuple(len(sets[d]) for d in doms)
        vals = np.ones(shape)
        mask = np.zeros(shape, dtype=bool)
        var = mp.var(v)
        flat = _flat_index(var, doms, sets)
        vals.flat[flat] = var.value
        mask.flat[flat] = ~var.fixed
        values[v], free[v] = vals, mask
    layout = VarLayout.from_state(sets, cross.var_domains, values, free)
    ctx = (cross.records, values, free)
    return cross.system(layout), list(layout.col_names), lambda n: n, ctx


def _legacy_build(backend, state, mp, period, sq, cross=None, proxies=False):
    from equilibria.family.squared_system import (
        SquaredSystem,
        is_cross_period,
    )

    col_pos = {c: k for k, c in enumerate(sq.col_names)}
    row_pos = {r: k for k, r in enumerate(sq.row_names)}
    raw = [_legacy_single_period_part(backend, state, mp, period, sq, col_pos)]
    if cross is not None:
        raw.append(_legacy_cross_period_part(cross, mp))
    x0 = np.empty(len(sq.col_names))
    for name, k in col_pos.items():
        x0[k] = sq.cells[name][0]
    parts, covered, backend_cols = [], set(), np.zeros(len(sq.col_names), bool)
    fscale_ctx = []
    for i, (system, col_names, row_name, ctx) in enumerate(raw):
        fscale_ctx.append((*ctx, row_name, i == 0))
        zperm = np.array([col_pos[n] for n in col_names], dtype=np.int64)
        backend_cols[zperm] = True
        fam_rows, fam_k = [], []
        for r, name in enumerate(system.row_names):
            k = row_pos.get(row_name(name))
            if k is None or (i == 0 and is_cross_period(name.partition("[")[0])):
                continue
            fam_rows.append(r)
            fam_k.append(k)
        fam_rows, fam_k = np.array(fam_rows, np.int64), np.array(fam_k, np.int64)
        covered |= set(fam_k.tolist())
        row_of = np.full(system.n_rows, -1, dtype=np.int64)
        row_of[fam_rows] = fam_k
        parts.append((system, fam_rows, fam_k, zperm, row_of))
    proxy_k = np.setdiff1d(np.arange(len(sq.row_names)), np.array(sorted(covered)))
    outside = np.flatnonzero(~backend_cols)
    ss = SquaredSystem(x0, parts, sq.row_names, proxy_k, len(outside))
    ss._row_pos = row_pos
    ss._fscale_ctx = fscale_ctx
    return ss


def _same_groups(a, b):
    assert a.name == b.name and a.row_labels == b.row_labels
    assert len(a.groups) == len(b.groups)
    for g, h in zip(a.groups, b.groups, strict=True):
        assert g.fn is h.fn and g.sign == h.sign
        for x, y in ((g.rows, h.rows), (g.cols, h.cols)):
            assert x.dtype == y.dtype and x.shape == y.shape
            assert np.array_equal(x, y)
        assert g.args.keys() == h.args.keys()
        for key in g.args:
            assert g.args[key].dtype == h.args[key].dtype
            assert g.args[key].shape == h.args[key].shape
            assert np.array_equal(g.args[key], h.args[key])


def _squared(route, k):
    from equilibria.family.prep_square import square
    from equilibria.family.squared_system import ArrayIncidence

    p, backend, state, cross, ref, masks, mps, periods = route
    inc = ArrayIncidence(backend, state, periods[k], cross)
    return square(
        mps[k], periods[k], incidence=inc, return_mp=True, **_kw(p, ref, masks)
    )


def _build_layouts(route, k, monkeypatch):
    """(backend, layout) of every ``system`` call of a build."""
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.family.squared_system import SquaredSystem

    p, backend, state, cross, *_ = route
    sq, mp_sq = _squared(route, k)
    seen = []
    orig = FamilyBackend.system

    def spy(self, layout):
        seen.append((self, layout))
        return orig(self, layout)

    monkeypatch.setattr(FamilyBackend, "system", spy)
    SquaredSystem.build(backend, state, mp_sq, route[7][k], sq, cross=cross)
    monkeypatch.undo()
    assert len(seen) == 2
    return seen


@pytest.mark.parametrize("k", range(3))
def test_compile_family_groups_equal_the_legacy_ones(route, k, monkeypatch):
    from equilibria.family.recorder import compile_family

    n = 0
    for be, lay in _build_layouts(route, k, monkeypatch):
        for name, recs in be.records.items():
            _same_groups(
                compile_family(name, recs, lay), _legacy_compile_family(name, recs, lay)
            )
            n += 1
    assert n


def _legacy_system(route, k, monkeypatch, **kw):
    """The legacy build on the legacy compile (per-layout term regrouping)."""
    import equilibria.backends.family_backend as fb

    p, backend, state, cross, *_ = route
    sq, mp_sq = _squared(route, k)
    monkeypatch.setattr(
        fb,
        "compile_family",
        lambda n, r, lay, table=None: _legacy_compile_family(n, r, lay),
    )
    try:
        ss = _legacy_build(backend, state, mp_sq, route[7][k], sq, **kw)
    finally:
        monkeypatch.undo()
    return ss, sq, mp_sq


@pytest.mark.parametrize("k", range(3))
def test_build_equals_the_legacy_build(route, k, monkeypatch):
    from equilibria.family.squared_system import SquaredSystem

    p, backend, state, cross, *_ = route
    period = route[7][k]
    want, sq, mp_sq = _legacy_system(route, k, monkeypatch, cross=cross)
    got = SquaredSystem.build(backend, state, mp_sq, period, sq, cross=cross)
    assert np.array_equal(got.x0, want.x0)
    assert np.array_equal(got._proxy_k, want._proxy_k)
    assert got.proxy_rows == want.proxy_rows
    assert got.proxy_cols_outside_backend == want.proxy_cols_outside_backend
    assert got.proxy_families == want.proxy_families
    assert len(got._parts) == len(want._parts)
    for (_, r1, k1, z1, o1), (_, r2, k2, z2, o2) in zip(
        got._parts, want._parts, strict=True
    ):
        for x, y in ((r1, r2), (k1, k2), (z1, z2), (o1, o2)):
            assert x.dtype == y.dtype and np.array_equal(x, y)
    assert got._row_pos == want._row_pos
    assert np.array_equal(got.fscale(), want.fscale())
    x = got.x0
    for _ in range(3):
        assert np.array_equal(got.F(x), want.F(x))
        Jg, Jw = got.J(x).tocsr(), want.J(x).tocsr()
        assert np.array_equal(Jg.indptr, Jw.indptr)
        assert np.array_equal(Jg.indices, Jw.indices)
        assert np.array_equal(Jg.data, Jw.data)


def test_build_without_cross_equals_the_legacy_build(route, monkeypatch):
    from equilibria.family.squared_system import SquaredSystem

    p, backend, state, cross, *_ = route
    if len(state.sets["r"]) > 3:
        pytest.skip("only 3x3 squares without the cross rows' columns")
    want, sq, mp_sq = _legacy_system(route, 1, monkeypatch, proxies=True)
    got = SquaredSystem.build(backend, state, mp_sq, route[7][1], sq, proxies=True)
    assert np.array_equal(got.x0, want.x0)
    assert np.array_equal(got.F(got.x0), want.F(got.x0))
    assert np.array_equal(got.J(got.x0).toarray(), want.J(got.x0).toarray())
    assert got.proxy_rows == want.proxy_rows


def test_build_does_not_read_the_cell_dict(route):
    from equilibria.family.squared_system import SquaredSystem

    p, backend, state, cross, *_ = route
    sq, mp_sq = _squared(route, 0)
    SquaredSystem.build(backend, state, mp_sq, route[7][0], sq, cross=cross)
    assert callable(sq._cells)  # the lazy dict was not built
