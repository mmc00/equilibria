"""``period_incidence`` read from the stored J pattern equals the one from evaluating J.

The oracle is the code before the change: ``_legacy_pattern`` evaluates ``system.jac``
and lists each row's columns sorted; keys, per-row column order and cross rows must
come out ``==``.
"""

from __future__ import annotations

import pytest

from tests.family.test_squared_system import _backends

DATASETS = ["gtap7_3x3", "gtap7_10x7"]


def _legacy_pattern(backend, layout, period):
    from equilibria.family.squared_system import _with_period

    system = backend.system(layout)
    J = system.jac(layout.free_values).tocsr()
    cols = list(layout.col_names)
    rows = list(system.row_names)
    if period is not None:
        cols = [_with_period(n, period) for n in cols]
        rows = [_with_period(n, period) for n in rows]
    out = {}
    for i, name in enumerate(rows):
        lo, hi = J.indptr[i], J.indptr[i + 1]
        out[name] = [cols[j] for j in sorted(J.indices[lo:hi])]
    return out


def _legacy_period_incidence(backend, state, period, cross=None):
    from equilibria.family.squared_system import (
        _all_free_layout,
        _cross_pattern,
        is_cross_period,
    )

    out = {
        k: v
        for k, v in _legacy_pattern(
            backend, _all_free_layout(backend, state), period
        ).items()
        if not is_cross_period(k.partition("[")[0])
    }
    if cross is not None:
        for name, cols in _cross_pattern(cross).items():
            if name.endswith(f",{period}]") or name.endswith(f"[{period}]"):
                out[name] = cols
    return out


@pytest.fixture(scope="module", params=DATASETS)
def route(request):
    p, backend, state, cross = _backends(request.param)
    return backend, state, cross


@pytest.mark.parametrize("period", ["base", "check", "shock"])
def test_period_incidence_equals_the_evaluated_one(route, period):
    from equilibria.family.squared_system import period_incidence

    backend, state, cross = route
    for cr in (None, cross):
        want = _legacy_period_incidence(backend, state, period, cr)
        got = period_incidence(backend, state, period, cr)
        assert list(got) == list(want)
        assert got == want  # per-row column order included


def test_repeated_calls_do_not_share_the_dict(route):
    from equilibria.family.squared_system import period_incidence

    backend, state, cross = route
    a = period_incidence(backend, state, "check", cross)
    a.clear()
    assert period_incidence(backend, state, "check", cross)


def test_csr_is_cached_on_the_backend(route):
    from equilibria.family.squared_system import incidence_csr

    backend, state, _ = route
    assert incidence_csr(backend, state) is incidence_csr(backend, state)
    assert incidence_csr(backend, state).indices.dtype.kind == "i"
