"""Family/FamilySystem on a toy CES price family with a closed-form Jacobian.

Toy family (3 rows r, 2 terms per row i):
    residual[r] = p[r]**e[r] - sum_i a[r,i] * (q[r,i] / l[r,i])**e[r]
Free columns: p[0..2] -> 0..2, q[r,i] -> 3 + 2r + i. Fixed slots start at n_cols.
"""

import numpy as np
import pytest

pytest.importorskip("jax")

from equilibria.family.core import Family, FamilySystem  # noqa: E402


def _row_fn(xs, e):
    return xs[0] ** e


def _term_fn(xs, a, l, e):
    return a * (xs[0] / l) ** e


E = np.array([-0.5, 0.3, 1.7])
A = np.array([[0.4, 0.6], [0.2, 0.8], [0.5, 0.5]])
L = np.array([[1.0, 1.1], [0.9, 1.0], [1.2, 0.8]])


def _toy(q_cols=None, fixed_values=()):
    n_cols = 9
    q_cols = np.arange(3, 9).reshape(3, 2) if q_cols is None else q_cols
    fam = Family(
        name="toy",
        row_fn=_row_fn,
        row_cols=np.arange(3).reshape(3, 1),
        row_args={"e": E},
        term_fn=_term_fn,
        term_rows=np.repeat(np.arange(3), 2),
        term_cols=q_cols.reshape(6, 1),
        term_args={"a": A.ravel(), "l": L.ravel(), "e": np.repeat(E, 2)},
    )
    return FamilySystem(
        [fam], n_cols=n_cols, fixed_values=np.asarray(fixed_values, float)
    )


def _z():
    return np.array([1.1, 0.9, 1.3, 0.8, 1.2, 1.05, 0.95, 1.4, 0.7])


def _closed_form(z, q=None):
    p = z[:3]
    q = z[3:9].reshape(3, 2) if q is None else q
    F = p**E - (A * (q / L) ** E[:, None]).sum(1)
    J = np.zeros((3, 9))
    J[np.arange(3), np.arange(3)] = E * p ** (E - 1)
    dq = -A * E[:, None] * q ** (E[:, None] - 1) / L ** E[:, None]
    for r in range(3):
        for i in range(2):
            J[r, 3 + 2 * r + i] = dq[r, i]
    return F, J


def test_ces_toy_matches_hand_derivative():
    sysm = _toy()
    z = _z()
    F, J = _closed_form(z)
    np.testing.assert_allclose(sysm.F(z), F, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(sysm.jac(z).toarray(), J, rtol=1e-14, atol=1e-15)
    assert sysm.nnz == 9
    assert sysm.rows == [("toy", 0), ("toy", 1), ("toy", 2)]


def test_fixed_column_dropped_from_pattern():
    # q[1,0] (col 5) becomes fixed slot n_cols + 0 = 9 with value 1.05 (same as z[5]).
    q_cols = np.arange(3, 9).reshape(3, 2).copy()
    q_cols[1, 0] = 9
    sysm = _toy(q_cols=q_cols, fixed_values=[1.05])
    z = _z()
    F, J = _closed_form(z)
    np.testing.assert_allclose(sysm.F(z), F, rtol=1e-14, atol=1e-15)
    Jc = sysm.jac(z).tocoo()
    assert 5 not in set(Jc.col.tolist())
    assert sysm.nnz == 8


def test_pattern_is_index_arrays():
    # At a point where every derivative wrt q is zero (q=0, e>1 only for row 2),
    # the structural pattern still carries all 9 entries.
    sysm = _toy()
    z = _z()
    z[3:9] = 0.0
    z[:3] = 1.0
    Jc = sysm.jac(z).tocoo()
    got = sorted(zip(Jc.row.tolist(), Jc.col.tolist(), strict=True))
    want = sorted(
        [(r, r) for r in range(3)]
        + [(r, 3 + 2 * r + i) for r in range(3) for i in range(2)]
    )
    assert got == want


def test_row_names_use_labels():
    sysm = _toy()
    sysm.families[0].row_labels = ["a", "b", "c"]
    assert sysm.row_names == ["toy[a]", "toy[b]", "toy[c]"]


def test_no_recompile_on_second_eval():
    from equilibria.family import core

    kernels = (core._eval_part, core._jac_part, core._scatter)
    sysm = _toy()
    z = _z()
    sysm.F(z)
    sysm.jac(z)
    before = [k._cache_size() for k in kernels]
    # Same shapes, new values and a fresh system object: no new compilation.
    sysm2 = _toy()
    sysm2.F(z * 1.1)
    sysm2.jac(z * 1.1)
    assert [k._cache_size() for k in kernels] == before
