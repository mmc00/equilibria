"""``newton_tr`` GMIN policy: ``on_null_pivot`` regularizes only when MUMPS needs it."""

import numpy as np
import pytest
import scipy.sparse as sp

from equilibria.solver.newton_tr import newton_tr


def _cubic(n=2):
    """F = (a^2 + b^2 - 2, a - b): at the origin row 0 of J is (stored) zeros, so the
    first factorization has a null pivot. The root is a = b = 1."""

    def F(x):
        return np.array([x[0] ** 2 + x[1] ** 2 - 2.0, x[0] - x[1]])

    def J(x):
        data = np.array([2.0 * x[0], 2.0 * x[1], 1.0, -1.0])  # explicit zeros kept
        return sp.csr_matrix((data, [0, 1, 0, 1], [0, 2, 4]), shape=(2, 2))

    return F, J, n


def _linear(n=5):
    """F = A x - b with a regular A: no factorization ever has a null pivot."""
    A = sp.diags([2.0 * np.ones(n), -np.ones(n - 1), -np.ones(n - 1)], [0, 1, -1])
    A = sp.csr_matrix(A)
    b = np.ones(n)
    return (lambda x: A @ x - b), (lambda x: A), n


def _run(system, **kw):
    F, J, n = system
    return newton_tr(
        F,
        J,
        np.zeros(n),
        np.full(n, -1e6),
        np.full(n, 1e6),
        colperm=np.arange(n),
        linsolve="mumps",
        icntl6=0,
        maxit=5,
        **kw,
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("EQUILIBRIA_GTAP_GMIN", raising=False)
    monkeypatch.delenv("EQUILIBRIA_GTAP_TR_RELTOL", raising=False)


def test_rejects_unknown_policy():
    F, J, n = _linear()
    with pytest.raises(ValueError):
        newton_tr(
            F, J, np.zeros(n), np.full(n, -1e6), np.full(n, 1e6), gmin_policy="nope"
        )


def test_regular_system_never_uses_gmin():
    pytest.importorskip("mumps")
    res = _run(_linear(), gmin=1e-9, gmin_policy="on_null_pivot")
    assert res.success
    assert res.gmin_factors == 0 and set(res.null_pivots) == {0}
    assert res.cfg["gmin"] == 1e-9 and res.cfg["gmin_policy"] == "on_null_pivot"


def test_singular_system_retries_with_gmin():
    pytest.importorskip("mumps")
    res = _run(_cubic(), gmin=1e-9, gmin_policy="on_null_pivot")
    assert res.null_pivots[0] > 0  # the unregularized factor saw the null pivots
    assert res.gmin_factors >= 1  # ... and was refactored with GMIN
    assert res.nfact == len(res.null_pivots) >= 2
    # every retry follows a factor with null pivots
    assert res.gmin_factors == sum(1 for k in res.null_pivots if k > 0)


def test_default_policy_adds_gmin_to_every_factor():
    pytest.importorskip("mumps")
    res = _run(_linear(), gmin=1e-9)
    assert res.gmin_factors == res.nfact >= 1


def test_default_without_gmin_never_uses_it():
    pytest.importorskip("mumps")
    res = _run(_linear())
    assert res.gmin_factors == 0
