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


def _offdiag_null():
    """F = (b^2 - 1, a + b - 2): row 0 stores no (0, 0) entry, so the GMIN retry
    analyses a pattern with one more nonzero than J·P; at b = 0 row 0 is null. The
    root is a = b = 1, reached after the retry's step is accepted."""

    def F(x):
        return np.array([x[1] ** 2 - 1.0, x[0] + x[1] - 2.0])

    def J(x):
        data = np.array([2.0 * x[1], 1.0, 1.0])
        return sp.csr_matrix((data, [1, 0, 1], [0, 1, 3]), shape=(2, 2))

    return F, J


def test_factor_after_gmin_retry_reanalyses_the_plain_pattern(capfd):
    """After a GMIN retry the live analysis is on J·P + diag; the next factor (GMIN on
    keeps one per step, even when the step was accepted and the LU is young) must
    re-analyse J·P, not hand MUMPS a matrix with fewer nonzeros ("nonzeros changed")."""
    pytest.importorskip("mumps")
    F, J = _offdiag_null()
    res = newton_tr(
        F,
        J,
        np.array([0.5, 0.0]),
        np.full(2, -1e6),
        np.full(2, 1e6),
        colperm=np.arange(2),
        linsolve="mumps",
        icntl6=0,
        maxit=20,
        gmin=1e-9,
        gmin_policy="on_null_pivot",
    )
    err = capfd.readouterr().err
    assert res.success
    assert res.null_pivots[0] > 0 and res.gmin_factors == 1
    assert "MUMPS FAILED" not in err
    assert res.factor_failures == 0


def test_failed_factorization_is_counted(monkeypatch):
    """A MUMPS factorization that raises is counted in ``factor_failures`` (the solve
    falls back to the gradient step for that iteration)."""
    pytest.importorskip("mumps")
    from pyomo.contrib.pynumero.linalg import mumps_interface as mi

    orig = mi.MumpsCentralizedAssembledLinearSolver.do_numeric_factorization
    calls = {"n": 0}

    def flaky(self, matrix, raise_on_error=True):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("injected factorization failure")
        return orig(self, matrix, raise_on_error)

    monkeypatch.setattr(
        mi.MumpsCentralizedAssembledLinearSolver, "do_numeric_factorization", flaky
    )
    res = _run(_linear())
    assert res.factor_failures == 1
