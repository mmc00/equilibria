"""``newton_tr``: the standalone Newton trust-region solver shared by both paths."""

import numpy as np
import pytest
import scipy.sparse as sp

from equilibria.solver.newton_tr import NewtonResult, newton_tr


def _system(n=40, seed=0):
    """F(x) = A x + 0.1 x^3 - b, with a zero diagonal in A (needs the column pairing)."""
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=0.15, random_state=seed, format="csr") * 0.3
    A = A + sp.csr_matrix((np.ones(n), (np.arange(n), (np.arange(n) + 1) % n)), (n, n))
    A.setdiag(0.0)
    A.eliminate_zeros()
    b = rng.normal(size=n)

    def F(x):
        return A @ x + 0.1 * x**3 - b

    def J(x):
        return A + sp.diags(0.3 * x**2)

    return F, J, n


@pytest.mark.parametrize("pairing", ["array", "hook", "none"])
def test_solves_and_reports(pairing, monkeypatch):
    for k in ("EQUILIBRIA_GTAP_TR_RELTOL", "EQUILIBRIA_GTAP_GMIN"):
        monkeypatch.delenv(k, raising=False)
    F, J, n = _system()
    x0 = np.zeros(n)
    lb, ub = np.full(n, -1e6), np.full(n, 1e6)
    perm = np.asarray(
        (np.arange(n) + 1) % n
    )  # column i+1 sits on the diagonal of row i
    kw = {}
    if pairing == "array":
        kw["colperm"] = perm
    elif pairing == "hook":
        kw["paired_colperm"] = lambda: (perm, n)
    res = newton_tr(F, J, x0, lb, ub, linsolve="direct", **kw)
    assert isinstance(res, NewtonResult)
    assert res.success and res.nit >= 1 and res.nfact >= 1
    assert np.linalg.norm(F(res.x), np.inf) < 1e-7
    assert len(res.trace) == res.nit
    assert res.trace[-1][0] < 1e-7 and res.trace[0][0] > res.trace[-1][0]
    assert res.linsolve == "direct" and res.null_pivots == []


def test_fscale_reltol_converges_on_the_scaled_residual(monkeypatch):
    monkeypatch.delenv("EQUILIBRIA_GTAP_TR_RELTOL", raising=False)
    F, J, n = _system()
    x0 = np.zeros(n)
    big = np.full(n, 1e6)
    perm = (np.arange(n) + 1) % n
    fs = np.full(n, 1e3)
    res = newton_tr(
        F, J, x0, -big, big, colperm=perm, fscale=fs, reltol=1e-3, linsolve="direct"
    )
    assert res.success
    assert res.trace[-1][1] is not None and res.trace[-1][1] < 1e-3


def test_structural_gate_names_the_empty_row():
    n = 5
    A = sp.diags(np.ones(n)).tolil()
    A[2, 2] = 0.0  # row 2 and column 2 empty
    A = A.tocsr()
    A.eliminate_zeros()
    rows = [f"eq{i}" for i in range(n)]
    cols = [f"v{i}" for i in range(n)]
    with pytest.raises(RuntimeError, match=r"eq2"):
        newton_tr(
            lambda x: A @ x - 1.0,
            lambda x: A,
            np.zeros(n),
            np.full(n, -10.0),
            np.full(n, 10.0),
            colperm=np.arange(n),
            row_names=rows,
            col_names=cols,
            linsolve="direct",
        )


def test_fixed_empty_column_is_not_a_singularity():
    n = 3
    A = sp.csr_matrix(np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 0.0]]))
    A.eliminate_zeros()
    # empty row 2 still trips the gate: only columns get the fixed exemption
    with pytest.raises(RuntimeError):
        newton_tr(
            lambda x: A @ x - 1.0,
            lambda x: A,
            np.zeros(n),
            np.full(n, -9.0),
            np.full(n, 9.0),
            colperm=np.arange(n),
            col_is_fixed=lambda j: j == 2,
            linsolve="direct",
        )


def test_mumps_matches_direct_to_roundoff(monkeypatch):
    pytest.importorskip("mumps")
    for k in ("EQUILIBRIA_GTAP_TR_RELTOL", "EQUILIBRIA_GTAP_GMIN"):
        monkeypatch.delenv(k, raising=False)
    F, J, n = _system()
    x0 = np.zeros(n)
    big = np.full(n, 1e6)
    perm = (np.arange(n) + 1) % n
    a = newton_tr(F, J, x0, -big, big, colperm=perm, linsolve="direct")
    for icntl6 in (None, 7):
        symbolic = []
        b = newton_tr(
            F,
            J,
            x0,
            -big,
            big,
            colperm=perm,
            linsolve="mumps",
            icntl6=icntl6,
            on_symbolic_factorization=lambda: symbolic.append(1),
        )
        assert b.success and 1 <= len(symbolic) <= b.nfact
        np.testing.assert_allclose(b.x, a.x, rtol=1e-9, atol=1e-10)
        assert b.cfg["icntl"]["6"] == (0 if icntl6 is None else icntl6)
