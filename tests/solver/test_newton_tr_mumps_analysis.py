"""``newton_tr(mumps_values_at_analysis=True)``: the MUMPS analysis sees the values.

Pyomo's MUMPS wrapper analyses the pattern alone (``job=1`` before any values), so
ICNTL(6)=7 and ICNTL(8)=8 cannot use them and the factor can meet a small pivot that a
value-aware ordering avoids. The family route asks for the values at analysis; the
default (the Pyomo path) is unchanged.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from equilibria.solver.newton_tr import newton_tr

# 9x9, cond ~1.3e10 (a near-singular direction as the 20x41 capFlex settle J has):
# analysed on the pattern alone MUMPS (ICNTL 6=7, 8=8, CNTL(3)=-1e-8) reports one null
# pivot; analysed with the values it reports none.
_ROWS = [0, 0, 0, 1, 1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8]
_COLS = [5, 8, 4, 5, 2, 0, 2, 8, 4, 3, 7, 6, 8, 6, 5, 2, 0, 6, 5, 2, 0, 8, 7, 1, 4, 3]
_VALS = [
    6.829515111988866e-08, -8.355316442887375, -7.537171803610085,
    0.8619771699627912, -1.216595456227999, 1.1885732789034003,
    2.3369716882228424, 0.3841124830165303, 0.906145679269503,
    0.14451330261762244, 1.2969022049990147, -0.13981148067041688,
    0.6870351412529035, 0.4086915118100357, -1.6958991208028968,
    2.39359374759186, -2.338461445317751, 0.013655033447662349,
    0.22886871567876096, -0.1167373915146341, 0.31558520261558387,
    -6.9705193483146735, 10.493056779023267, 3.764398981435745,
    30.896734952452558, 7.9782508710794025,
]  # fmt: skip


def _near_singular():
    A = sp.csr_matrix((np.array(_VALS), (_ROWS, _COLS)), shape=(9, 9))
    x_star = np.linspace(0.5, 1.5, 9)
    b = A @ x_star
    return (lambda x: A @ x - b), (lambda x: A), A, b


def _run(F, J, n, **kw):
    return newton_tr(
        F,
        J,
        np.zeros(n),
        np.full(n, -1e6),
        np.full(n, 1e6),
        colperm=np.arange(n),
        linsolve="mumps",
        icntl6=7,
        maxit=3,
        gmin=1e-9,
        gmin_policy="on_null_pivot",
        **kw,
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in (
        "EQUILIBRIA_GTAP_GMIN",
        "EQUILIBRIA_GTAP_TR_RELTOL",
        "EQUILIBRIA_GTAP_MUMPS_ICNTL7",
        "EQUILIBRIA_GTAP_MUMPS_ICNTL8",
        "EQUILIBRIA_GTAP_MUMPS_ICNTL24",
        "EQUILIBRIA_GTAP_MUMPS_CNTL3",
    ):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def mumps_calls(monkeypatch):
    """Every ``set_centralized_assembled_values`` and ``run(job)`` on any context."""
    mumps = pytest.importorskip("mumps")
    log: list = []
    ctx = mumps.DMumpsContext
    set_values, run = ctx.set_centralized_assembled_values, ctx.run

    def spy_values(self, a):
        log.append("values")
        return set_values(self, a)

    def spy_run(self, job):
        log.append(job)
        return run(self, job=job)

    monkeypatch.setattr(ctx, "set_centralized_assembled_values", spy_values)
    monkeypatch.setattr(ctx, "run", spy_run)
    return log


def _before_analysis(log):
    return log[: log.index(1)]


def test_default_analyses_the_pattern_alone(mumps_calls):
    F, J, _, _ = _near_singular()
    _run(F, J, 9)
    assert 1 in mumps_calls and "values" not in _before_analysis(mumps_calls)


def test_values_at_analysis_hands_the_values_first(mumps_calls):
    F, J, _, _ = _near_singular()
    _run(F, J, 9, mumps_values_at_analysis=True)
    assert _before_analysis(mumps_calls)[-1] == "values"


def test_pattern_alone_meets_a_spurious_null_pivot():
    pytest.importorskip("mumps")
    F, J, _, _ = _near_singular()
    res = _run(F, J, 9)
    assert res.null_pivots[0] == 1 and res.gmin_factors >= 1


def test_values_at_analysis_needs_no_gmin():
    pytest.importorskip("mumps")
    F, J, _, _ = _near_singular()
    res = _run(F, J, 9, mumps_values_at_analysis=True)
    assert set(res.null_pivots) == {0} and res.gmin_factors == 0
    assert np.max(np.abs(F(res.x))) < 1e-10
