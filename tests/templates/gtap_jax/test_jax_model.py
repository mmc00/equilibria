"""JaxGTAPModel: a PEDRO MCPModel whose F and (sparsejac) Jacobian reproduce Pyomo's.

Validated on a SQUARE Pyomo equality system (a small hand-built CES-like square model), so the
MCPModel contract + sparsejac Jacobian correctness are checked without the full GTAP driver's
squaring. Parity on the real (driver-squared) GTAP system is Task 6 (which needs the driver to
produce the square _solve_target). JaxGTAPModel(m) requires m to be ALREADY SQUARE
(#free vars == #active equality constraints) — the driver's newton_tr path guarantees this.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "gtap"))

from equilibria.templates.gtap_jax.jax_model import JaxGTAPModel


def _square_ces_model(n=12, seed=0):
    """A square Pyomo equality system with CES-like bodies: for each i,
        f_i:  y[i] == a_i * x[i] * (p[i]/q[i])**s_i
    with y,x,p,q the 4 vars per cell but we FIX x,p,q so each row has exactly one free var
    (y[i]) -> n eqs, n free vars -> square. This exercises translate/build_F/sparsejac/CSR
    without the GTAP driver. The Jacobian is diag (df_i/dy_i = 1)."""
    from pyomo.environ import ConcreteModel, Var, Constraint, value
    rng = np.random.default_rng(seed)
    m = ConcreteModel()
    m.I = range(n)
    a = rng.uniform(0.5, 2.0, n); s = rng.uniform(0.5, 1.5, n)
    x = rng.uniform(0.5, 2.0, n); p = rng.uniform(0.5, 2.0, n); q = rng.uniform(0.5, 2.0, n)
    m.y = Var(m.I, initialize=lambda m, i: 1.0)
    m.x = Var(m.I, initialize=lambda m, i: float(x[i]));
    m.p = Var(m.I, initialize=lambda m, i: float(p[i]))
    m.q = Var(m.I, initialize=lambda m, i: float(q[i]))
    for i in m.I:
        m.x[i].fix(); m.p[i].fix(); m.q[i].fix()

    def _eq(m, i):
        return m.y[i] == a[i] * m.x[i] * (m.p[i] / m.q[i]) ** s[i]
    m.eq = Constraint(m.I, rule=_eq)
    return m


def test_mcpmodel_interface():
    m = _square_ces_model()
    jm = JaxGTAPModel(m)
    assert jm.n == len(jm.seed())
    assert jm.lb.shape == (jm.n,) and jm.ub.shape == (jm.n,)
    assert np.all(np.isneginf(jm.lb)) and np.all(np.isposinf(jm.ub))


def test_F_matches_pyomo_on_square():
    from pyomo.environ import value
    m = _square_ces_model()
    jm = JaxGTAPModel(m)
    f_jax = jm.F(jm.seed())
    f_pyo = np.array([value(c.body) for c in jm._cons_order])
    assert np.max(np.abs(f_jax - f_pyo)) < 1e-10


def test_jacobian_matches_pyomo_nlp_on_square():
    """sparsejac Jacobian values == PyomoNLP.evaluate_jacobian_eq on the square system."""
    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
    from pyomo.environ import Objective
    m = _square_ces_model()
    jm = JaxGTAPModel(m)
    z = jm.seed()
    Jj = jm.jacobian(z).tocsr()
    m._obj = Objective(expr=0.0)
    nlp = PyomoNLP(m)
    Jp = nlp.evaluate_jacobian_eq().tocsr()
    assert Jj.shape == Jp.shape
    D = (Jj - Jp)
    assert abs(D).max() < 1e-7, f"max jacobian diff {abs(D).max():.2e}"


def test_jacobian_is_differentiated_not_probed():
    """The Jacobian must actually depend on z (autodiff), not be a constant pattern.
    Use a model where df/dy != 1 by making the free var enter nonlinearly."""
    from pyomo.environ import ConcreteModel, Var, Constraint
    m = ConcreteModel(); m.I = range(5)
    m.y = Var(m.I, initialize=2.0)
    m.eq = Constraint(m.I, rule=lambda m, i: m.y[i] ** 2 == 9.0)  # df/dy = 2y
    jm = JaxGTAPModel(m)
    J1 = jm.jacobian(np.full(5, 2.0)).tocsr()
    J2 = jm.jacobian(np.full(5, 3.0)).tocsr()
    # df/dy = 2y -> diag 4 at y=2, diag 6 at y=3
    assert abs(J1.diagonal() - 4.0).max() < 1e-7
    assert abs(J2.diagonal() - 6.0).max() < 1e-7
