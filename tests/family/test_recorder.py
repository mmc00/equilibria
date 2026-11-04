"""RecordingModel: block rules run against arrays and leave a small expression tree."""

import numpy as np
import pytest

pytest.importorskip("jax")

from pyomo.environ import value  # noqa: E402

from equilibria.family.recorder import (  # noqa: E402
    Node,
    RecordingModel,
    residual_terms,
)


def _rec():
    sets = {"r": ["A", "B"], "i": ["x", "y", "z"]}
    return RecordingModel(
        sets=sets,
        params={
            "share": (("r", "i"), np.array([[0.2, 0.0, 0.8], [0.5, 0.5, 0.0]])),
            "sigma": (("r",), np.array([0.5, 2.0])),
            "scale": ((), np.array(3.0)),
        },
        variables={
            "p": (("r",), np.array([1.1, 0.9])),
            "q": (("r", "i"), np.arange(1.0, 7.0).reshape(2, 3)),
            "t": ((), np.array(4.0)),
        },
    )


def _ces(m, r):
    e = 1.0 - value(m.sigma[r])
    return m.p[r] ** e == sum(
        m.share[r, i] * m.q[r, i] ** e for i in m.i if value(m.share[r, i]) > 0
    )


def test_sets_params_and_vars_look_like_pyomo():
    m = _rec()
    assert list(m.i) == ["x", "y", "z"]
    assert m.share["A", "z"] == 0.8 and m.sigma["B"] == 2.0 and m.scale == 3.0
    assert isinstance(m.q["B", "y"], Node) and isinstance(m.t, Node)
    assert value(m.q["B", "y"]) == 5.0  # build-time level, like Pyomo
    assert hasattr(m, "share") and not hasattr(m, "nope")


def test_residual_is_lhs_minus_rhs_as_signed_terms():
    m = _rec()
    res = _ces(m, "A")
    terms = residual_terms(res)
    # p**0.5 (+), 0.2*q_x**0.5 (-), 0.8*q_z**0.5 (-); q_y skipped by the rule's mask
    assert [s for s, _ in terms] == [1.0, -1.0, -1.0]
    assert res.evaluate() == pytest.approx(1.1**0.5 - 0.2 * 1.0**0.5 - 0.8 * 3.0**0.5)
    leaves = [t.var_leaves() for _, t in terms]
    assert leaves == [[("p", 0)], [("q", 0)], [("q", 2)]]


def test_long_sum_is_flat_and_linear_time():
    m = _rec()
    n = 50_000
    total = sum(m.t * float(k + 1) for k in range(n))
    assert total.op == "add" and len(total.args) == n


def test_zero_coefficients_vanish_like_the_nl_writer():
    m = _rec()
    res = (0.0 * m.p["A"] + m.q["A", "x"] * m.p["B"] * 0.0 + m.t**0.0 + m.p["B"]) == 0.0
    terms = residual_terms(res)
    consts = [t for _, t in terms if not isinstance(t, Node)]
    leaves = [t.var_leaves() for _, t in terms if isinstance(t, Node)]
    assert consts == [1.0] and leaves == [[("p", 1)]]


def test_cancelling_linear_terms_stay_structural():
    m = _rec()
    terms = residual_terms(m.t - m.t == m.p["A"])
    assert sum(len(t.var_leaves()) for _, t in terms if isinstance(t, Node)) == 3


def _toy_system():
    from equilibria.family.layout import VarLayout
    from equilibria.family.recorder import compile_family

    m = _rec()
    records = [(("A",), _ces(m, "A")), (("B",), _ces(m, "B"))]
    values = {v: np.asarray(m._vars[v][1], float) for v in m.var_domains}
    free = {v: np.ones(a.shape, bool) for v, a in values.items()}
    layout = VarLayout.from_state(m.sets, m.var_domains, values, free)
    from equilibria.family.core import FamilySystem

    fam = compile_family("eq_ces", records, layout)
    return records, layout, fam, FamilySystem.from_families([fam], layout)


def test_recorded_family_matches_closed_form():
    records, layout, fam, system = _toy_system()
    z = layout.free_values
    np.testing.assert_allclose(
        system.F(z), [res.evaluate() for _, res in records], rtol=1e-14
    )
    J = system.jac(z).toarray()
    names = layout.col_names
    col = {n: k for k, n in enumerate(names)}
    # Row A: e = 0.5, shares x=0.2, z=0.8 ; Row B: e = -1, shares x=0.5, y=0.5
    p, q = np.array([1.1, 0.9]), np.arange(1.0, 7.0).reshape(2, 3)
    want = np.zeros_like(J)
    want[0, col["p[A]"]] = 0.5 * p[0] ** -0.5
    want[0, col["q[A,x]"]] = -0.2 * 0.5 * q[0, 0] ** -0.5
    want[0, col["q[A,z]"]] = -0.8 * 0.5 * q[0, 2] ** -0.5
    want[1, col["p[B]"]] = -1.0 * p[1] ** -2.0
    want[1, col["q[B,x]"]] = -0.5 * -1.0 * q[1, 0] ** -2.0
    want[1, col["q[B,y]"]] = -0.5 * -1.0 * q[1, 1] ** -2.0
    np.testing.assert_allclose(J, want, rtol=1e-14)
    assert system.row_names == ["eq_ces[A]", "eq_ces[B]"]


def test_group_count_is_distinct_term_shapes():
    _, _, fam, _ = _toy_system()
    # p**e and share*q**e: two shapes, whatever the number of rows or terms per row
    assert len(fam.groups) == 2
