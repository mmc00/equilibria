"""``build_arrays``: the multiperiod arrays of any blocks model, here simple_open."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from equilibria.family.generic_arrays import build_arrays  # noqa: E402
from equilibria.templates.simple_open import SimpleOpenEconomy  # noqa: E402
from equilibria.templates.simple_open.template import INSTRUMENTS  # noqa: E402
from tests.family.test_family_e2e import _spies  # noqa: E402

PERIODS = ("base", "check", "shock")


@pytest.fixture(scope="module")
def built():
    model = SimpleOpenEconomy().create_model()
    return build_arrays(model, INSTRUMENTS, PERIODS)


def test_every_var_has_one_cell_per_period(built) -> None:
    mp, _, state = built
    assert set(mp.vars) == set(state.var_domains)
    for name, v in mp.vars.items():
        assert len(v.keys) == 3 * state.values[name].size, name
        assert set(v.period) == set(PERIODS), name


def test_instruments_fixed_and_exogenous(built) -> None:
    mp, _, _ = built
    assert mp.exogenous == frozenset(INSTRUMENTS)
    for name in INSTRUMENTS:
        assert mp.vars[name].fixed.all(), name
    assert not mp.vars["XV"].fixed.any()


def test_bounds_come_from_state(built) -> None:
    mp, _, state = built
    xv = mp.vars["XV"]
    for t in PERIODS:
        assert np.array_equal(xv.lb[xv.period == t], state.lower["XV"].ravel())
    assert (xv.lb == 1e-5).all()
    # a lower bound other than the floor is carried as is (TM is bounded at 0)
    assert (mp.vars["TM"].lb == 0.0).all()
    assert np.isinf(mp.vars["WALRAS"].lb).all()


def test_rows_one_per_period(built) -> None:
    mp, backend, _ = built
    n_sp = sum(len(r) for r in backend.records.values())
    assert set(mp.rows) == set(backend.records)
    for t in PERIODS:
        assert sum(int((r.period == t).sum()) for r in mp.rows.values()) == n_sp
    assert all(r.active.all() for r in mp.rows.values())


def test_no_pyomo_model_built() -> None:
    with _spies() as calls:
        build_arrays(SimpleOpenEconomy().create_model(), INSTRUMENTS, PERIODS)
    assert calls == {"ConcreteModel": 0, "PyomoNLP": 0, "identify_variables": 0}
