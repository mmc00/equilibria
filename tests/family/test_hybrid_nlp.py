"""HybridNLP: family rows served to the Newton driver, verified against PyomoNLP.

The multiperiod model is frozen to one period the way the driver does it
(``freeze_inactive_periods``); the wrapper must give the same F and J as the plain
PyomoNLP, serving every family it can verify and leaving only the cross-period rows
to Pyomo.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("pyomo.contrib.pynumero.interfaces.pyomo_nlp")

from tests.family._data import load_params  # noqa: E402
from tests.family._oracle import comp_stat_closure  # noqa: E402

DATASET = "gtap7_3x3"
CROSS_PERIOD = {
    "eq_pabs",
    "eq_pfact",
    "eq_pwfact",
    "eq_rgdpmp",
    "eq_pgdpmp",
    "eq_chisave",
}


def _frozen_nlp(period: str):
    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
    from pyomo.environ import Objective

    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_multiperiod_driver import (
        freeze_inactive_periods,
    )

    p = load_params(DATASET)
    rr = list(p.sets.r)[-1]
    m, _ = build_block_model(p, p.sets, comp_stat_closure(), rr)
    freeze_inactive_periods(m, period)
    m._family_test_obj = Objective(expr=0.0)
    return m, PyomoNLP(m)


def _backend():
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.templates.gtap.gtap_block_model import compose_block_state

    p = load_params(DATASET)
    model, state = compose_block_state(
        p, p.sets, comp_stat_closure(), list(p.sets.r)[-1]
    )
    backend = FamilyBackend()
    backend.build(model)
    return backend, state


@pytest.fixture(scope="module")
def setup():
    from equilibria.family.hybrid import HybridNLP

    backend, state = _backend()
    m, nlp = _frozen_nlp("check")
    return m, nlp, HybridNLP(nlp, backend, state), backend


def _points(nlp):
    x0 = nlp.get_primals().copy()
    rng = np.random.default_rng(0)
    return {"seed": x0, "perturbed": x0 * (1 + 1e-3 * rng.standard_normal(x0.size))}


@pytest.mark.parametrize("point", ["seed", "perturbed"])
def test_hybrid_matches_pyomo_at_seed_and_perturbed(setup, point):
    _, nlp, h, _ = setup
    x = _points(nlp)[point]
    h.set_primals(x)
    F_h = h.evaluate_eq_constraints()
    J_h = h.evaluate_jacobian_eq().tocsr()
    nlp.set_primals(x)
    F_p = nlp.evaluate_eq_constraints()
    J_p = nlp.evaluate_jacobian_eq().tocsr()

    np.testing.assert_allclose(F_h, F_p, rtol=1e-12, atol=1e-12)
    assert J_h.shape == J_p.shape
    J_h.sort_indices()
    J_p.sort_indices()
    np.testing.assert_array_equal(J_h.indptr, J_p.indptr)
    np.testing.assert_array_equal(J_h.indices, J_p.indices)
    np.testing.assert_allclose(J_h.data, J_p.data, rtol=1e-12, atol=1e-12)


def test_fallback_is_only_cross_period(setup):
    _, _, h, backend = setup
    expected = {
        n for n in backend.records if n in CROSS_PERIOD or n.startswith("eq_mq_")
    }
    assert h.period == "check"
    assert set(h.fallback) <= expected
    # The intra-period Fisher rows are SP-only: the multiperiod model drops them.
    assert set(h.unmapped) <= {
        f"eq_mf{a}_{b}" for a in "rw" for b in ("bs", "sb", "ss")
    }
    assert set(h.served) == set(backend.records) - set(h.fallback) - set(h.unmapped)
    assert h.family_rows.size > 0.9 * h.n_eq_constraints()


def test_missing_pyomo_column_raises():
    from equilibria.family.hybrid import HybridNLP

    backend, state = _backend()
    m, nlp = _frozen_nlp("check")
    var = next(v for v, d in backend.var_domains.items() if d)
    cell = next(k for k in getattr(m, var) if k[-1] == "check")
    del getattr(m, var)[cell]
    with pytest.raises(KeyError, match=var):
        HybridNLP(nlp, backend, state)
