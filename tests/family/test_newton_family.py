"""``newton_family``: the recipe solver of a generic family model (simple_open)."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")

from equilibria.core.steps import freeze_other_periods, seed_from_prior  # noqa: E402
from equilibria.solver.newton_family import newton_family  # noqa: E402
from equilibria.templates.simple_open.steps import apply_shock  # noqa: E402
from tests.family.test_generic_periods import _ctx, _model  # noqa: E402


def _prepare(m, period: str, prior: str, shock=None) -> None:
    ctx = _ctx(m, period, prior)
    ctx.shock = dict(shock or {})
    freeze_other_periods(ctx)
    seed_from_prior(ctx)
    apply_shock(ctx)


def test_check_returns_the_base() -> None:
    m = _model()
    _prepare(m, "check", "base")
    assert newton_family()(m) == "optimal"
    for name, index in m.cells("base"):
        assert m.value(name, index, "check") == pytest.approx(
            m.value(name, index, "base"), rel=1e-9, abs=1e-9
        ), (name, index)


def test_shock_converges() -> None:
    m = _model()
    _prepare(m, "check", "base")
    newton_family()(m)
    _prepare(m, "shock", "check", {"tau_m": 0.0})
    assert newton_family()(m) == "optimal"
    assert m.residual("shock") < 1e-8
    assert m.last_solve is not None and m.last_solve["code"] == "optimal"
    assert all(
        m.value("tau_m", k, "shock") == 0.0 for n, k in m.cells("shock") if n == "tau_m"
    )


def test_not_square_raises() -> None:
    m = _model()
    _prepare(m, "check", "base")
    m.unfix("tau_d", (), "check")
    with pytest.raises(ValueError, match="not square"):
        newton_family()(m)
