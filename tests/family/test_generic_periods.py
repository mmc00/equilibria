"""``GenericFamilyPeriodModel``: simple_open's periods on the family arrays, with a
residual evaluated from the recorded rows."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")

from equilibria.core.recipe import KindPlan, Period, Recipe, StepContext  # noqa: E402
from equilibria.core.steps import freeze_other_periods, seed_from_prior  # noqa: E402
from equilibria.family.generic_arrays import build_arrays  # noqa: E402
from equilibria.family.generic_periods import GenericFamilyPeriodModel  # noqa: E402
from equilibria.templates.simple_open import SimpleOpenEconomy  # noqa: E402
from equilibria.templates.simple_open.template import INSTRUMENTS  # noqa: E402

PERIODS = ("base", "check", "shock")


def _model() -> GenericFamilyPeriodModel:
    mp, backend, state = build_arrays(
        SimpleOpenEconomy().create_model(), INSTRUMENTS, PERIODS
    )
    return GenericFamilyPeriodModel(mp, backend, state, PERIODS)


def _ctx(model, period: str, prior: str | None) -> StepContext:
    p = {t: Period(t, t) for t in PERIODS}
    return StepContext(
        model=model,
        template=None,
        recipe=Recipe(periods=tuple(p.values()), plans={t: KindPlan() for t in p}),
        period=p[period],
        prior=p[prior] if prior else None,
    )


def test_base_residual_is_zero_at_calibration() -> None:
    assert _model().residual("base") < 1e-9


def test_residual_sees_a_moved_cell() -> None:
    m = _model()
    free = next(c for c in m.cells("check") if c not in m.exogenous())
    name, index = free
    m.set_value(name, index, "check", 2.0 * m.value(name, index, "check"))
    assert m.residual("check") > 1e-3
    assert m.residual("base") < 1e-9


def test_protocol_steps_run() -> None:
    m = _model()
    freeze_other_periods(_ctx(m, "check", "base"))
    seed_from_prior(_ctx(m, "check", "base"))
    assert all(m.is_fixed(n, k, "shock") for n, k in m.cells("shock"))
    assert any(not m.is_fixed(n, k, "check") for n, k in m.cells("check"))
    assert m.residual("check") < 1e-9


def test_residual_ignores_inactive_rows() -> None:
    m = _model()
    free = next(c for c in m.cells("check") if c not in m.exogenous())
    name, index = free
    m.set_value(name, index, "check", 2.0 * m.value(name, index, "check"))
    assert m.residual("check") > 1e-3
    for fam in m.row_names():
        m.set_rows_active(fam, "check", m.row_cells(fam, "check"), False)
    assert m.residual("check") == 0.0
