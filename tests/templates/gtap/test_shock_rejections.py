"""A shock that cannot be honoured is rejected up front.

``apply_shock`` rejects a bad cell or value when it writes the shock, before any
solve. A run that would silently drop the shock must fail in milliseconds, not
after the base and check solves. (Altertax and the tariff continuation are no
longer rejected: change tariff-to-shock-block.)
"""

from __future__ import annotations

import pytest
from tests.templates.gtap._nus333 import closure, nus333_params

from equilibria.templates.gtap.shocks import apply_shock


@pytest.fixture(scope="module")
def built():
    from equilibria.templates.gtap.gtap_block_model import build_block_model

    p = nus333_params()
    gc = closure("capFix")
    m, _ = build_block_model(p, p.sets, gc, "ROW")
    return m, p, gc


# ── apply_shock: la celda y el valor ────────────────────────────────────────


NONPOS = [("lambdava", ("USA", "SER")), ("aft", ("USA", "CAPITAL"))]


@pytest.mark.parametrize("name,cell", NONPOS)
@pytest.mark.parametrize("pct", [-100.0, -200.0])
def test_rejects_a_change_of_minus_100_or_less(built, name, cell, pct):
    m, _, _ = built
    with pytest.raises(ValueError, match=r"USA.*must be > -100%"):
        apply_shock(m, {name: {cell: pct}})


@pytest.mark.parametrize("name,cell", NONPOS)
@pytest.mark.parametrize("level", [0.0, -1.0])
def test_rejects_a_non_positive_level(built, name, cell, level):
    m, _, _ = built
    with pytest.raises(ValueError, match=r"USA.*must be > 0"):
        apply_shock(m, {name: {cell: level}}, levels=True)


@pytest.mark.parametrize(
    ("name", "cell", "hint"),
    [
        # TBL66/TBL77 escriben el factor en minuscula; el modelo lo tiene en mayuscula.
        ("aft", ("USA", "labor"), "LABOR"),
        ("aft", ("USA", "NOPE"), "CAPITAL"),
        ("lambdava", ("USA", "ser"), "SER"),
    ],
)
def test_rejects_an_unknown_index_naming_the_valid_ones(built, name, cell, hint):
    m, _, _ = built
    with pytest.raises(ValueError, match=hint):
        apply_shock(m, {name: {cell: 10.0}})


def test_rejects_an_unregistered_instrument(built):
    m, _, _ = built
    with pytest.raises(ValueError, match="not a registered instrument"):
        apply_shock(m, {"tms": {("ROW", "MFG", "USA"): 10.0}})


def test_rejects_a_fixed_endowment_in_the_shock_period(built):
    """Con xft fijada, el solver desactiva eq_xfteq y el shock se pierde con code=1."""
    m, _, _ = built
    idx = ("USA", "CAPITAL", "shock")
    m.xft[idx].fix()
    try:
        with pytest.raises(ValueError, match="fixed"):
            apply_shock(m, {"aft": {("USA", "CAPITAL"): 10.0}})
    finally:
        m.xft[idx].unfix()


# ── el driver: un modelo sin ShockBlock ─────────────────────────────────────


def test_a_model_without_shockblock_is_rejected_before_the_solver(monkeypatch):
    """El monolito (GTAPMultiPeriodModel) no tiene ShockBlock: no hay donde
    escribir el shock, ni el arancel por defecto. Error claro, sin solver."""
    from tests.templates.gtap.test_multiperiod_sets import _load_3x3_params

    from equilibria.templates.gtap import gtap_multiperiod_driver as driver
    from equilibria.templates.gtap.gtap_model_multiperiod import GTAPMultiPeriodModel

    def _explode():
        raise AssertionError("must be rejected before loading the solver")

    monkeypatch.setattr(driver, "_load_run_gtap", _explode)
    p = _load_3x3_params()
    mp = GTAPMultiPeriodModel(p.sets, p, None, residual_region=list(p.sets.r)[-1])
    m = mp.build_sets()
    mp.build_vars(m)
    with pytest.raises(ValueError, match="ShockBlock.*GTAPBlockMultiPeriodModel"):
        driver.solve_multiperiod(m, p, None)
