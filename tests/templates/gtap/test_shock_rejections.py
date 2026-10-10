"""A shock that cannot be honoured is rejected up front.

``apply_shock`` rejects a bad cell or value when it writes the shock, before any
solve; the driver rejects a shock it cannot run (altertax, the tariff
continuation) before loading the solver. A run that would silently drop the shock
must fail in milliseconds, not after the base and check solves.
"""

from __future__ import annotations

import pytest
from tests.templates.gtap._nus333 import closure, nus333_params

from equilibria.templates.gtap import gtap_multiperiod_driver as driver
from equilibria.templates.gtap.shocks import apply_shock


def _explode():
    raise AssertionError("validation must run before loading the solver")


@pytest.fixture
def no_solver(monkeypatch):
    monkeypatch.setattr(driver, "_load_run_gtap", _explode)


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


# ── el driver: un shock en el ShockBlock que no puede correr ────────────────


@pytest.fixture
def shocked(built):
    """aft[USA,CAPITAL] +10% en el ShockBlock; se restaura al salir."""
    from pyomo.environ import value

    m, p, gc = built
    idx = ("USA", "CAPITAL", "shock")
    before = float(value(m.aft[idx]))
    apply_shock(m, {"aft": {("USA", "CAPITAL"): 10.0}})
    yield m, p, gc
    m.aft[idx].fix(before)


def test_rejected_in_altertax_mode(no_solver, shocked):
    m, p, gc = shocked
    with pytest.raises(ValueError, match="mode='gtap'"):
        driver.solve_multiperiod(m, p, gc, mode="altertax")


def test_rejected_with_tariff_continuation(no_solver, monkeypatch, shocked):
    m, p, gc = shocked
    monkeypatch.setenv("EQUILIBRIA_GTAP_SHOCK_CONTINUATION", "0.5,1.0")
    with pytest.raises(ValueError, match="CONTINUATION"):
        driver.solve_multiperiod(m, p, gc, mode="gtap")
