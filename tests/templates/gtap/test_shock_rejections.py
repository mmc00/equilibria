"""The shock-period-only shocks (``lambdava_shock``, ``qe_shock``) are rejected up
front where they cannot be honoured.

Every check must fire BEFORE the solver is loaded: a run that would silently drop
the shock (altertax), walk the wrong shock (the tariff continuation), or hit an
index the model does not have must fail in milliseconds, not after the base and
check solves.
"""

from __future__ import annotations

import pytest
from tests.templates.gtap._nus333 import closure, nus333_params

from equilibria.templates.gtap import gtap_multiperiod_driver as driver

SHOCKS = {
    "lambdava_shock": ("USA", "SER"),
    "qe_shock": ("USA", "CAPITAL"),
}


def _solve(m, p, gc, kind, shock, mode="gtap"):
    """solve_multiperiod con el shock de tipo ``kind`` (kwargs explicitos para ty)."""
    if kind == "lambdava_shock":
        return driver.solve_multiperiod(m, p, gc, mode=mode, lambdava_shock=shock)
    return driver.solve_multiperiod(m, p, gc, mode=mode, qe_shock=shock)


def _explode():
    raise AssertionError("validation must run before loading the solver")


@pytest.fixture
def no_solver(monkeypatch):
    monkeypatch.setattr(driver, "_load_run_gtap", _explode)


@pytest.mark.parametrize("kind", SHOCKS)
def test_rejected_in_altertax_mode(no_solver, kind):
    with pytest.raises(ValueError, match="mode='gtap'"):
        _solve(None, None, None, kind, {SHOCKS[kind]: 1.1}, mode="altertax")


@pytest.mark.parametrize("kind", SHOCKS)
def test_rejected_with_tariff_continuation(no_solver, monkeypatch, kind):
    monkeypatch.setenv("EQUILIBRIA_GTAP_SHOCK_CONTINUATION", "0.5,1.0")
    with pytest.raises(ValueError, match="CONTINUATION"):
        _solve(None, None, None, kind, {SHOCKS[kind]: 1.1})


@pytest.mark.parametrize("kind", SHOCKS)
@pytest.mark.parametrize("factor", [0.0, -1.0])
def test_rejects_a_non_positive_factor(no_solver, kind, factor):
    with pytest.raises(ValueError, match=r"USA.*must be > 0"):
        _solve(None, None, None, kind, {SHOCKS[kind]: factor})


def test_rejects_combining_shocks(no_solver):
    """Ningun ejercicio del libro combina los dos; no hay test que lo respalde."""
    with pytest.raises(ValueError, match="one shock"):
        driver.solve_multiperiod(
            None,
            None,
            None,
            mode="gtap",
            lambdava_shock={SHOCKS["lambdava_shock"]: 1.1},
            qe_shock={SHOCKS["qe_shock"]: 1.1},
        )


# ── Index checks: need a built model (nus333, no solve) ─────────────────────


@pytest.fixture(scope="module")
def built():
    from equilibria.templates.gtap.gtap_block_model import build_block_model

    p = nus333_params()
    gc = closure("capFix")
    m, _ = build_block_model(p, p.sets, gc, "ROW")
    return m, p, gc


@pytest.mark.parametrize(
    ("kind", "key", "hint"),
    [
        # TBL66/TBL77 escriben el factor en minuscula; el modelo lo tiene en mayuscula.
        ("qe_shock", ("USA", "labor"), "LABOR"),
        ("qe_shock", ("USA", "NOPE"), "CAPITAL"),
        ("lambdava_shock", ("USA", "ser"), "SER"),
    ],
)
def test_rejects_an_unknown_index_naming_the_valid_ones(
    no_solver, built, kind, key, hint
):
    m, p, gc = built
    with pytest.raises(ValueError, match=hint):
        _solve(m, p, gc, kind, {key: 1.1})


def test_rejects_a_fixed_endowment_in_the_shock_period(no_solver, built):
    """Con xft fijada, el solver desactiva eq_xfteq y el shock se pierde con code=1."""
    m, p, gc = built
    idx = ("USA", "CAPITAL", "shock")
    m.xft[idx].fix()
    try:
        with pytest.raises(ValueError, match="fixed"):
            driver.solve_multiperiod(
                m, p, gc, mode="gtap", qe_shock={("USA", "CAPITAL"): 1.1}
            )
    finally:
        m.xft[idx].unfix()


# ── Shock aplicado directo con apply_shock (sin kwargs) ─────────────────────


@pytest.fixture
def direct_aft(built):
    """aft[USA,CAPITAL,shock] x1.1 via fix_instrument_shock; se restaura al salir."""
    from pyomo.environ import value

    from equilibria.templates.gtap.instruments import fix_instrument_shock

    m, p, gc = built
    idx = ("USA", "CAPITAL", "shock")
    before = float(value(m.aft[idx]))
    fix_instrument_shock(m, "aft", ("USA", "CAPITAL"), factor=1.1)
    yield m, p, gc
    m.aft[idx].fix(before)


def test_direct_shock_plus_kwarg_is_rejected(no_solver, direct_aft):
    m, p, gc = direct_aft
    with pytest.raises(ValueError, match="one shock"):
        driver.solve_multiperiod(
            m, p, gc, mode="gtap", lambdava_shock={("USA", "SER"): 1.1}
        )


def test_direct_shock_rejected_in_altertax_mode(no_solver, direct_aft):
    m, p, gc = direct_aft
    with pytest.raises(ValueError, match="mode='gtap'"):
        driver.solve_multiperiod(m, p, gc, mode="altertax")
