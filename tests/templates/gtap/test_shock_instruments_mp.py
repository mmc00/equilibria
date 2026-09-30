"""Instrumentos en el modelo de 3 periodos: fijos en todos, y el driver no los toca.

La traza de 2026-09-29 (spec ShockBlock) mostro que freeze_inactive_periods
libera toda celda del periodo activo y _seed_period_from_prior la re-siembra con
el valor del check: un xft fijado a mano volvio de 3,5791 a 3,2537 con code=1.
Sin solver: nus333.
"""

import pytest


@pytest.fixture(scope="module")
def built():
    from equilibria._local_refs import nus333_dir

    har = nus333_dir()
    if not (har / "basedata.har").exists():
        pytest.skip(f"nus333 no disponible en {har}")
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    p = GTAPParameters()
    p.load_from_har(
        basedata_path=har / "basedata.har",
        sets_path=har / "sets.har",
        default_path=har / "default.prm",
        baserate_path=har / "baserate.har",
    )
    gc = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        numeraire="pnum",
    )
    m, _ = build_block_model(p, p.sets, gc, "ROW")
    return m


def test_registro_y_fijado_antes_del_cache(built):
    from equilibria.blocks.gtap.shock import SHOCK_INSTRUMENTS

    m = built
    assert m._exogenous_instruments == frozenset(SHOCK_INSTRUMENTS)
    for name in SHOCK_INSTRUMENTS:
        for idx, vd in getattr(m, name).items():
            assert vd.fixed, (name, idx)


@pytest.mark.parametrize("period", ["base", "check", "shock"])
def test_freeze_no_libera_instrumentos_en_ningun_periodo(built, period):
    from pyomo.environ import value

    from equilibria.templates.gtap.gtap_multiperiod_driver import (
        freeze_inactive_periods,
    )

    m = built
    idx = ("USA", "CAPITAL", period)
    x0 = float(value(m.aft[idx]))
    m.aft[idx].fix(x0 * 1.1)
    try:
        freeze_inactive_periods(m, period)
        assert m.aft[idx].fixed
        assert float(value(m.aft[idx])) == pytest.approx(x0 * 1.1)
    finally:
        m.aft[idx].fix(x0)


def test_seed_from_prior_no_pisa_un_instrumento(built):
    """Aun liberado (lo que hacia freeze antes), el re-sembrado no lo toca."""
    from pyomo.environ import value

    from equilibria.templates.gtap.gtap_multiperiod_driver import (
        _seed_period_from_prior,
    )

    m = built
    idx = ("USA", "CAPITAL", "shock")
    x0 = float(value(m.aft[idx]))
    m.aft[idx].unfix()
    m.aft[idx].set_value(x0 * 1.1)
    try:
        _seed_period_from_prior(m, "check", "shock")
        assert float(value(m.aft[idx])) == pytest.approx(x0 * 1.1)
    finally:
        m.aft[idx].fix(x0)
