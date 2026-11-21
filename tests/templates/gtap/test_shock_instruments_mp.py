"""Instrumentos en el modelo de 3 periodos: fijos en todos, y la receta no los toca.

La traza de 2026-09-29 (spec ShockBlock) mostro que el congelado de periodos
liberaba toda celda del periodo activo y la siembra desde el periodo previo la
re-sembraba con el valor del check: un xft fijado a mano volvio de 3,5791 a
3,2537 con code=1. Aqui, los pasos del core (``steps.freeze_inactive``,
``steps.seed_from``, ``steps.copy_from_base``) sobre el modelo Pyomo. Sin
solver: nus333.
"""

import pytest
from tests.templates.gtap._core_step import run_step

from equilibria.sources import dataset_dir
from equilibria.templates.gtap.instruments import period_key


def test_registro_y_fijado_antes_del_cache(nus333_mp_model):
    from equilibria.blocks.gtap.shock import SHOCK_INSTRUMENTS

    m = nus333_mp_model
    assert m._exogenous_instruments == frozenset(SHOCK_INSTRUMENTS)
    for name in SHOCK_INSTRUMENTS:
        for idx, vd in getattr(m, name).items():
            assert vd.fixed, (name, idx)


@pytest.mark.parametrize("period", ["base", "check", "shock"])
def test_freeze_no_libera_instrumentos_en_ningun_periodo(nus333_mp_model, period):
    from pyomo.environ import value

    from equilibria.templates.gtap.steps import freeze_inactive

    m = nus333_mp_model
    idx = ("USA", "CAPITAL", period)
    x0 = float(value(m.aft[idx]))
    m.aft[idx].fix(x0 * 1.1)
    try:
        run_step(freeze_inactive, m, period)
        assert m.aft[idx].fixed
        assert float(value(m.aft[idx])) == pytest.approx(x0 * 1.1)
    finally:
        m.aft[idx].fix(x0)


def test_seed_from_prior_no_pisa_un_instrumento(nus333_mp_model):
    """Aun liberado (lo que hacia freeze antes), el re-sembrado no lo toca."""
    from pyomo.environ import value

    from equilibria.templates.gtap.steps import seed_from

    m = nus333_mp_model
    idx = ("USA", "CAPITAL", "shock")
    x0 = float(value(m.aft[idx]))
    m.aft[idx].unfix()
    m.aft[idx].set_value(x0 * 1.1)
    try:
        run_step(seed_from("check"), m, "shock")
        assert float(value(m.aft[idx])) == pytest.approx(x0 * 1.1)
    finally:
        m.aft[idx].fix(x0)


def test_copia_base_a_check_no_pisa_instrumentos(nus333_mp_model):
    """F3.5 (base_calibrated sin solve_check) copia base->check en TODA Var: tiene
    que saltar los instrumentos, como freeze y seed_from_prior."""
    from pyomo.environ import value

    from equilibria.templates.gtap.steps import copy_from_base

    m = nus333_mp_model
    ins = ("USA", "CAPITAL", "check")
    otra = next(k for k in m.xft if k[-1] == "check")
    antes_ins = float(value(m.aft[ins]))
    antes_otra = float(value(m.xft[otra]))
    m.aft[ins].fix(antes_ins * 1.1)
    m.xft[otra].set_value(antes_otra * 3.0)
    try:
        run_step(copy_from_base, m, "check")
        assert float(value(m.aft[ins])) == pytest.approx(antes_ins * 1.1)
        base = (*otra[:-1], "base")
        assert float(value(m.xft[otra])) == float(value(m.xft[base]))
    finally:
        m.aft[ins].fix(antes_ins)
        m.xft[otra].set_value(antes_otra)


def test_build_vars_fija_y_registra_sin_pasar_por_build_block_model():
    """Los gates arman el modelo paso a paso (build_sets/build_vars/...), sin
    build_block_model. Si solo este fijara los instrumentos, en los gates quedarian
    libres y _replicate_sp_fixing los fijaria con el valor del SP del shock: el
    arancel +10% entraria vivo en todas las ecuaciones (altertax 3x3: 100% -> 73,4%).
    """

    from equilibria.blocks.gtap.shock import SHOCK_INSTRUMENTS
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_block_model import GTAPBlockMultiPeriodModel
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    d = dataset_dir("gtap7_3x3")
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    gc = GTAPClosureConfig(
        name="altertax",
        closure_type="MCP",
        capital_mobility="mobile",
        fix_endowments=False,
        fix_taxes=True,
        fix_technology=True,
        if_sub=False,
        numeraire="pnum",
    )
    mp = GTAPBlockMultiPeriodModel(p.sets, p, gc, residual_region=list(p.sets.r)[-1])
    m = mp.build_sets()
    mp.build_vars(m)

    assert m._exogenous_instruments == frozenset(SHOCK_INSTRUMENTS)
    for name in SHOCK_INSTRUMENTS:
        var = getattr(m, name)
        libres = [k for k, vd in var.items() if not vd.fixed]
        assert not libres, (name, libres[:3])
        for k, vd in var.items():
            kt = k if isinstance(k, tuple) else (k,)  # pnum: indexado solo por t
            if kt[-1] != "base":
                assert vd.value == var[period_key(kt[:-1], "base")].value, (name, k)
