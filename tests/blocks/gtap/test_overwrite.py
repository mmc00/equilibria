"""``@overwrite``: modificar un bloque GTAP desde el notebook de un ejercicio.

Un notebook (un ejercicio) registra hooks sobre la CLASE del bloque; todo lo que
construye bloques los ve: ``build_block_model``, el SP de referencia del driver y la
calibracion. Primer uso: TBL65B (desempleo) libera ``aft[USA,LABOR]`` y agrega la fila
del salario real.

Spec: dev-tools/equilibria-tools/plans/superpowers/specs/2026-10-01-overwrite-bloques-design.md
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from tests.templates.gtap._desempleo import (
    closure,
    nus333_params,
    register_desempleo_hooks,
)

from equilibria.blocks.gtap import ClosureBlock, ShockBlock, overwrite


@pytest.fixture(autouse=True)
def _limpio():
    yield
    overwrite.clear()


@pytest.fixture
def mp_desempleo():
    """Modelo multiperiodo nus333 con los hooks de TBL65B (sin resolver)."""
    from equilibria.templates.gtap.gtap_block_model import build_block_model

    p = nus333_params()
    register_desempleo_hooks()
    m, _ = cast(Any, build_block_model(p, p.sets, closure(), "ROW"))
    return m


# --------------------------------------------------------------------------- API


def test_overwrite_registra_en_la_clase():
    @overwrite(ShockBlock)
    def uno(b):
        pass

    assert overwrite.registered(ShockBlock) == ["uno"]


def test_reregistrar_la_misma_funcion_no_duplica():
    """Re-ejecutar la celda del notebook reemplaza el hook, no lo apila."""
    for _ in range(2):

        @overwrite(ShockBlock)
        def uno(b):
            pass

    assert overwrite.registered(ShockBlock) == ["uno"]


def test_clear_limpia_todo():
    @overwrite(ShockBlock)
    def uno(b):
        pass

    @overwrite(ClosureBlock)
    def dos(b):
        pass

    assert overwrite.active()
    overwrite.clear()
    assert not overwrite.active()
    assert overwrite.registered(ShockBlock) == []


def test_endogenous_de_algo_que_no_es_instrumento_falla():
    from equilibria.templates.gtap.gtap_block_model import build_block_single_period

    p = nus333_params()

    @overwrite(ShockBlock)
    def mal(b):
        b.endogenous("pft", ("USA", "LABOR"))

    with pytest.raises(ValueError, match="pft"):
        build_block_single_period(p, p.sets, closure(), "ROW")


def test_cache_de_modelos_se_salta_con_hooks():
    """El cache se indexa por codigo y datos; un hook del notebook no esta en la clave."""
    from equilibria.blocks.gtap import model_cache

    p = nus333_params()
    register_desempleo_hooks()
    assert model_cache.cache_key(p, closure(), "ROW", False) is None


def test_monolito_con_hooks_falla(monkeypatch):
    """El monolito no ve los hooks: con hooks activos pedirlo es un error, no un
    silencio (el SP de referencia volveria a fijar aft[USA,LABOR])."""
    from equilibria.templates.gtap.gtap_multiperiod_driver import _build_sp_reference

    p = nus333_params()
    register_desempleo_hooks()
    monkeypatch.setenv("EQUILIBRIA_GTAP_REF_MODEL", "monolith")
    with pytest.raises(RuntimeError, match="overwrite"):
        _build_sp_reference(p.sets, p, closure(), "ROW")


# ------------------------------------------------------------------------ modelo


def test_sp_de_referencia_del_driver_con_hooks():
    """El SP que lee el driver tiene la celda libre y la fila."""
    from equilibria.templates.gtap.gtap_multiperiod_driver import _build_sp_reference

    p = nus333_params()
    register_desempleo_hooks()
    sp = cast(Any, _build_sp_reference(p.sets, p, closure(), "ROW"))

    assert not sp.aft["USA", "LABOR"].fixed
    assert sp.aft["ROW", "LABOR"].fixed
    assert sp.aft["USA", "CAPITAL"].fixed
    assert ("USA", "LABOR") in sp.eq_wreal
    assert len(sp.eq_wreal) == 1
    assert sp._endogenous_instrument_cells == {"aft": frozenset({("USA", "LABOR")})}


def test_multiperiodo_con_hooks(mp_desempleo):
    from pyomo.environ import value

    m = mp_desempleo
    for t in ("base", "check", "shock"):
        assert not m.aft["USA", "LABOR", t].fixed, t
        assert m.aft["ROW", "LABOR", t].fixed, t
        assert ("USA", "LABOR", t) in m.eq_wreal, t
    assert m._endogenous_instrument_cells == {"aft": frozenset({("USA", "LABOR")})}
    # En la base la fila se cumple: el Tornqvist vale 1 y pft es su valor de base.
    row = m.eq_wreal["USA", "LABOR", "base"]
    assert abs(value(row.body) - value(row.upper)) < 1e-12


def test_sin_hooks_no_cambia_nada():
    """Sin hooks: aft fijo, sin fila nueva. (Que el modelo sin hooks es identico al de
    main lo fija test_no_shock_identical_to_main.)"""
    from equilibria.templates.gtap.gtap_block_model import build_block_single_period

    p = nus333_params()
    sp = cast(Any, build_block_single_period(p, p.sets, closure(), "ROW"))
    assert sp.aft["USA", "LABOR"].fixed
    assert not hasattr(sp, "eq_wreal")
    assert sp._endogenous_instrument_cells == {}


def test_fix_instrument_shock_sobre_celda_endogena_falla(mp_desempleo):
    from equilibria.templates.gtap.instruments import fix_instrument_shock

    with pytest.raises(ValueError, match="endogenous"):
        fix_instrument_shock(mp_desempleo, "aft", ("USA", "LABOR"), factor=1.1)


def test_is_exogenous_por_celda(mp_desempleo):
    from equilibria.templates.gtap.instruments import is_exogenous

    m = mp_desempleo
    assert not is_exogenous(m, "aft", ("USA", "LABOR", "shock"))
    assert is_exogenous(m, "aft", ("ROW", "LABOR", "shock"))
    assert is_exogenous(m, "prdtx_rai", ("USA", "MFG", "MFG", "shock"))
    assert not is_exogenous(m, "pft", ("USA", "LABOR", "shock"))


# ------------------------------------------------------------------------ driver


def test_freeze_inactive_periods_libera_la_celda_endogena(mp_desempleo):
    """Periodo activo: la celda endogena se libera como cualquier variable; en los
    inactivos se fija. Las celdas exogenas de aft quedan fijas en los 3."""
    from equilibria.templates.gtap.gtap_multiperiod_driver import (
        freeze_inactive_periods,
    )

    m = mp_desempleo
    m.aft["USA", "LABOR", "shock"].fix()
    freeze_inactive_periods(m, "shock")
    assert not m.aft["USA", "LABOR", "shock"].fixed
    assert m.aft["USA", "LABOR", "check"].fixed
    for t in ("base", "check", "shock"):
        assert m.aft["ROW", "LABOR", t].fixed, t


def test_seed_desde_el_periodo_previo_incluye_la_celda_endogena(mp_desempleo):
    from equilibria.templates.gtap.gtap_multiperiod_driver import (
        _seed_period_from_prior,
    )

    m = mp_desempleo
    m.aft["USA", "LABOR", "check"].set_value(7.0)
    m.aft["ROW", "LABOR", "check"].set_value(7.0)
    row_shock = m.aft["ROW", "LABOR", "shock"].value
    _seed_period_from_prior(m, "check", "shock")
    assert m.aft["USA", "LABOR", "shock"].value == 7.0
    assert m.aft["ROW", "LABOR", "shock"].value == row_shock


def test_cotas_de_gams_sueltan_la_celda_endogena(mp_desempleo):
    """_apply_gams_bounds deja libre (Reals, sin cotas) a la celda endogena; a las
    exogenas no las toca."""
    from equilibria.templates.gtap.gtap_multiperiod_driver import _apply_gams_bounds

    m = mp_desempleo
    m.aft["USA", "LABOR", "shock"].setlb(0.0)
    m.aft["ROW", "LABOR", "shock"].setlb(0.0)
    _apply_gams_bounds(m, "shock")
    assert m.aft["USA", "LABOR", "shock"].lb is None
    assert m.aft["ROW", "LABOR", "shock"].lb == 0.0
