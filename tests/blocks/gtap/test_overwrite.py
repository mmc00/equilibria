"""``@overwrite``: modificar un bloque GTAP desde el notebook de un ejercicio.

Un notebook (un ejercicio) registra hooks sobre la CLASE del bloque; todo lo que
construye bloques los ve: ``build_block_model``, el SP de referencia del driver y la
calibracion. Primer uso: TBL65B (desempleo) libera ``aft[USA,LABOR]`` y agrega la fila
del salario real.

Spec: dev-tools/equilibria-tools/plans/superpowers/specs/2026-10-01-overwrite-bloques-design.md
"""

from __future__ import annotations

import importlib
from typing import Any, cast

import pytest


def _ow() -> Any:
    return cast(Any, importlib.import_module("equilibria.blocks.gtap.overwrite"))


@pytest.fixture(autouse=True)
def _limpio():
    yield
    try:
        _ow().overwrite.clear()
    except ModuleNotFoundError:
        pass


def _nus333_params():
    from equilibria._local_refs import nus333_dir

    har = nus333_dir()
    if not (har / "basedata.har").exists():
        pytest.skip(f"nus333 no disponible en {har}")
    from equilibria.templates.gtap import GTAPParameters

    p = GTAPParameters()
    p.load_from_har(
        basedata_path=har / "basedata.har",
        sets_path=har / "sets.har",
        default_path=har / "default.prm",
        baserate_path=har / "baserate.har",
    )
    return p


def _closure():
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        savf_flag="capFlex",
        numeraire="pnum",
    )


def _hooks_desempleo():
    """Los dos hooks del notebook de TBL65B."""
    from pyomo.environ import value

    from equilibria.blocks.gtap import ClosureBlock, ShockBlock

    ow = _ow()

    @ow.overwrite(ShockBlock)
    def desempleo(b):
        b.endogeno("aft", ("USA", "LABOR"))

    @ow.overwrite(ClosureBlock)
    def salario_real(b):
        b.ecuacion(
            "eq_wreal",
            ("USA", "LABOR"),
            lambda m, r, f: m.pft[r, f]
            == value(m.pft[r, f]) * ow.ppriv_tornqvist(m, r),
        )


# --------------------------------------------------------------------------- API


def test_overwrite_registra_en_la_clase():
    from equilibria.blocks.gtap import ShockBlock

    ow = _ow()

    @ow.overwrite(ShockBlock)
    def uno(b):
        pass

    assert ow.overwrite.registered(ShockBlock) == ["uno"]


def test_reregistrar_la_misma_funcion_no_duplica():
    """Re-ejecutar la celda del notebook reemplaza el hook, no lo apila."""
    from equilibria.blocks.gtap import ShockBlock

    ow = _ow()
    for _ in range(2):

        @ow.overwrite(ShockBlock)
        def uno(b):
            pass

    assert ow.overwrite.registered(ShockBlock) == ["uno"]


def test_clear_limpia_todo():
    from equilibria.blocks.gtap import ClosureBlock, ShockBlock

    ow = _ow()

    @ow.overwrite(ShockBlock)
    def uno(b):
        pass

    @ow.overwrite(ClosureBlock)
    def dos(b):
        pass

    assert ow.overwrite.active()
    ow.overwrite.clear()
    assert not ow.overwrite.active()
    assert ow.overwrite.registered(ShockBlock) == []


def test_endogeno_de_algo_que_no_es_instrumento_falla():
    from equilibria.blocks.gtap import ShockBlock
    from equilibria.templates.gtap.gtap_block_model import build_block_single_period

    p = _nus333_params()
    ow = _ow()

    @ow.overwrite(ShockBlock)
    def mal(b):
        b.endogeno("pft", ("USA", "LABOR"))

    with pytest.raises(ValueError, match="pft"):
        build_block_single_period(p, p.sets, _closure(), "ROW")


def test_cache_de_modelos_se_salta_con_hooks():
    """El cache se indexa por codigo y datos; un hook del notebook no esta en la clave."""
    from equilibria.blocks.gtap import ShockBlock, model_cache

    p = _nus333_params()
    ow = _ow()

    @ow.overwrite(ShockBlock)
    def desempleo(b):
        b.endogeno("aft", ("USA", "LABOR"))

    assert model_cache.cache_key(p, _closure(), "ROW", False) is None


# ------------------------------------------------------------------------ modelo


def test_single_period_con_hooks():
    """El SP que lee el driver (_build_sp_reference) tiene la celda libre y la fila."""
    from equilibria.templates.gtap.gtap_block_model import build_block_single_period

    p = _nus333_params()
    _hooks_desempleo()
    sp = cast(Any, build_block_single_period(p, p.sets, _closure(), "ROW"))

    assert not sp.aft["USA", "LABOR"].fixed
    assert sp.aft["ROW", "LABOR"].fixed
    assert sp.aft["USA", "CAPITAL"].fixed
    assert ("USA", "LABOR") in sp.eq_wreal
    assert len(sp.eq_wreal) == 1
    assert sp._endogenous_instrument_cells == {"aft": frozenset({("USA", "LABOR")})}


def test_multiperiodo_con_hooks():
    from pyomo.environ import value

    from equilibria.templates.gtap.gtap_block_model import build_block_model

    p = _nus333_params()
    _hooks_desempleo()
    m, _ = cast(
        Any, build_block_model(p, p.sets, _closure(), "ROW", base_calibrated=False)
    )

    for t in ("base", "check", "shock"):
        assert not m.aft["USA", "LABOR", t].fixed, t
        assert m.aft["ROW", "LABOR", t].fixed, t
        assert ("USA", "LABOR", t) in m.eq_wreal, t
    assert m._endogenous_instrument_cells == {"aft": frozenset({("USA", "LABOR")})}
    # En la base la fila se cumple: el Tornqvist vale 1 y pft es su valor de base.
    body = m.eq_wreal["USA", "LABOR", "base"]
    assert abs(value(body.body) - value(body.upper)) < 1e-12


def test_sin_hooks_no_cambia_nada():
    from equilibria.templates.gtap.gtap_block_model import build_block_single_period

    p = _nus333_params()
    sp = cast(Any, build_block_single_period(p, p.sets, _closure(), "ROW"))
    assert sp.aft["USA", "LABOR"].fixed
    assert not hasattr(sp, "eq_wreal")
    assert sp._endogenous_instrument_cells == {}


def test_fix_instrument_shock_sobre_celda_endogena_falla():
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.instruments import fix_instrument_shock

    p = _nus333_params()
    _hooks_desempleo()
    m, _ = cast(
        Any, build_block_model(p, p.sets, _closure(), "ROW", base_calibrated=False)
    )
    with pytest.raises(ValueError, match="endogen"):
        fix_instrument_shock(m, "aft", ("USA", "LABOR"), factor=1.1)


def test_is_exogenous_por_celda():
    from equilibria.templates.gtap.gtap_block_model import build_block_model

    inst = cast(Any, importlib.import_module("equilibria.templates.gtap.instruments"))
    p = _nus333_params()
    _hooks_desempleo()
    m, _ = cast(
        Any, build_block_model(p, p.sets, _closure(), "ROW", base_calibrated=False)
    )
    assert not inst.is_exogenous(m, "aft", ("USA", "LABOR", "shock"))
    assert inst.is_exogenous(m, "aft", ("ROW", "LABOR", "shock"))
    assert inst.is_exogenous(m, "prdtx_rai", ("USA", "MFG", "MFG", "shock"))
    assert not inst.is_exogenous(m, "pft", ("USA", "LABOR", "shock"))
