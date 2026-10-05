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
from tests.templates.gtap._desempleo import register_desempleo_hooks
from tests.templates.gtap._nus333 import (
    closure,
    nus333_params,
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
    @overwrite(ShockBlock, period="shock")
    def uno(b):
        pass

    assert overwrite.registered(ShockBlock) == ["uno"]


def test_reregistrar_la_misma_funcion_no_duplica():
    """Re-ejecutar la celda del notebook reemplaza el hook, no lo apila."""
    for _ in range(2):

        @overwrite(ShockBlock, period="shock")
        def uno(b):
            pass

    assert overwrite.registered(ShockBlock) == ["uno"]


def test_clear_limpia_todo():
    @overwrite(ShockBlock, period="shock")
    def uno(b):
        pass

    @overwrite(ClosureBlock, period="shock")
    def dos(b):
        pass

    assert overwrite.active()
    overwrite.clear()
    assert not overwrite.active()
    assert overwrite.registered(ShockBlock) == []


def test_period_distinto_de_shock_falla():
    """El cierre de @overwrite rige solo en el shock (como el swap de GEMPACK y GAMS)."""
    with pytest.raises(ValueError, match="shock"):
        overwrite(ShockBlock, period="check")


def test_equation_repetida_suma_celdas_a_la_misma_fila():
    """Varias llamadas con el mismo nombre arman UNA familia de filas (ME9A: eq_aoreg
    en MFG y SER de cada region)."""
    from equilibria.templates.gtap.gtap_block_model import build_block_single_period

    p = nus333_params()

    @overwrite(ShockBlock, period="shock")
    def libre(b):
        for a in ("MFG", "SER"):
            b.endogenous("axp", ("USA", a))

    @overwrite(ClosureBlock, period="shock")
    def parejo(b):
        for a in ("MFG", "SER"):
            b.equation(
                "eq_parejo",
                ("USA", a),
                lambda m, r, a: m.axp[r, a] == m.axp[r, "AGR"],
                domains=("r", "a"),
            )

    sp = cast(Any, build_block_single_period(p, p.sets, closure(), "ROW"))
    assert set(sp.eq_parejo) == {("USA", "MFG"), ("USA", "SER")}


def test_endogenous_de_algo_que_no_es_instrumento_falla():
    from equilibria.templates.gtap.gtap_block_model import build_block_single_period

    p = nus333_params()

    @overwrite(ShockBlock, period="shock")
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


def test_multiperiodo_con_hooks_solo_en_el_shock(mp_desempleo):
    """En base y check rige el cierre estandar: la celda fija y sin la fila nueva.
    En el shock, la celda libre y la fila."""
    m = mp_desempleo
    for t in ("base", "check"):
        assert m.aft["USA", "LABOR", t].fixed, t
        assert ("USA", "LABOR", t) not in m.eq_wreal, t
    assert not m.aft["USA", "LABOR", "shock"].fixed
    assert set(m.eq_wreal) == {("USA", "LABOR", "shock")}
    for t in ("base", "check", "shock"):
        assert m.aft["ROW", "LABOR", t].fixed, t
    assert m._endogenous_instrument_cells == {"aft": frozenset({("USA", "LABOR")})}


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
    assert is_exogenous(m, "aft", ("USA", "LABOR", "check"))
    assert is_exogenous(m, "aft", ("USA", "LABOR", "base"))
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
    # Activo el check: la celda sigue fija (exogena fuera del shock).
    freeze_inactive_periods(m, "check")
    assert m.aft["USA", "LABOR", "check"].fixed


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


# ------------------------------------------------------------------------ target


def _register_qca_target() -> None:
    """``b.target``: objetivo de cantidad de TBL94, ``x[shock] = qca_target x x[check]``."""

    @overwrite(ShockBlock, period="shock")
    def qca_target(b):
        b.endogenous("prdtx_rai", ("USA", "MFG", "MFG"))
        b.target(
            "qca_target",
            ("USA", "MFG", "MFG"),
            quantity=lambda m, r, a, i: m.x[r, a, i],
            domains=("r", "a", "i"),
        )


def test_target_es_un_factor_fijo_en_1():
    """El objetivo es una Var nueva, registrada como instrumento y fija en 1 (un
    factor sobre el check). En el SP la fila es ``x = qca_target x x_base``."""
    from pyomo.environ import value

    from equilibria.templates.gtap.gtap_block_model import build_block_single_period

    p = nus333_params()
    _register_qca_target()
    sp = cast(Any, build_block_single_period(p, p.sets, closure(), "ROW"))

    assert "qca_target" in sp._exogenous_instruments
    assert all(vd.fixed for vd in sp.qca_target.values())
    assert all(float(value(vd)) == 1.0 for vd in sp.qca_target.values())
    cell = ("USA", "MFG", "MFG")
    assert set(sp.eq_qca_target) == {cell}
    row = sp.eq_qca_target[cell]
    assert abs(value(row.body) - value(row.upper)) < 1e-9


def test_target_multiperiodo_fila_solo_en_el_shock_contra_el_check():
    """La fila vive solo en el shock y lee la cantidad del CHECK:
    ``x[shock] = qca_target[shock] x x[check]``. El shock entra como factor."""
    from pyomo.core.expr.visitor import identify_variables
    from pyomo.environ import value

    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.instruments import fix_instrument_shock

    p = nus333_params()
    _register_qca_target()
    m, _ = cast(Any, build_block_model(p, p.sets, closure(), "ROW"))
    cell = ("USA", "MFG", "MFG")
    for t in ("base", "check", "shock"):
        assert m.qca_target[(*cell, t)].fixed, t
        assert float(value(m.qca_target[(*cell, t)])) == 1.0, t
    assert set(m.eq_qca_target) == {(*cell, "shock")}
    row = m.eq_qca_target[(*cell, "shock")]
    names = {v.name for v in identify_variables(row.body)}
    assert {
        "x[USA,MFG,MFG,shock]",
        "x[USA,MFG,MFG,check]",
        "qca_target[USA,MFG,MFG,shock]",
    } <= names

    fix_instrument_shock(m, "qca_target", cell, factor=0.99)
    assert float(value(m.qca_target[(*cell, "shock")])) == pytest.approx(0.99)
    assert float(value(m.qca_target[(*cell, "check")])) == 1.0


def test_target_con_nombre_de_variable_existente_falla():
    from equilibria.templates.gtap.gtap_block_model import build_block_single_period

    p = nus333_params()

    @overwrite(ShockBlock, period="shock")
    def mal(b):
        b.target("x", ("USA", "MFG", "MFG"), quantity=lambda m, *c: 1.0)

    with pytest.raises(ValueError, match="x"):
        build_block_single_period(p, p.sets, closure(), "ROW")


def test_block_edit_expone_lo_que_declararon_los_hooks():
    """``_HookedBlock`` lee lo declarado por metodos publicos del ``BlockEdit``."""
    from equilibria.blocks.gtap.overwrite import BlockEdit, Target

    def ini(m, r):
        return 1.0

    edit = BlockEdit(None, {}, [])
    edit.endogenous("aft", ("USA", "LABOR"))
    edit.target("t", ("USA",), quantity=ini, domains=("r",))
    assert edit.endogenized() == {"aft": {("USA", "LABOR")}}
    assert edit.declared_targets() == {"t": Target(("r",), [(("USA",), ini)])}


def test_target_con_dominios_distintos_falla():
    """Mismo objetivo con otros dominios: falla en el mismo hook y entre bloques."""
    from equilibria.blocks.gtap.overwrite import BlockEdit, Target, collect_targets

    edit = BlockEdit(None, {}, [])
    edit.target("t", ("USA",), quantity=lambda m, r: 1.0, domains=("r",))
    with pytest.raises(ValueError, match="dominios"):
        edit.target("t", ("USA",), quantity=lambda m, r: 1.0, domains=("rp",))

    def ini(m, r):
        return 1.0

    class _B:
        def __init__(self, doms):
            self.targets = {"t": Target(doms, [(("USA",), ini)])}

    with pytest.raises(ValueError, match="dominios"):
        collect_targets([_B(("r",)), _B(("rp",))])
    got = collect_targets([_B(("r",)), _B(("r",))])
    assert got == {"t": Target(("r",), [(("USA",), ini), (("USA",), ini)])}


def test_vista_de_periodo_traduce_vars_y_rechaza_lo_demas():
    """``quantity`` de ``b.target`` se evalua en un periodo: ``v[k]`` es ``v[(*k, t)]``.
    Un componente indexado que no es Var no se puede traducir: falla en voz alta."""
    from pyomo.environ import ConcreteModel, Param, Var

    from equilibria.blocks.gtap.overwrite import _AtPeriod

    m = ConcreteModel()
    keys = [("USA", t) for t in ("base", "check", "shock")]
    m.x = Var(keys, initialize=1.0)
    m.p = Param(keys, initialize=2.0, mutable=True)
    m.k = Param(initialize=3.0)
    # Una Var escalar del modelo de un periodo queda indexada solo por el periodo.
    m.s = Var(["base", "check", "shock"], initialize=1.0)

    view = _AtPeriod(m, "check")
    assert view.x["USA"] is m.x["USA", "check"]
    assert view.s is m.s["check"]
    assert view.k is m.k
    with pytest.raises(ValueError, match="p"):
        _ = view.p
