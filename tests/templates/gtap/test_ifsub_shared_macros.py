"""Instancias de macro ifSUB compartidas entre filas (memoria del build).

Bajo ifSUB=1, M_PM[e,c,imp] (y M_PMCIF/M_PEFOB/M_PWMG/M_XWMG) aparece inline en
eq_xweq, eq_pmteq, eq_gdpmp y eq_ytax. Antes cada fila construia su propia copia:
en 20x41 dos tercios de los nodos de expresion eran repetidos. Ahora cada instancia
se arma una vez en el SP y la copia multiperiodo la copia una vez por periodo.

Lo que hay que garantizar: (1) fuera de ``sharing()`` nada se comparte (el driver
reconstruye filas con params cambiados); (2) en el modelo multiperiodo las filas de
un periodo comparten la instancia; (3) ninguna fila de un periodo toca variables de
otro, o sea la copia compartida no se filtra entre periodos.
"""

import contextlib
import io
from pathlib import Path

import pytest
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import Constraint

from equilibria.blocks.gtap import _ifsub_macros as mac
from equilibria.sources import dataset_dir
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

ROOT = Path(__file__).resolve().parents[3]
FAMILIAS = ("eq_xweq", "eq_pmteq", "eq_gdpmp", "eq_ytax")


def test_shared_only_inside_context():
    fuera = [mac.shared("m", "pm", (1,), object) for _ in range(2)]
    assert fuera[0] is not fuera[1]
    with mac.sharing() as memo:
        dentro = [mac.shared("m", "pm", (1,), object) for _ in range(2)]
        otro_idx = mac.shared("m", "pm", (2,), object)
    assert dentro[0] is dentro[1]
    assert otro_idx is not dentro[0]
    assert len(memo) == 2
    # Al salir del contexto vuelve a construir cada vez.
    assert mac.shared("m", "pm", (1,), object) is not dentro[0]


@pytest.fixture(scope="module")
def modelo_3x3():
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_block_model import build_block_model

    d = dataset_dir("gtap7_3x3")
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    closure = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        if_sub=True,
        savf_flag="capFix",
        numeraire="pnum",
        va_subsidy_basis="gempack",
    )
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        m, _ = build_block_model(
            p, p.sets, closure, list(p.sets.r)[-1], base_calibrated=False
        )
    return m


def _nodos(expr) -> set[int]:
    """ids de los nodos no-hoja de una expresion."""
    vistos: set[int] = set()
    pila = [expr]
    while pila:
        x = pila.pop()
        if not hasattr(x, "is_expression_type") or not x.is_expression_type():
            continue
        if id(x) in vistos:
            continue
        vistos.add(id(x))
        pila.extend(x.args)
    return vistos


def _filas(m, familia, periodo):
    con = getattr(m, familia)
    return [con[k] for k in con if k[-1] == periodo and con[k].active]


def test_filas_de_un_periodo_comparten_la_instancia(modelo_3x3):
    m = modelo_3x3
    xweq = set().union(*(_nodos(c.body) for c in _filas(m, "eq_xweq", "shock")))
    pmteq = set().union(*(_nodos(c.body) for c in _filas(m, "eq_pmteq", "shock")))
    # M_PM[rp,i,r] esta en xweq[rp,i,r] y en pmteq[r,i]: el mismo objeto.
    assert xweq & pmteq


def test_periodos_no_comparten_nodos(modelo_3x3):
    m = modelo_3x3
    check = set().union(
        *(_nodos(c.body) for f in FAMILIAS for c in _filas(m, f, "check"))
    )
    shock = set().union(
        *(_nodos(c.body) for f in FAMILIAS for c in _filas(m, f, "shock"))
    )
    assert check and shock
    assert not (check & shock)


@pytest.mark.parametrize("familia", FAMILIAS)
def test_cada_fila_solo_toca_su_periodo(modelo_3x3, familia):
    m = modelo_3x3
    for periodo in ("check", "shock"):
        filas = _filas(m, familia, periodo)
        assert filas
        for c in filas:
            for v in identify_variables(c.body, include_fixed=True):
                idx = v.index()
                assert (
                    isinstance(idx, tuple) and idx[-1] == periodo or idx == periodo
                ), f"{c.name} usa {v.name}"


def test_sp_registra_las_instancias(modelo_3x3):
    # build_block_model no expone el SP; se arma uno y se mira el registro.
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_block_model import build_block_single_period

    d = dataset_dir("gtap7_3x3")
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    closure = GTAPClosureConfig(
        name="base", closure_type="MCP", capital_mobility="sluggish", if_sub=True
    )
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        sp = build_block_single_period(p, p.sets, closure, list(p.sets.r)[-1])
    shared: dict = getattr(sp, "_ifsub_shared", {})
    assert shared
    # Cada registro es una expresion viva usada por alguna fila activa del SP.
    usados = set().union(
        *(_nodos(c.body) for c in sp.component_data_objects(Constraint, active=True))
    )
    assert set(shared) & usados
