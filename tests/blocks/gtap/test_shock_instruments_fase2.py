"""ShockBlock, fase 2: imptx, prdtx_rai, fcttx, dintx_tgt, mintx_tgt, kappaf, exptx,
lambdaf, axp, lambdam.

Cada instrumento es una Var FIJA en su benchmark, registrada, y las ecuaciones que
lo usan lo leen como Var: moverlo mueve su residuo. Si una ecuacion horneara el
literal, un shock en la celda 'shock' no entraria al modelo. Sin solver: gtap7_3x3.
"""

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATASET = ROOT / "datasets" / "gtap7_3x3"

# instrumento -> ecuaciones que tienen que leerlo
READERS = {
    "imptx": ("eq_pmeq", "eq_ytax"),
    "prdtx_rai": ("eq_pp_rai", "eq_ytax"),
    "fcttx": ("eq_pfaeq", "eq_ytax"),
    "dintx_tgt": ("eq_dintxeq", "eq_ytax"),
    "mintx_tgt": ("eq_mintxeq", "eq_ytax"),
    # tinc -> kappaf: pfy, recaudacion dt, arent (model.gms:1121/684/1145) y xf
    # de los factores sluggish (pf*(1-kappaf) inline).
    "kappaf": ("eq_pfyeq", "eq_ytax", "eq_arent", "eq_xfeq"),
    # txs -> exptx: pefob y recaudacion et (model.gms:1034/676).
    "exptx": ("eq_pefobeq", "eq_ytax"),
    "lambdaf": ("eq_xfeq", "eq_pvaeq"),
    "axp": ("eq_nd", "eq_pxeq"),
    "lambdam": ("eq_xweq", "eq_pmteq"),
}


@pytest.fixture(scope="module")
def sp():
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_block_model import build_block_single_period
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    p = GTAPParameters()
    p.load_from_har(
        basedata_path=DATASET / "basedata.har",
        sets_path=DATASET / "sets.har",
        default_path=DATASET / "default.prm",
        baserate_path=DATASET / "baserate.har",
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
    return build_block_single_period(p, p.sets, gc, list(p.sets.r)[-1])


@pytest.mark.parametrize("name", sorted(READERS))
def test_es_instrumento_registrado_y_fijo(sp, name):
    from pyomo.environ import Var

    from equilibria.blocks.gtap.shock import SHOCK_INSTRUMENTS

    assert name in SHOCK_INSTRUMENTS
    assert name in sp._exogenous_instruments
    var = getattr(sp, name)
    assert var.ctype is Var
    assert len(var) > 0, f"{name} sin celdas"
    assert all(vd.fixed for vd in var.values()), f"{name} con celdas libres"


@pytest.mark.parametrize(
    ("name", "eq"), [(n, e) for n, eqs in READERS.items() for e in eqs]
)
def test_la_ecuacion_lee_el_instrumento(sp, name, eq):
    """Mover el instrumento (todas sus celdas) tiene que mover algun residuo de eq."""
    from pyomo.environ import value

    var = getattr(sp, name)
    con = getattr(sp, eq)
    antes = {k: float(value(cd.body)) for k, cd in con.items()}
    orig = {k: float(value(vd)) for k, vd in var.items()}
    try:
        for k, vd in var.items():
            vd.set_value(orig[k] * 1.1 + 0.05)
        cambian = sum(
            abs(float(value(cd.body)) - antes[k]) > 1e-9 for k, cd in con.items()
        )
    finally:
        for k, vd in var.items():
            vd.set_value(orig[k])
    assert cambian > 0, f"{eq} no lee {name}: el shock no entraria"


@pytest.mark.parametrize("endogena", ["dintx", "mintx"])
def test_eq_ytax_no_acopla_la_tasa_endogena(sp, endogena):
    """eq_ytax lee la tasa del instrumento fijo (dintx_tgt/mintx_tgt), no la Var.

    En GAMS dintx esta fijo por periodo: en ytaxeq es una constante. Leer la Var
    endogena da la misma solucion pero agrega 2700 acoples en 15x10 y PATH cae en
    otra raiz (pure ifSUB=1 shock 100% -> 88,34%, pft[USA,Land] en su piso).
    """
    from pyomo.core.expr.visitor import identify_variables

    acoplan = [
        k
        for k, cd in sp.eq_ytax.items()
        if any(
            v.parent_component() is getattr(sp, endogena)
            for v in identify_variables(cd.body)
        )
    ]
    assert not acoplan, acoplan[:3]
