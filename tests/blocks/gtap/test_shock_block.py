"""ShockBlock: los instrumentos de shock son Vars FIJAS en su benchmark.

GAMS declara lambdava/aft/imptx/... como variables (o parametros por periodo) y el
shock es `x.fx(...,'shock') = v`. Aca el ShockBlock los declara como Vars, el
compositor las fija y deja el registro `_exogenous_instruments` para que el driver
no las libere. Sin solver: gtap7_3x3.
"""

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATASET = ROOT / "datasets" / "gtap7_3x3"


def _params():
    from equilibria.templates.gtap import GTAPParameters

    p = GTAPParameters()
    p.load_from_har(
        basedata_path=DATASET / "basedata.har",
        sets_path=DATASET / "sets.har",
        default_path=DATASET / "default.prm",
        baserate_path=DATASET / "baserate.har",
    )
    return p


def _sp(p):
    from equilibria.templates.gtap.gtap_block_model import build_block_single_period
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

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


def test_registro_lista_los_instrumentos():
    from equilibria.blocks.gtap.shock import SHOCK_INSTRUMENTS

    m = _sp(_params())
    assert m._exogenous_instruments == frozenset(SHOCK_INSTRUMENTS)


def test_lambdava_es_var_fija_en_su_benchmark():
    from pyomo.environ import Var, value

    p = _params()
    m = _sp(p)
    assert m.lambdava.ctype is Var
    for (r, a), vd in m.lambdava.items():
        assert vd.fixed, (r, a)
        assert float(value(vd)) == pytest.approx(
            float(p.shifts.lambdava.get((r, a), 1.0)), abs=0.0
        )
