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


def test_eq_va_y_eq_pxeq_dependen_del_instrumento():
    """Mover lambdava[r,a] tiene que mover el residuo de eq_va y eq_pxeq: si la
    ecuacion horneara el literal, el shock no entraria."""
    from pyomo.environ import value

    m = _sp(_params())
    for name in ("eq_va", "eq_pxeq"):
        con = getattr(m, name)
        (r, a), cd = next(iter(con.items()))
        antes = float(value(cd.body))
        m.lambdava[r, a].set_value(1.25)
        despues = float(value(cd.body))
        m.lambdava[r, a].set_value(1.0)
        assert abs(despues - antes) > 1e-9, f"{name}[{r},{a}] no lee lambdava"


def test_aft_es_instrumento_y_aft0_el_benchmark():
    from pyomo.environ import value

    from equilibria.blocks.gtap import _derived_params as dp

    p = _params()
    m = _sp(p)
    bench = dp.aft_data(p, p.sets)
    for (r, f), vd in m.aft.items():
        assert vd.fixed, (r, f)
        assert float(value(vd)) == pytest.approx(float(bench.get((r, f), 0.0)))
        assert float(value(m.aft0[r, f])) == pytest.approx(
            float(bench.get((r, f), 0.0))
        )


def test_eq_xfteq_depende_del_instrumento_aft():
    from pyomo.environ import value

    m = _sp(_params())
    (r, f), cd = next(iter(m.eq_xfteq.items()))
    antes = float(value(cd.body))
    x0 = float(value(m.aft[r, f]))
    m.aft[r, f].set_value(x0 * 1.1)
    despues = float(value(cd.body))
    m.aft[r, f].set_value(x0)
    assert abs(despues - antes) > 1e-9


def test_kstock_no_depende_del_instrumento_aft():
    """krat = xft_bench/kstock_bench es CALIBRACION: con qe=10 el kstock sube 10%
    (GAMS kstockeq: krat*kstock = xft), no se recalibra krat."""
    from pyomo.environ import value

    m = _sp(_params())
    for r, cd in m.eq_kstock.items():
        antes = float(value(cd.body))
        for f in m.f:
            if (r, f) in m.aft:
                m.aft[r, f].set_value(float(value(m.aft[r, f])) * 1.1)
        despues = float(value(cd.body))
        for f in m.f:
            if (r, f) in m.aft:
                m.aft[r, f].set_value(float(value(m.aft[r, f])) / 1.1)
        assert despues == pytest.approx(antes, abs=1e-12), r
