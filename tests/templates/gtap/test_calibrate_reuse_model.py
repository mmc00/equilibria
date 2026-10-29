"""calibrate_base resuelve el settle sobre el modelo ya construido y lo restaura.

Antes armaba un segundo modelo multiperiodo identico (15x10: 8 de los 14,5 s de la
calibracion). Lo que hay que garantizar: (1) la foto de model_state devuelve el
modelo exactamente a como estaba, haga lo que haga el driver; (2) el punto asentado
es el mismo que con el modelo gemelo; (3) build_block_model ya no construye dos
veces.
"""

import contextlib
import io
from pathlib import Path

import pytest
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Param,
    Reals,
    Suffix,
    Var,
)

from equilibria.sources import dataset_dir
from equilibria.templates.gtap import model_state
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

ROOT = Path(__file__).resolve().parents[3]


def _digest(m):
    """Todo lo mutable del modelo, por valor (y la identidad de cada expresion)."""
    out = [("componentes", tuple((n, id(c)) for n, c in m.component_map().items()))]
    out.append(("atributos", tuple(sorted(model_state._atributos(m).items(), key=str))))
    for v in m.component_objects(Var):
        for k, vd in v.items():
            out.append((v.name, k, vd.value, vd.fixed, vd.lb, vd.ub, vd.domain.name))
    for c in m.component_objects(Constraint):
        out.append((c.name, c.active))
        for k, cd in c.items():
            out.append((c.name, k, cd.active, id(cd._expr)))
    for p in m.component_objects(Param):
        if p.mutable:
            out.extend((p.name, k, pd.value) for k, pd in p.items())
    for s in m.component_objects(Suffix):
        out.append((s.name, tuple(sorted((id(k), v) for k, v in s.items()))))
    return out


def _juguete():
    m = ConcreteModel()
    m.x = Var([1, 2], bounds=(0, 10), initialize=1.0, domain=NonNegativeReals)
    m.p = Param(mutable=True, initialize=2.0)
    m.c = Constraint([1, 2], rule=lambda m, i: m.x[i] >= m.p)
    m.s = Suffix(direction=Suffix.EXPORT)
    m.s[m.x[1]] = 3.0
    m._etiqueta = "antes"
    return m


def test_restore_deshace_todo_lo_que_hace_el_driver():
    m = _juguete()
    antes = _digest(m)
    st = model_state.capture(m)
    # Lo que el driver hace al resolver un periodo.
    m.x[1].value = 7.0
    m.x[2].fix(3.0)
    m.x[1].setlb(None)
    m.x[1].setub(99)
    m.x[2].domain = Reals
    m.c[1].deactivate()
    m.c[2].set_value(m.x[2] <= 5)
    m.p.value = 9.0
    m.s[m.x[1]] = 8.0
    m.s[m.x[2]] = 1.0
    m.obj = Objective(expr=m.x[1])
    m.c.deactivate()
    m._etiqueta = "despues"
    m._nuevo = 1
    m.del_component("s")
    m.s = Suffix(direction=Suffix.EXPORT)
    assert _digest(m) != antes
    model_state.restore(m, st)
    assert _digest(m) == antes


@pytest.fixture(scope="module")
def p3x3():
    from equilibria.templates.gtap import GTAPParameters

    d = dataset_dir("gtap7_3x3")
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    return p


def _closure():
    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        if_sub=True,
        savf_flag="capFix",
        numeraire="pnum",
        va_subsidy_basis="gempack",
    )


def _quieto(fn, *a, **k):
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        return fn(*a, **k)


@pytest.mark.needs_path
def test_settle_sobre_el_modelo_da_el_mismo_punto_y_lo_deja_intacto(p3x3, monkeypatch):
    from equilibria.blocks.gtap.factor import FactorBlock
    from equilibria.templates.gtap.gtap_block_model import build_block_model

    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE_DISABLE", "1")
    rres = list(p3x3.sets.r)[-1]
    fb = FactorBlock(sets=p3x3.sets, params=p3x3)
    gemelo = _quieto(fb.calibrate_base, p3x3, p3x3.sets, _closure(), rres)

    m, _ = _quieto(build_block_model, p3x3, p3x3.sets, _closure(), rres)
    antes = _digest(m)
    propio = _quieto(fb.calibrate_base, p3x3, p3x3.sets, _closure(), rres, model=m)
    assert _digest(m) == antes
    assert propio.keys() == gemelo.keys()
    for vn, celdas in gemelo.items():
        for k, val in celdas.items():
            assert propio[vn][k] == pytest.approx(val, rel=1e-12, abs=1e-14), (vn, k)


@pytest.mark.needs_path
def test_build_calibrado_construye_una_sola_vez(p3x3, monkeypatch):
    import equilibria.templates.gtap.gtap_block_model as gbm

    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE_DISABLE", "1")
    llamadas = []
    original = gbm.GTAPBlockMultiPeriodModel.build_equations_all_periods

    def espia(self, *a, **k):
        llamadas.append(1)
        return original(self, *a, **k)

    monkeypatch.setattr(
        gbm.GTAPBlockMultiPeriodModel, "build_equations_all_periods", espia
    )
    m, _ = _quieto(
        gbm.build_block_model,
        p3x3,
        p3x3.sets,
        _closure(),
        list(p3x3.sets.r)[-1],
        base_calibrated=True,
    )
    assert m._settled_seed
    assert m._base_calibrated is True
    assert len(llamadas) == 1
