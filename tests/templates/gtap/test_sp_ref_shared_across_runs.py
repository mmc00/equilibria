"""calibrate_base y solve_multiperiod no construyen dos veces el mismo modelo de referencia.

QUE PASA.  ``calibrate_base`` resuelve el modelo con su propio ``PeriodPreparer``
(``settle_only``) y ``solve_multiperiod`` crea otro.  Cada uno construia el
modelo de referencia del base: en 20x41 son ~46 s por construccion y las dos
salen IDENTICAS (medido en 15x10: 0 diferencias en fijadas, valores y cotas;
mismo objeto ``params``).

QUE SE FIJA AQUI.  La foto del modelo de referencia (fijadas + valor + cotas) se
comparte entre los ``PeriodPreparer`` del MISMO modelo ``m``, con clave por
CONTENIDO (``fingerprint``: params, cierre, region residual y codigo).  Cualquier
diferencia en esa clave obliga a construir de nuevo; reusar con otra clave
cambiaria la solucion sin lanzar nada.  Y replicar desde la foto deja ``m`` igual
que replicar desde el modelo.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pyomo.environ import ConcreteModel, Var

from equilibria.templates.gtap import gtap_multiperiod_driver as drv
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
from equilibria.templates.gtap.period_prep import PeriodPreparer


def _closure(name: str = "base", **kw) -> GTAPClosureConfig:
    cfg = {
        "name": name,
        "closure_type": "MCP",
        "capital_mobility": "sluggish",
        "savf_flag": "capFix",
        "fix_endowments": False,
        "fix_taxes": False,
        "fix_technology": False,
        "if_sub": True,
        "numeraire": "pnum",
    }
    cfg.update(kw)
    return GTAPClosureConfig(**cfg)  # ty: ignore[invalid-argument-type]


class _Params:
    """Lo minimo que ``fingerprint`` digiere: un benchmark con datos y un impuesto."""

    def __init__(self, tax: float = 0.1) -> None:
        self.sets = None
        self.benchmark = SimpleNamespace(vfm={("land", "a", "r"): 1.0})
        self.taxes = SimpleNamespace(imptx={("a", "b"): tax})


def _cell(var, idx=None) -> Any:
    """La celda de Pyomo (``VarData``); ty no la distingue de ``ComponentData``."""
    return var if idx is None else var[idx]


def _sp_model() -> ConcreteModel:
    """Un modelo de referencia chico con las tres formas de indice que replica el
    driver: tupla, escalar y sin indice."""
    sp = ConcreteModel()
    sp.x = Var([("r1", "i1"), ("r1", "i2")], initialize=2.0)
    sp.y = Var(["r1", "r2"], initialize=3.0)
    sp.z = Var(initialize=4.0)
    _cell(sp.x, ("r1", "i1")).fix(1.5)
    _cell(sp.x, ("r1", "i2")).setlb(0.25)
    _cell(sp.y, "r1").fix()  # valor 3.0
    _cell(sp.y, "r2").setub(9.0)
    _cell(sp.z).fix(None)  # sin valor: se replica como 0.0
    return sp


def _mp_model() -> ConcreteModel:
    m = ConcreteModel()
    periods = ["base", "check"]
    m.x = Var([("r1", "i1", p) for p in periods] + [("r1", "i2", p) for p in periods])
    m.y = Var([(r, p) for r in ("r1", "r2") for p in periods])
    m.z = Var(periods)
    _cell(m.y, ("r1", "check")).fix(7.0)  # ya fija en m: no se pisa
    return m


def _state(m) -> dict:
    return {
        (v.name, i): (v[i].fixed, v[i].value, v[i].lb, v[i].ub)
        for v in m.component_objects(Var)
        for i in v
    }


@pytest.fixture
def builds(monkeypatch):
    """Cuenta las construcciones del modelo de referencia (reemplazado por uno chico)."""
    calls: list[str] = []

    def fake_build(sets, params, closure, residual_region, model=None):
        calls.append(closure.name)
        return _sp_model()

    monkeypatch.setattr(drv, "_build_sp_reference", fake_build)
    return calls


def _prep(closure=None, residual_region: str = "ROW") -> PeriodPreparer:
    c = closure or _closure()
    return PeriodPreparer(
        mode="gtap",
        base_closure=c,
        alt_closure=_closure("altertax"),
        residual_region=residual_region,
    )


# ── Replicar desde la foto = replicar desde el modelo ───────────────────────────


@pytest.mark.parametrize("period", ["base", "check"])
def test_replicating_the_snapshot_leaves_m_like_replicating_the_model(period):
    via_model, via_snapshot = _mp_model(), _mp_model()
    drv._replicate_sp_fixing(via_model, _sp_model(), period)
    drv._replicate_sp_bounds(via_model, _sp_model(), period)

    snap = drv._sp_snapshot(_sp_model())
    drv._replicate_snapshot_fixing(via_snapshot, snap, period)
    drv._replicate_snapshot_bounds(via_snapshot, snap, period)

    assert _state(via_snapshot) == _state(via_model)


# ── Se comparte entre corridas del mismo modelo, solo con la misma clave ────────


def test_a_second_run_on_the_same_model_reuses_the_reference(builds):
    m, params, closure = _mp_model(), _Params(), _closure()
    _prep(closure)._reference_model(m, params, closure, "base")  # calibrate_base
    _prep(closure)._reference_model(m, params, closure, "base")  # solve_multiperiod
    assert builds == ["base"]


def test_other_params_content_builds_again(builds):
    m, closure = _mp_model(), _closure()
    _prep(closure)._reference_model(m, _Params(tax=0.1), closure, "base")
    _prep(closure)._reference_model(m, _Params(tax=0.2), closure, "base")
    assert builds == ["base", "base"]


def test_params_mutated_in_place_builds_again(builds):
    """La clave es el CONTENIDO: el mismo objeto cambiado no reusa la foto vieja."""
    m, params, closure = _mp_model(), _Params(), _closure()
    _prep(closure)._reference_model(m, params, closure, "base")
    params.taxes.imptx[("a", "b")] = 0.3
    _prep(closure)._reference_model(m, params, closure, "base")
    assert builds == ["base", "base"]


def test_another_closure_builds_again(builds):
    m, params = _mp_model(), _Params()
    alt = _closure("altertax", capital_mobility="mobile", fix_taxes=True)
    _prep()._reference_model(m, params, _closure(), "base")
    _prep()._reference_model(m, params, alt, "check")
    assert builds == ["base", "altertax"]


def test_another_residual_region_builds_again(builds):
    m, params, closure = _mp_model(), _Params(), _closure()
    _prep(closure, "ROW")._reference_model(m, params, closure, "base")
    _prep(closure, "USA")._reference_model(m, params, closure, "base")
    assert builds == ["base", "base"]


def test_another_model_builds_again(builds):
    params, closure = _Params(), _closure()
    _prep(closure)._reference_model(_mp_model(), params, closure, "base")
    _prep(closure)._reference_model(_mp_model(), params, closure, "base")
    assert builds == ["base", "base"]


def test_an_active_overwrite_hook_disables_the_sharing(builds, monkeypatch):
    """Un hook @overwrite cambia el modelo y no esta en la clave (como el cache
    del modelo): con uno activo no se comparte."""
    from equilibria.blocks.gtap.overwrite import overwrite

    monkeypatch.setattr(overwrite, "active", lambda: True)
    m, params, closure = _mp_model(), _Params(), _closure()
    _prep(closure)._reference_model(m, params, closure, "base")
    _prep(closure)._reference_model(m, params, closure, "base")
    assert builds == ["base", "base"]


def test_the_shared_reference_does_not_travel_with_the_model(builds):
    """La foto no se guarda como atributo de ``m``: el cache del modelo lo pickle-a
    y no debe llevarse una foto de un modelo de un periodo."""
    m, params, closure = _mp_model(), _Params(), _closure()
    _prep(closure)._reference_model(m, params, closure, "base")
    assert not [k for k in vars(m) if "sp_ref" in k or "snapshot" in k]
