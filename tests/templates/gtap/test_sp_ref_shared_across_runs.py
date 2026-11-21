"""La calibracion del base y la corrida no construyen dos veces el mismo modelo de referencia.

QUE PASA.  La calibracion del base resuelve el modelo con su propio cache de
referencias (``SPReferenceCache``, ``settle_only``) y la corrida crea otro.  Cada uno construia el
modelo de referencia del base: en 20x41 son ~46 s por construccion y las dos
salen IDENTICAS (medido en 15x10: 0 diferencias en fijadas, valores y cotas;
mismo objeto ``params``).

QUE SE FIJA AQUI.  La foto del modelo de referencia (fijadas + valor + cotas) se
comparte entre los ``SPReferenceCache`` que reciben el MISMO objeto ``params``
(calibrate_base resuelve otro ``m``), con clave por CONTENIDO (``fingerprint``:
params, cierre, region residual, fuente del modelo y codigo).  Cualquier
diferencia en esa clave obliga a construir de nuevo; reusar con otra clave
cambiaria la solucion sin lanzar nada.  Y replicar desde la foto deja ``m`` igual
que replicar desde el modelo.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pyomo.environ import ConcreteModel, Var

from equilibria.templates.gtap import sp_reference
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
from equilibria.templates.gtap.sp_reference import SPReferenceCache


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
    """Lo minimo que ``fingerprint`` digiere: un benchmark con datos y un impuesto.

    NO hasheable, como ``GTAPParameters`` (dataclass con igualdad)."""

    __hash__ = None  # ty: ignore[invalid-assignment]

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

    monkeypatch.setattr(sp_reference, "build_sp_reference", fake_build)
    return calls


def _prep(closure=None, residual_region: str = "ROW") -> SPReferenceCache:
    return SPReferenceCache(residual_region=residual_region)


# ── Replicar desde la foto = replicar desde el modelo ───────────────────────────


def _expected(period: str) -> dict:
    """``_mp_model`` tras replicar ``_sp_model`` en ``period``, derivado a mano: las
    fijadas del SP se fijan con su valor (sin valor: 0.0) salvo la que ``m`` ya tenia
    fija, las cotas se copian, el otro periodo no se toca."""
    out = {}
    for p in ("base", "check"):
        on = p == period
        out["x", ("r1", "i1", p)] = (True, 1.5, None, None) if on else _FREE
        out["x", ("r1", "i2", p)] = (False, None, 0.25, None) if on else _FREE
        if p == "check":
            out["y", ("r1", p)] = (True, 7.0, None, None)  # ya fija en m
        else:
            out["y", ("r1", p)] = (True, 3.0, None, None) if on else _FREE
        out["y", ("r2", p)] = (False, None, None, 9.0) if on else _FREE
        out["z", p] = (True, 0.0, None, None) if on else _FREE
    return out


_FREE = (False, None, None, None)


@pytest.mark.parametrize("period", ["base", "check"])
def test_replicating_the_snapshot_leaves_m_as_derived(period):
    """El core replica la foto del modelo de referencia (``sp_snapshot``) en ``m``."""
    from equilibria.templates.gtap.period_model import apply_reference
    from equilibria.templates.gtap.pyomo_periods import (
        GTAPPyomoPeriodModel,
        _as_reference,
    )

    m = _mp_model()
    snap = sp_reference.sp_snapshot(_sp_model())
    apply_reference(GTAPPyomoPeriodModel(m), _as_reference(snap, period), period)

    assert _state(m) == _expected(period)


# ── Se comparte entre corridas del mismo modelo, solo con la misma clave ────────


def test_a_second_run_on_the_same_model_reuses_the_reference(builds):
    m, params, closure = _mp_model(), _Params(), _closure()
    _prep(closure).cached(m, params, closure, "base")  # calibrate_base
    _prep(closure).cached(m, params, closure, "base")  # la corrida
    assert builds == ["base"]


def test_other_params_content_builds_again(builds):
    m, closure = _mp_model(), _closure()
    _prep(closure).cached(m, _Params(tax=0.1), closure, "base")
    _prep(closure).cached(m, _Params(tax=0.2), closure, "base")
    assert builds == ["base", "base"]


def test_params_mutated_in_place_builds_again(builds):
    """La clave es el CONTENIDO: el mismo objeto cambiado no reusa la foto vieja."""
    m, params, closure = _mp_model(), _Params(), _closure()
    _prep(closure).cached(m, params, closure, "base")
    params.taxes.imptx[("a", "b")] = 0.3
    _prep(closure).cached(m, params, closure, "base")
    assert builds == ["base", "base"]


def test_another_closure_builds_again(builds):
    m, params = _mp_model(), _Params()
    alt = _closure("altertax", capital_mobility="mobile", fix_taxes=True)
    _prep().cached(m, params, _closure(), "base")
    _prep().cached(m, params, alt, "check")
    assert builds == ["base", "altertax"]


def test_another_residual_region_builds_again(builds):
    m, params, closure = _mp_model(), _Params(), _closure()
    _prep(closure, "ROW").cached(m, params, closure, "base")
    _prep(closure, "USA").cached(m, params, closure, "base")
    assert builds == ["base", "base"]


def test_a_new_model_with_the_same_params_reuses_the_reference(builds):
    """El flujo real: calibrate_base resuelve OTRO modelo (medido: otro id(m)) con
    el MISMO objeto params que despues usa la corrida."""
    params, closure = _Params(), _closure()
    _prep(closure).cached(_mp_model(), params, closure, "base")
    _prep(closure).cached(_mp_model(), params, closure, "base")
    assert builds == ["base"]


def test_another_reference_source_builds_again(builds):
    """Lo unico de ``m`` que lee la construccion es su fuente (monolito o bloques)."""
    params, closure = _Params(), _closure()
    monolith = _mp_model()
    monolith._sp_source = "monolith"
    _prep(closure).cached(_mp_model(), params, closure, "base")
    _prep(closure).cached(monolith, params, closure, "base")
    assert builds == ["base", "base"]


def test_the_env_override_of_the_source_builds_again(builds, monkeypatch):
    params, closure = _Params(), _closure()
    _prep(closure).cached(_mp_model(), params, closure, "base")
    monkeypatch.setenv("EQUILIBRIA_GTAP_REF_MODEL", "blocks")
    _prep(closure).cached(_mp_model(), params, closure, "base")
    assert builds == ["base", "base"]


def test_an_active_overwrite_hook_disables_the_sharing(builds, monkeypatch):
    """Un hook @overwrite cambia el modelo y no esta en la clave (como el cache
    del modelo): con uno activo no se comparte."""
    from equilibria.blocks.gtap.overwrite import overwrite

    monkeypatch.setattr(overwrite, "active", lambda: True)
    m, params, closure = _mp_model(), _Params(), _closure()
    _prep(closure).cached(m, params, closure, "base")
    _prep(closure).cached(m, params, closure, "base")
    assert builds == ["base", "base"]


def test_the_shared_reference_does_not_travel_with_the_model_or_the_params(builds):
    """La foto no se guarda como atributo de ``m`` ni de ``params``: el cache del
    modelo los serializa, y la clave de ese cache digiere TODOS los atributos de
    ``params``."""
    m, params, closure = _mp_model(), _Params(), _closure()
    _prep(closure).cached(m, params, closure, "base")
    assert not [k for k in vars(m) if "sp_ref" in k or "snapshot" in k]
    assert set(vars(params)) == {"sets", "benchmark", "taxes"}


def test_the_shared_reference_dies_with_the_params(builds):
    """No retiene memoria: muerto ``params``, su foto se libera."""
    import gc

    closure = _closure()
    params = _Params()
    _prep(closure).cached(_mp_model(), params, closure, "base")
    assert sp_reference._SHARED_SP_REF
    del params
    gc.collect()
    assert not sp_reference._SHARED_SP_REF
