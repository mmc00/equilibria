"""Foto y restauracion del estado mutable de un modelo multiperiodo.

Para que ``calibrate_base`` resuelva el settle SOBRE el modelo ya construido en vez
de armar un segundo modelo identico (en 20x41 era una construccion entera, ~45%
de la calibracion). El settle muta el modelo de muchas formas (valores, fijados,
cotas, dominios, filas activas, params mutables, componentes y atributos que agrega
el driver); en vez de perseguir cada sitio, se fotografia todo lo mutable y se
vuelve atras. Lo que no esta aca es inmutable para el driver: la estructura de
indices y las expresiones de cada fila (se guarda la referencia y se repone si
alguien la cambia).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyomo.core.base.component import Component
from pyomo.environ import ConcreteModel, Constraint, Param, Suffix, Var

# Atributos internos de cualquier bloque Pyomo: no se tocan.
_BLOCK_INTERNOS = frozenset(ConcreteModel().__dict__)


@dataclass
class ModelState:
    componentes: dict[str, Any] = field(default_factory=dict)
    activos: list[tuple[Any, bool]] = field(default_factory=list)
    atributos: dict[str, Any] = field(default_factory=dict)
    vars: list[tuple[Any, Any, bool, Any, Any, Any]] = field(default_factory=list)
    filas: list[tuple[Any, bool, Any]] = field(default_factory=list)
    params: list[tuple[Any, Any]] = field(default_factory=list)
    sufijos: list[tuple[Any, list]] = field(default_factory=list)


def _atributos(m: Any) -> dict[str, Any]:
    return {
        k: v
        for k, v in m.__dict__.items()
        if k not in _BLOCK_INTERNOS and not isinstance(v, Component)
    }


def capture(m: Any) -> ModelState:
    """Fotografia lo que el driver puede mutar al resolver ``m``."""
    st = ModelState()
    st.componentes = dict(m.component_map())
    st.activos = [
        (c, c.active) for c in st.componentes.values() if hasattr(c, "active")
    ]
    st.atributos = _atributos(m)
    for v in m.component_objects(Var, descend_into=True):
        for vd in v.values():
            st.vars.append((vd, vd._value, vd._fixed, vd._lb, vd._ub, vd._domain))
    for c in m.component_objects(Constraint, descend_into=True):
        for cd in c.values():
            st.filas.append((cd, cd._active, cd._expr))
    for p in m.component_objects(Param, descend_into=True):
        if p.mutable:
            for pd in p.values():
                st.params.append((pd, pd._value))
    for s in m.component_objects(Suffix, descend_into=True):
        # Suffix es un ComponentMap (claves no hasheables): se guarda como lista.
        st.sufijos.append((s, list(s.items())))
    return st


def restore(m: Any, st: ModelState) -> None:
    """Deja ``m`` como estaba en ``capture``."""
    actuales = dict(m.component_map())
    for nombre, obj in actuales.items():
        if st.componentes.get(nombre) is not obj:
            m.del_component(nombre)
    for nombre, obj in st.componentes.items():
        if m.component(nombre) is None:
            m.add_component(nombre, obj)
    for c, activo in st.activos:
        if c.active != activo:
            c.activate() if activo else c.deactivate()
    for k in set(_atributos(m)) - set(st.atributos):
        delattr(m, k)
    for k, v in st.atributos.items():
        setattr(m, k, v)
    for vd, val, fijo, lb, ub, dom in st.vars:
        vd._value = val
        vd._fixed = fijo
        vd._lb = lb
        vd._ub = ub
        vd._domain = dom
    for cd, activo, expr in st.filas:
        cd._active = activo
        cd._expr = expr
    for pd, val in st.params:
        pd._value = val
    for s, contenido in st.sufijos:
        s.clear()
        for k, v in contenido:
            s[k] = v
