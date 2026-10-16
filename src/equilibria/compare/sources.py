"""Fuentes: leen una solucion y la devuelven como :class:`Solution`.

Toda fuente pide que se DECLARE de donde sale el periodo: ``period_dim`` (posicion
de la dimension de periodo en el indice, p. ej. ``-1`` para ``xp(r,a,t)``) o
``period`` (un nombre fijo para una solucion sin periodos). Nunca se adivina.
Solo se leen niveles.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from equilibria.compare.errors import CompareError
from equilibria.compare.solution import CellKey, Index, Solution


def _period_rule(period_dim: int | None, period: str | None) -> None:
    if (period_dim is None) == (period is None):
        raise CompareError("declara exactamente uno de 'period_dim' o 'period'")


def _split(key: Index, period_dim: int | None, period: str | None) -> CellKey:
    if period is not None:
        return period, key
    if period_dim is None:
        raise CompareError("declara exactamente uno de 'period_dim' o 'period'")
    n = len(key)
    if not -n <= period_dim < n:
        raise CompareError(f"period_dim={period_dim} fuera del indice {key}")
    i = period_dim % n
    return key[i], key[:i] + key[i + 1 :]


def _no_period_dim(dimension: int, period_dim: int | None) -> str:
    """Motivo por el que una variable de ``dimension`` no tiene la dimension de
    periodo declarada (vacio si la tiene)."""
    if period_dim is None or -dimension <= period_dim < dimension:
        return ""
    return f"tiene {dimension} dimensiones: no hay dimension de periodo {period_dim}"


def _as_index(key: Any) -> Index:
    if isinstance(key, tuple):
        return tuple(str(k) for k in key)
    if key is None:
        return ()
    return (str(key),)


def _add(
    levels: dict[str, dict[CellKey, float]],
    ndim: dict[str, int],
    var: str,
    cells: Iterable[tuple[Any, Any]],
    period_dim: int | None,
    period: str | None,
    full_ndim: int | None = None,
) -> None:
    out: dict[CellKey, float] = {}
    for raw_key, raw_val in cells:
        if raw_val is None:
            continue  # sin nivel: la celda queda ausente (sin pareja), nunca "pasa"
        key = _split(_as_index(raw_key), period_dim, period)
        n = len(key[1])
        if ndim.setdefault(var, n) != n:
            raise CompareError(f"{var}: indices de distinto largo ({ndim[var]} y {n})")
        if key in out:
            raise CompareError(f"{var}: celda repetida {key}")
        out[key] = float(raw_val)
    if var not in ndim and full_ndim is not None:
        ndim[var] = full_ndim - (1 if period is None else 0)
    levels[var] = out


def _read_variables(
    found: Iterable[tuple[str, int, Callable[[], Iterable[tuple[Any, Any]]]]],
    period_dim: int | None,
    period: str | None,
    explicit: bool,
    origin: str,
) -> tuple[dict[str, dict[CellKey, float]], dict[str, int], dict[str, str]]:
    """Lee ``(nombre, dimension, celdas)`` de una fuente con variables declaradas.

    Una variable sin la dimension de periodo es error si se pidio a mano
    (``explicit``); leyendo todo, queda en ``skipped`` con su motivo. ``celdas`` es
    perezosa: solo se lee si la variable se acepta.
    """
    levels: dict[str, dict[CellKey, float]] = {}
    ndim: dict[str, int] = {}
    skipped: dict[str, str] = {}
    for name, dimension, cells in found:
        why = _no_period_dim(dimension, period_dim)
        if why:
            if explicit:
                raise CompareError(f"{origin}: {name!r} {why}")
            skipped[name] = why
            continue
        _add(levels, ndim, name, cells(), period_dim, period, full_ndim=dimension)
    return levels, ndim, skipped


def from_dict(
    data: Mapping[str, Mapping[Any, Any]],
    *,
    period_dim: int | None = None,
    period: str | None = None,
    origin: str = "dict",
) -> Solution:
    """``{var: {indices: nivel}}``. Un indice escalar vale como tupla de uno."""
    _period_rule(period_dim, period)
    levels: dict[str, dict[CellKey, float]] = {}
    ndim: dict[str, int] = {}
    for var, cells in data.items():
        _add(levels, ndim, str(var), cells.items(), period_dim, period)
    return Solution(levels, ndim, origin)


def from_gdx(
    path: str | Path,
    *,
    period_dim: int | None = None,
    period: str | None = None,
    variables: Iterable[str] | None = None,
) -> Solution:
    """Niveles (``.L``) de las variables de un ``.gdx`` de GAMS.

    Lee con :mod:`equilibria.babel.gdx` (sin GAMS). ``variables`` limita la lectura;
    por defecto, todas.

    GAMS no escribe los registros con todos los atributos en su default, o sea con
    nivel 0. Por eso la :class:`Solution` sale con ``absent_is_zero=True``: al
    comparar, una celda que el otro lado tiene y el GDX no vale 0, siempre que la
    variable tenga algun registro en ese periodo (el dominio lo dan las celdas del
    otro lado, no los conjuntos del GDX). Sin ningun registro en el periodo, la
    celda queda sin pareja.
    """
    from equilibria.babel.gdx.reader import (
        get_variables,
        read_gdx,
        read_variable_values,
    )

    _period_rule(period_dim, period)
    data = read_gdx(path)
    gdx_vars = {v["name"]: v for v in get_variables(data)}
    names = list(gdx_vars) if variables is None else list(variables)
    for name in names:
        if name not in gdx_vars:
            raise CompareError(f"{path}: no hay una variable {name!r}")

    def cells(name: str) -> Callable[[], Iterable[tuple[Any, Any]]]:
        return lambda: (
            (k, rec["level"]) for k, rec in read_variable_values(data, name).items()
        )

    levels, ndim, skipped = _read_variables(
        ((n, int(gdx_vars[n]["dimension"]), cells(n)) for n in names),
        period_dim,
        period,
        variables is not None,
        str(path),
    )
    return Solution(levels, ndim, str(path), skipped, absent_is_zero=True)


def from_pyomo(
    model: Any,
    *,
    period_dim: int | None = None,
    period: str | None = None,
    variables: Iterable[str] | None = None,
) -> Solution:
    """Valores de las variables (``pyo.Var``) de un modelo Pyomo resuelto.

    Una variable sin valor (``None``) no aporta celda. ``variables`` limita la
    lectura; por defecto, todas las ``Var`` activas del modelo.
    """
    from pyomo.environ import Var

    _period_rule(period_dim, period)
    if variables is None:
        comps = list(model.component_objects(Var, active=True, descend_into=True))
    else:
        comps = []
        for name in variables:
            c = model.find_component(name)
            if c is None or c.ctype is not Var:
                raise CompareError(f"el modelo no tiene una variable {name!r}")
            comps.append(c)
    levels, ndim, skipped = _read_variables(
        ((c.name, c.dim(), lambda c=c: ((k, c[k].value) for k in c)) for c in comps),
        period_dim,
        period,
        variables is not None,
        "pyomo",
    )
    return Solution(levels, ndim, "pyomo", skipped)
