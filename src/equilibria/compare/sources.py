"""Fuentes: leen una solucion y la devuelven como :class:`Solution`.

Toda fuente pide que se DECLARE de donde sale el periodo: ``period_dim`` (posicion
de la dimension de periodo en el indice, p. ej. ``-1`` para ``xp(r,a,t)``) o
``period`` (un nombre fijo para una solucion sin periodos). Nunca se adivina.
Solo se leen niveles.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from equilibria.compare.solution import CellKey, Index, Solution


def _period_rule(period_dim: int | None, period: str | None) -> None:
    if (period_dim is None) == (period is None):
        raise ValueError("declara exactamente uno de 'period_dim' o 'period'")


def _split(key: Index, period_dim: int | None, period: str | None) -> CellKey:
    if period is not None:
        return period, key
    assert period_dim is not None
    n = len(key)
    if not -n <= period_dim < n:
        raise ValueError(f"period_dim={period_dim} fuera del indice {key}")
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
            raise ValueError(f"{var}: indices de distinto largo ({ndim[var]} y {n})")
        if key in out:
            raise ValueError(f"{var}: celda repetida {key}")
        out[key] = float(raw_val)
    if var not in ndim and full_ndim is not None:
        ndim[var] = full_ndim - (1 if period is None else 0)
    levels[var] = out


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
    por defecto, todas. GAMS no escribe los registros con todos los atributos en su
    default, asi que una celda con nivel 0 puede no estar: queda sin pareja.
    """
    from equilibria.babel.gdx.reader import (
        get_symbol,
        get_variables,
        read_gdx,
        read_variable_values,
    )

    _period_rule(period_dim, period)
    data = read_gdx(path)
    names = (
        [v["name"] for v in get_variables(data)]
        if variables is None
        else list(variables)
    )
    levels: dict[str, dict[CellKey, float]] = {}
    ndim: dict[str, int] = {}
    skipped: dict[str, str] = {}
    for name in names:
        sym = get_symbol(data, name)
        if sym is None or sym["type"] != 2:
            raise KeyError(f"{path}: no hay una variable {name!r}")
        why = _no_period_dim(int(sym["dimension"]), period_dim)
        if why:
            if variables is not None:
                raise ValueError(f"{path}: {name!r} {why}")
            skipped[name] = why
            continue
        records = read_variable_values(data, name)
        _add(
            levels,
            ndim,
            name,
            ((k, rec["level"]) for k, rec in records.items()),
            period_dim,
            period,
            full_ndim=int(sym["dimension"]),
        )
    return Solution(levels, ndim, str(path), skipped)


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
                raise KeyError(f"el modelo no tiene una variable {name!r}")
            comps.append(c)
    levels: dict[str, dict[CellKey, float]] = {}
    ndim: dict[str, int] = {}
    skipped: dict[str, str] = {}
    for comp in comps:
        why = _no_period_dim(comp.dim(), period_dim)
        if why:
            if variables is not None:
                raise ValueError(f"{comp.name!r} {why}")
            skipped[comp.name] = why
            continue
        _add(
            levels,
            ndim,
            comp.name,
            ((k, comp[k].value) for k in comp),
            period_dim,
            period,
            full_ndim=comp.dim(),
        )
    return Solution(levels, ndim, "pyomo", skipped)
