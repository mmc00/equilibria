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
from typing import Any, cast

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


def _params(
    params: Mapping[str, Mapping[Any, Any]] | None, origin: str
) -> dict[str, dict[Index, float]]:
    """``{nombre: {indices: valor}}`` de una fuente, validado (sin periodo)."""
    out: dict[str, dict[Index, float]] = {}
    for name, cells in (params or {}).items():
        table: dict[Index, float] = {}
        width: int | None = None
        for raw_key, raw_val in cells.items():
            key = _as_index(raw_key)
            if width is None:
                width = len(key)
            elif len(key) != width:
                raise CompareError(
                    f"{origin}: parametro {name!r} con indices de distinto largo"
                )
            table[key] = float(raw_val)
        out[str(name)] = table
    return out


def from_dict(
    data: Mapping[str, Mapping[Any, Any]],
    *,
    period_dim: int | None = None,
    period: str | None = None,
    origin: str = "dict",
    params: Mapping[str, Mapping[Any, Any]] | None = None,
) -> Solution:
    """``{var: {indices: nivel}}``. Un indice escalar vale como tupla de uno.

    ``params`` (``{nombre: {indices: valor}}``) son parametros de esta misma fuente
    para el ``divide_by`` del mapeo."""
    _period_rule(period_dim, period)
    levels: dict[str, dict[CellKey, float]] = {}
    ndim: dict[str, int] = {}
    for var, cells in data.items():
        _add(levels, ndim, str(var), cells.items(), period_dim, period)
    return Solution(levels, ndim, origin, params=_params(params, origin))


def from_gdx(
    path: str | Path,
    *,
    period_dim: int | None = None,
    period: str | None = None,
    variables: Iterable[str] | None = None,
    params: Iterable[str] | None = None,
) -> Solution:
    """Niveles (``.L``) de las variables de un ``.gdx`` de GAMS.

    ``params`` nombra los parametros del GDX que se leen para el ``divide_by`` del
    mapeo (``Mapping.params(lado)``).

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
        get_parameters,
        get_variables,
        read_gdx,
        read_parameter_values,
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
    gdx_params = {p["name"] for p in get_parameters(data)}
    for name in params or ():
        if name not in gdx_params:
            raise CompareError(f"{path}: no hay un parametro {name!r}")
    table = _params(
        {n: read_parameter_values(data, n) for n in params or ()}, str(path)
    )
    return Solution(levels, ndim, str(path), skipped, absent_is_zero=True, params=table)


def from_pyomo(
    model: Any,
    *,
    period_dim: int | None = None,
    period: str | None = None,
    variables: Iterable[str] | None = None,
    params: Mapping[str, Mapping[Any, Any]] | Iterable[str] | None = None,
) -> Solution:
    """Valores de las variables (``pyo.Var``) de un modelo Pyomo resuelto.

    Una variable sin valor (``None``) no aporta celda. ``variables`` limita la
    lectura; por defecto, todas las ``Var`` activas del modelo.

    ``params``, para el ``divide_by`` del mapeo: nombres de ``Param`` del modelo, o
    las tablas ya calculadas por el codigo del modelo cuando no son ``Param`` (p. ej.
    ``xscale_data`` de GTAP), como ``{nombre: {indices: valor}}``.
    """
    from pyomo.environ import Param, Var
    from pyomo.environ import value as pyo_value

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
    if params is None or isinstance(params, Mapping):
        table = _params(cast("Mapping[str, Mapping[Any, Any]] | None", params), "pyomo")
    else:
        found: dict[str, Mapping[Any, Any]] = {}
        for name in params:
            c = model.find_component(name)
            if c is None or c.ctype is not Param:
                raise CompareError(f"el modelo no tiene un parametro {name!r}")
            found[name] = {k: pyo_value(c[k]) for k in c}
        table = _params(found, "pyomo")
    return Solution(levels, ndim, "pyomo", skipped, params=table)


_CONVERSION_HINT = (
    "Convierte la solucion a un .har de NIVELES (LEVB/LEVA) con "
    "scripts/gtap/export_sl4_levels.py (usa `sltoht -SHL -SIC`)."
)


def from_har(
    path: str | Path,
    *,
    variables: Iterable[str] | None = None,
) -> Solution:
    """Niveles de una solucion GEMPACK desde su ``.har`` de NIVELES.

    Es la salida de ``scripts/gtap/export_sl4_levels.py``; la convencion vive en
    :mod:`equilibria.babel.har.sl4_levels`. Periodos: ``pre`` (cabeceras LEVB) y
    ``post`` (LEVA); un ``.har`` solo con LEVA trae solo ``post``.

    Una variable sin nivel (listada en ``NOLV``, o con el centinela 1e10 en TODAS sus
    celdas en un ``.har`` viejo) no aporta celdas: queda en ``Solution.no_level`` y,
    si el mapeo la pide, el reporte la declara excluida. El centinela en solo algunas
    celdas es error. Un ``.sl4`` o un ``.har`` de cambios % tambien: dicen como
    convertirlos. ``variables`` limita la lectura; por defecto, todas.
    """
    import numpy as np

    from equilibria.babel.har import sl4_levels as lv
    from equilibria.babel.har.reader import read_har

    path = Path(path)
    if path.suffix.lower() == ".sl4":
        raise CompareError(f"{path}: un .sl4 no se lee directo. {_CONVERSION_HINT}")
    headers = read_har(str(path))
    if all(h in headers for h in lv.PERCENT_CHANGE_HEADERS):
        raise CompareError(
            f"{path}: es un .har de CAMBIOS % de sltoht (trae "
            f"{'/'.join(lv.PERCENT_CHANGE_HEADERS)}), no de niveles. {_CONVERSION_HINT}"
        )

    no_level: dict[str, str] = {}
    if lv.NO_LEVEL_HEADER in headers:
        for name in headers[lv.NO_LEVEL_HEADER].array.ravel():
            no_level[str(name).strip()] = (
                f"sin nivel en GEMPACK (cabecera {lv.NO_LEVEL_HEADER} de {path.name})"
            )

    def cells(var: str, ha: Any, period: str) -> dict[CellKey, float]:
        arr = np.asarray(ha.array, dtype=np.float64)
        sentinel = arr == lv.NO_LEVEL
        if sentinel.any() and not sentinel.all():
            raise CompareError(
                f"{path}: {var!r} trae el centinela 1e10 en {int(sentinel.sum())} de "
                f"{sentinel.size} celdas; GEMPACK da nivel por variable, no por celda"
            )
        if not ha.set_elements:  # escalar: el HAR lo guarda como (1,) sin conjuntos
            return {(period, ()): float(arr.ravel()[0])}
        out: dict[CellKey, float] = {}
        for pos in np.ndindex(arr.shape):
            index = tuple(str(ha.set_elements[d][i]) for d, i in enumerate(pos))
            out[(period, index)] = float(arr[pos])
        return out

    leva = {
        lv.variable_name(ha.long_name): (hid, ha)
        for hid, ha in headers.items()
        if lv.is_leva_header(hid) and np.asarray(ha.array).dtype.kind in "fiu"
    }
    wanted = list(leva) if variables is None else list(variables)
    for name in wanted:
        if name not in leva and name not in no_level:
            raise CompareError(f"{path}: no hay una variable {name!r}")

    levels: dict[str, dict[CellKey, float]] = {}
    ndim: dict[str, int] = {}
    for name in wanted:
        if name in no_level:
            continue
        hid, ha = leva[name]
        post = cells(name, ha, "post")
        if all(v == lv.NO_LEVEL for v in post.values()):
            no_level[name] = f"sin nivel en GEMPACK (centinela 1e10 en {path.name})"
            continue
        levb = headers.get(lv.levb_header(hid))
        pre = {} if levb is None else cells(name, levb, "pre")
        levels[name] = {**pre, **post}
        ndim[name] = len(ha.set_elements)
    return Solution(levels, ndim, str(path), no_level=no_level)
