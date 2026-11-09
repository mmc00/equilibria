"""Niveles de una solucion GEMPACK leidos directo del ``.sl4``, sin sltoht.

El ``.sl4`` es un archivo HAR. Ademas de los cambios %, trae los niveles pre (LEVB) y
post (LEVA) de cada variable que los tiene, en un solo arreglo con punteros. Este
modulo los ubica por variable; :mod:`equilibria.babel.har.sl4_levels` es la
convencion del ``.har`` de niveles que escribe ``export_sl4_levels.py``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HAS_LEVEL = "y"  # LVOK de una variable con nivel


@dataclass(frozen=True)
class Sl4Levels:
    """Niveles de una variable del ``.sl4``: pre (LEVB) y post (LEVA), con la forma
    de sus conjuntos. Un escalar tiene ``set_elements == []`` y forma ``()``."""

    set_elements: list[list[str]]
    pre: np.ndarray
    post: np.ndarray


# Cabeceras del .sl4 que hacen falta para ubicar los niveles de cada variable. Todas
# las por-variable son de la lista de la SOLUCION (condensadas + backsolved); las de
# todo el modelo (VCSP/VCSN, VCNI) traen ademas las omitidas y no se alinean con VARS.
_LAYOUT_HEADERS = (
    "VARS",  # nombres de las variables
    "VCNA",  # cuantos argumentos tiene cada una
    "VCAR",  # numero de conjunto (1-based en STNM) de cada argumento, seguidos
    "SSZ",  # tamano de cada conjunto
    "ELAD",  # donde empiezan sus elementos en STEL (1-based)
    "STEL",  # elementos de todos los conjuntos
)
_SL4_HEADERS = (
    *_LAYOUT_HEADERS,
    "LVOK",  # "y" si la variable tiene nivel
    "PLEV",  # donde empiezan sus niveles en LEVB/LEVA (1-based)
    "LEVB",  # niveles pre-simulacion
    "LEVA",  # niveles post-simulacion
)


def _layout(
    path: str | Path, extra: tuple[str, ...] = ()
) -> tuple[dict, list[str], list[list[list[str]]]]:
    """Headers of the ``.sl4`` and, per variable of ``VARS``, its name and the element
    labels of each of its sets (empty for a scalar)."""
    from equilibria.babel.har.reader import read_har

    headers = read_har(str(path))
    missing = [h for h in (*_LAYOUT_HEADERS, *extra) if h not in headers]
    if missing:
        raise ValueError(f"{path}: al .sl4 le faltan las cabeceras {missing}")
    col = _col(headers)
    names = [str(v).strip() for v in col("VARS")]
    n_args, set_of_arg = col("VCNA"), col("VCAR")
    first_arg = np.concatenate(([0], np.cumsum(n_args)[:-1]))
    size, first_elem, elements = col("SSZ"), col("ELAD"), col("STEL")
    sets_of: list[list[list[str]]] = []
    for i in range(len(names)):
        a0 = int(first_arg[i])
        sets = [int(s) - 1 for s in set_of_arg[a0 : a0 + int(n_args[i])]]
        sets_of.append(
            [
                [
                    str(e).strip()
                    for e in elements[first_elem[s] - 1 : first_elem[s] - 1 + size[s]]
                ]
                for s in sets
            ]
        )
    return headers, names, sets_of


def _col(headers: dict) -> Callable[[str], np.ndarray]:
    return lambda name: np.asarray(headers[name].array).reshape(-1)


def read_sl4_levels(path: str | Path) -> dict[str, Sl4Levels | None]:
    """Niveles de cada variable de una solucion GEMPACK, leidos del ``.sl4``.

    ``None`` = la variable no tiene nivel en el ``.sl4`` (``LVOK`` != "y"). Las celdas
    van en el orden de FORTRAN (primer indice mas rapido), como el ``sl4dump``. Los
    niveles son REAL*4 en el archivo. ``ValueError`` si falta una cabecera o un
    puntero sale de ``LEVB``/``LEVA``.
    """
    headers, names, sets_of = _layout(path, ("LVOK", "PLEV", "LEVB", "LEVA"))
    col = _col(headers)
    has_level, first_level = col("LVOK"), col("PLEV")
    pre_all = col("LEVB").astype(np.float64)
    post_all = col("LEVA").astype(np.float64)

    out: dict[str, Sl4Levels | None] = {}
    for i, name in enumerate(names):
        if str(has_level[i]).strip() != HAS_LEVEL:
            out[name] = None
            continue
        set_elements = sets_of[i]
        shape = tuple(len(e) for e in set_elements)
        n = int(np.prod(shape))
        p0 = int(first_level[i]) - 1
        if p0 < 0 or p0 + n > pre_all.size:
            raise ValueError(
                f"{path}: los niveles de {name!r} ({n} celdas desde {p0 + 1}) salen "
                f"de LEVB/LEVA ({pre_all.size})"
            )
        out[name] = Sl4Levels(
            set_elements,
            pre_all[p0 : p0 + n].reshape(shape, order="F"),
            post_all[p0 : p0 + n].reshape(shape, order="F"),
        )
    return out


@dataclass(frozen=True)
class Sl4Change:
    """Cumulative change of one variable of the ``.sl4`` (CUMS): a % change for
    ``kind == "p"`` and a plain change for ``"c"``. ``values`` has the shape of its sets
    (a scalar has ``set_elements == []`` and shape ``()``)."""

    set_elements: list[list[str]]
    kind: str
    values: np.ndarray


@dataclass(frozen=True)
class Sl4Changes:
    """Solution of an ``.sl4``: ``changes`` of every variable that could be placed, and
    ``unplaced[var]`` = why one could not (partly exogenous: the ``.sl4`` stores which
    components are endogenous only as a compressed list we do not decode)."""

    changes: dict[str, Sl4Change]
    unplaced: dict[str, str]


_SHOCK_RE = re.compile(
    r"(?im)^\s*shock\s+(\w+)\s*(?:\(([^)]*)\))?\s*=\s*([-+0-9.eE]+)\s*;"
)


def _shocks(lines: np.ndarray) -> dict[str, dict[tuple[str, ...], float]]:
    """``Shock v("a","b") = x;`` statements of the command file kept in ``CMDF``."""
    out: dict[str, dict[tuple[str, ...], float]] = {}
    for var, cell, val in _SHOCK_RE.findall("\n".join(str(x) for x in lines)):
        key = tuple(x.strip().strip("\"'") for x in cell.split(",")) if cell else ()
        out.setdefault(var, {})[key] = float(val)
    return out


def read_sl4_changes(path: str | Path) -> Sl4Changes:
    """Cumulative changes of every variable of a GEMPACK solution, read from the ``.sl4``.

    ``CUMS`` holds only the cumulatively-retained endogenous components, ``CMND`` per
    variable, from ``PCUM`` (1-based), in FORTRAN order. A fully exogenous variable
    (``CMND == 0``) changes by 0 except where the command file shocks it (``Shock
    v("a","b") = x;`` kept in ``CMDF``; any other shock form is an error). A variable
    with some but not all components endogenous cannot be placed and is reported in
    ``unplaced``.
    """
    headers, names, sets_of = _layout(
        path, ("CMND", "PCUM", "CUMS", "VCTP", "VNCP", "SHCK", "CMDF")
    )
    col = _col(headers)
    n_comp, n_cum, start = col("VNCP"), col("CMND"), col("PCUM")
    kind, n_shock = col("VCTP"), col("SHCK")
    cums = col("CUMS").astype(np.float64)
    shocks = _shocks(col("CMDF"))
    out: dict[str, Sl4Change] = {}
    unplaced: dict[str, str] = {}
    for i, name in enumerate(names):
        shape = tuple(len(e) for e in sets_of[i])
        n = int(np.prod(shape))
        if n != int(n_comp[i]):
            raise ValueError(
                f"{path}: {name!r} has {n} cells but VNCP says {n_comp[i]}"
            )
        k = str(kind[i]).strip()
        if n_cum[i] == n and n > 0:
            p0 = int(start[i]) - 1
            if p0 < 0 or p0 + n > cums.size:
                raise ValueError(f"{path}: the results of {name!r} leave CUMS")
            values = cums[p0 : p0 + n].reshape(shape, order="F")
        elif n_cum[i] == 0 or n == 0:
            values = np.zeros(shape)
            for cell, x in shocks.get(name, {}).items():
                values[tuple(sets_of[i][d].index(e) for d, e in enumerate(cell))] = x
            placed = len(shocks.get(name, {}))
            if int(n_shock[i]) != 0 and placed == 0:
                raise ValueError(
                    f"{path}: {name!r} is shocked but no Shock statement was found"
                )
        else:
            unplaced[name] = (
                f"{int(n_cum[i])} of {n} components endogenous: the list of which ones "
                "is not decoded"
            )
            continue
        out[name] = Sl4Change(sets_of[i], k, values)
    return Sl4Changes(out, unplaced)
