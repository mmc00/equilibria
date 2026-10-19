"""Niveles de una solucion GEMPACK leidos directo del ``.sl4``, sin sltoht.

El ``.sl4`` es un archivo HAR. Ademas de los cambios %, trae los niveles pre (LEVB) y
post (LEVA) de cada variable que los tiene, en un solo arreglo con punteros. Este
modulo los ubica por variable; :mod:`equilibria.babel.har.sl4_levels` es la
convencion del ``.har`` de niveles que escribe ``export_sl4_levels.py``.
"""

from __future__ import annotations

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
_SL4_HEADERS = (
    "VARS",  # nombres de las variables
    "VCNA",  # cuantos argumentos tiene cada una
    "VCAR",  # numero de conjunto (1-based en STNM) de cada argumento, seguidos
    "SSZ",  # tamano de cada conjunto
    "ELAD",  # donde empiezan sus elementos en STEL (1-based)
    "STEL",  # elementos de todos los conjuntos
    "LVOK",  # "y" si la variable tiene nivel
    "PLEV",  # donde empiezan sus niveles en LEVB/LEVA (1-based)
    "LEVB",  # niveles pre-simulacion
    "LEVA",  # niveles post-simulacion
)


def read_sl4_levels(path: str | Path) -> dict[str, Sl4Levels | None]:
    """Niveles de cada variable de una solucion GEMPACK, leidos del ``.sl4``.

    ``None`` = la variable no tiene nivel en el ``.sl4`` (``LVOK`` != "y"). Las celdas
    van en el orden de FORTRAN (primer indice mas rapido), como el ``sl4dump``. Los
    niveles son REAL*4 en el archivo. ``ValueError`` si falta una cabecera o un
    puntero sale de ``LEVB``/``LEVA``.
    """
    from equilibria.babel.har.reader import read_har

    headers = read_har(str(path))
    missing = [h for h in _SL4_HEADERS if h not in headers]
    if missing:
        raise ValueError(f"{path}: al .sl4 le faltan las cabeceras {missing}")

    def col(name: str) -> np.ndarray:
        return np.asarray(headers[name].array).reshape(-1)

    names = [str(v).strip() for v in col("VARS")]
    n_args, set_of_arg = col("VCNA"), col("VCAR")
    first_arg = np.concatenate(([0], np.cumsum(n_args)[:-1]))
    size, first_elem, elements = col("SSZ"), col("ELAD"), col("STEL")
    has_level, first_level = col("LVOK"), col("PLEV")
    pre_all = col("LEVB").astype(np.float64)
    post_all = col("LEVA").astype(np.float64)

    out: dict[str, Sl4Levels | None] = {}
    for i, name in enumerate(names):
        if str(has_level[i]).strip() != HAS_LEVEL:
            out[name] = None
            continue
        a0 = int(first_arg[i])
        sets = [int(s) - 1 for s in set_of_arg[a0 : a0 + int(n_args[i])]]
        set_elements = [
            [
                str(e).strip()
                for e in elements[first_elem[s] - 1 : first_elem[s] - 1 + size[s]]
            ]
            for s in sets
        ]
        shape = tuple(int(size[s]) for s in sets)
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
