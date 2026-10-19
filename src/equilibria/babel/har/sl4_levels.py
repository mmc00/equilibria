"""Convencion del ``.har`` de NIVELES de una solucion GEMPACK (``sl4levels_*.har``).

Lo escribe ``scripts/gtap/export_sl4_levels.py`` y lo lee
:func:`equilibria.compare.from_har`. Esta es la UNICA definicion de la convencion:

* LEVA (post-simulacion): cabecera = el id numerico del ``sl4dump`` (``"0052"``),
  ``long_name`` copiado del ``sl4dump`` (``"qo # output of activity ... #"``).
* LEVB (pre-simulacion): cabecera ``"B"`` + los 3 ultimos digitos de ese id
  (``"B052"``) y ``long_name`` = ``LEVB_PREFIX`` + el del ``sl4dump``.
* Una variable SIN nivel en el SL4 (centinela ``NO_LEVEL`` en las dos columnas) no
  tiene LEVA ni LEVB: su nombre va, uno por elemento, en la cabecera de caracteres
  ``NO_LEVEL_HEADER`` (siempre escrita, vacia si no hay).
* Un ``.har`` sin ``B***`` ni ``NOLV`` es un fixture LEVA-only anterior a #127, que
  puede traer el centinela 1e10 como si fuera un nivel.
* Un ``.har`` de CAMBIOS % (``sl4dump_*.har``, salida HAR de sltoht) trae las
  cabeceras ``PERCENT_CHANGE_HEADERS``; uno de niveles, ninguna (medido sobre los
  fixtures: 73 ``sl4dump_*`` las traen todas, 7 ``sl4levels_*`` ninguna).

:func:`read_sl4_levels` lee los mismos niveles directo del ``.sl4``, sin sltoht.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from equilibria.babel.har.wire import LONG_NAME_WIDTH

NO_LEVEL = 1.0e10  # centinela de GEMPACK: "esta variable no tiene nivel"
NO_LEVEL_HEADER = "NOLV"
NO_LEVEL_LONG_NAME = "variables with NO level in the SL4 (sentinel 1e10): no LEVA/LEVB"
LEVB_PREFIX = "LEVB "
PERCENT_CHANGE_HEADERS = ("XXCR", "XXCD", "XXCP", "XXTD", "XXTT")
_LEVA_ID_RE = re.compile(r"^0\d{3}$")
_LEVB_ID_RE = re.compile(r"^B\d{3}$")


def is_leva_header(hid: str) -> bool:
    return bool(_LEVA_ID_RE.match(hid))


def is_levb_header(hid: str) -> bool:
    return bool(_LEVB_ID_RE.match(hid))


def levb_header(hid: str) -> str:
    """Cabecera LEVB de la cabecera LEVA ``hid``: "B" + id[1:] ("0052" -> "B052")."""
    if not is_leva_header(hid):
        raise ValueError(
            f"sl4dump header {hid!r} is not a numeric id 0000-0999; "
            f"cannot derive its LEVB header 'B'+id[1:]"
        )
    return "B" + hid[1:]


def levb_long_name(long_name: str) -> str:
    """``long_name`` de LEVB: ``LEVB_PREFIX`` + el de LEVA, cortado al limite de 70
    bytes del HAR (sin los blancos finales, como hace el lector)."""
    return (LEVB_PREFIX + long_name)[:LONG_NAME_WIDTH].rstrip()


def variable_name(long_name: str) -> str:
    """Nombre de la variable en un ``long_name`` de LEVA (``"qo # output ... #"`` -> ``"qo"``)."""
    return (long_name or "").split("#", 1)[0].strip()


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
        if str(has_level[i]).strip() != "y":
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
