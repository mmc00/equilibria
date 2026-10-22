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

:func:`equilibria.babel.har.sl4.read_sl4_levels` lee los mismos niveles directo del
``.sl4``, sin sltoht.
"""

from __future__ import annotations

import re

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
