"""Los periodos del modelo multiperiodo, como el ``loop(tsim)`` del compStat de GAMS.

``base`` son los niveles de cal.gms, ``check`` replica el benchmark y ``shock`` lleva el
shock (y el cierre de ``@overwrite``). Vive en ``blocks/`` para que lo lean los bloques
y los templates sin que los bloques importen hacia arriba.
"""

from __future__ import annotations

BASE = "base"
CHECK = "check"
SHOCK = "shock"
PERIODS = (BASE, CHECK, SHOCK)
