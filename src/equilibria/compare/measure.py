"""Medida: diferencia relativa con piso absoluto, y resumenes.

``d = |A - B| / max(|ref|, abs_floor)``, con ``ref`` el lado de referencia. Una
celda pasa si ``d <= tol.rel`` de su variable.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any


def rel_diff(va: float, vb: float, ref: float, abs_floor: float) -> float:
    return abs(va - vb) / max(abs(ref), abs_floor)


def summarize(
    ds: Sequence[float],
    passed: Sequence[bool],
    cells: Sequence[Any],
    thresholds: Sequence[float],
) -> dict[str, Any]:
    """Resumen de un grupo de celdas: conteo, % que pasa, % dentro de cada umbral,
    mediana, maximo y la celda del maximo (``cells[i]`` acompana a ``ds[i]``).
    NaN cuenta como fuera de todo umbral y como maximo."""
    n = len(ds)
    if n == 0:
        return {
            "cells": 0,
            "passed": 0,
            "pass_pct": None,
            "pct_within": dict.fromkeys(thresholds),
            "median": None,
            "max": None,
            "max_cell": None,
        }
    key = [math.inf if math.isnan(d) else d for d in ds]
    imax = max(range(n), key=lambda i: key[i])
    return {
        "cells": n,
        "passed": sum(passed),
        "pass_pct": 100.0 * sum(passed) / n,
        "pct_within": {t: 100.0 * sum(d <= t for d in ds) / n for t in thresholds},
        "median": statistics.median(key),
        "max": ds[imax],
        "max_cell": cells[imax],
    }
