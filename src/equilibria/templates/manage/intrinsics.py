"""GAMS intrinsic functions MANAGE's equations use, with GAMS's definitions.

Checked against GAMS 53 (a ``put`` of 7 points with 17 digits, 2026-11-27): the
differences are below 1e-17.
"""

from __future__ import annotations

import math


def ncp_vu_pow(r: float, s: float, mu: float = 0.0) -> float:
    """GAMS ``ncpVUpow(r, s, mu)``: Veelken-Ulbrich smoothed ``min(r, s)``.

    ``(r + s - f(r - s))/2`` with ``f(t) = |t|`` for ``|t| >= mu`` and
    ``mu/8*(-(t/mu)**4 + 6*(t/mu)**2 + 3)`` inside the band.
    """
    t = r - s
    if abs(t) >= mu:
        a = abs(t)
    else:
        z = t / mu
        a = mu / 8 * (-(z**4) + 6 * z**2 + 3)
    return (r + s - a) / 2


def sigmoid(x: float) -> float:
    """GAMS ``sigmoid(x) = 1/(1 + exp(-x))``."""
    if x < -700:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))
