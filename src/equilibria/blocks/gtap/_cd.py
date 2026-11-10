"""Exact Cobb-Douglas price duals for CES nests at sigma = 1 (``unit_sigma="gempack"``).

A CES price dual ``P^(1-s) = sum_i alpha_i q_i^(1-s)`` reads ``1 = 1`` at s = 1 and
stops pinning ``P`` (GAMS #192, where cal.gms moves every such elasticity to 1.01).
Its limit at s = 1 is the Cobb-Douglas dual, written here in log form:

    log(P) [+ log(shift)] == c + sum_i w_i * log(q_i)

with ``w_i = alpha_i / S``, ``S = sum_i alpha_i`` (1 up to rounding when ``alpha`` was
calibrated at s = 1) and ``c = log(P0) - sum_i w_i log(q0_i)`` from the base prices of
the calibration that built ``alpha`` (``_derived_params.cd_coefficients``). ``q_i`` is
the input price over its efficiency shift; a shift on ``P`` (``axp``, ``axi``) moves to
the left as its log.

Log form because the family recorder splits a row into additive terms
(``family/recorder.py``): each ``w_i * log(q_i)`` is one term. ``log`` is imported at
module level so ``backends/family_backend`` swaps it for the recording one.

The switch is ``GTAPClosureConfig.unit_sigma``, carried on the parameters
(``GTAPParameters.set_unit_sigma``) the way ``va_subsidy_basis`` is.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pyomo.environ import log

UnitSigma = Literal["gams", "gempack"]
UNIT_SIGMA_MODES: tuple[str, ...] = ("gams", "gempack")


def check_unit_sigma(mode: str) -> UnitSigma:
    if mode not in UNIT_SIGMA_MODES:
        raise ValueError(f"unit_sigma={mode!r}: expected one of {UNIT_SIGMA_MODES}")
    return mode  # type: ignore[return-value]


def unit_sigma(params: Any) -> UnitSigma:
    """The closure's ``unit_sigma`` as stamped on ``params`` (default ``"gams"``)."""
    return check_unit_sigma(getattr(params, "unit_sigma", "gams"))


def exact_cd(params: Any, sigma: float) -> bool:
    """True when this nest is written as exact Cobb-Douglas: ``unit_sigma="gempack"``
    and the elasticity is exactly 1 (a 1.01 stand-in stays CES)."""
    return float(sigma) == 1.0 and unit_sigma(params) == "gempack"


def cd_price_row(
    price: Any,
    terms: Sequence[tuple[float, Any]],
    c: float,
    shifts: Sequence[Any] = (),
):
    """``log(price) + sum log(shift) == c + sum_i w_i log(q_i)`` for ``terms`` =
    ``[(w_i, q_i), ...]`` (weights already normalised)."""
    lhs = log(price)
    for s in shifts:
        lhs = lhs + log(s)
    rhs = c
    for w, q in terms:
        rhs = rhs + w * log(q)
    return lhs == rhs
