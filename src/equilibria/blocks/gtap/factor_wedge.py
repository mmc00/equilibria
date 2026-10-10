"""Cuna de impuestos/subsidios a los factores (r, f, a): la unica fuente.

pfa = pf*(1 + fcttx + fctts), con fcttx = FTRV/EVFB (impuesto) y fctts segun el
signo del subsidio (``va_subsidy_basis``):

- "gams" (default): fctts = -FBEP/EVFB, como cal.gms. FBEP es <= 0, asi que el
  subsidio SUBE pfa.
- "gempack": fctts = +FBEP/EVFB, la identidad de los datos EVFP = EVFB + FTRV +
  FBEP (cierra a 3e-8 en gtap7_15x10; la de GAMS falla en 0,46%): 1 + fcttx +
  fctts = EVFP/EVFB, la cuna neta de GTAPv7.jl y GEMPACK.

Bajo "gams" las cuentas son las mismas operaciones de punto flotante que antes
(-1.0*fbep == -fbep y ftrv + (-fbep) == ftrv - fbep), asi que el cierre por
defecto queda identico bit a bit.
"""

from __future__ import annotations

from typing import Any, Literal

VaSubsidyBasis = Literal["gams", "gempack"]
VA_SUBSIDY_BASES: tuple[VaSubsidyBasis, ...] = ("gams", "gempack")


def va_subsidy_basis(params: Any) -> VaSubsidyBasis:
    """El signo del subsidio que lleva ``params`` ("gams" si no tiene)."""
    return check_basis(getattr(params, "va_subsidy_basis", "gams"))


def check_basis(basis: str) -> VaSubsidyBasis:
    if basis not in VA_SUBSIDY_BASES:
        raise ValueError(
            f"va_subsidy_basis={basis!r}: tiene que ser uno de {VA_SUBSIDY_BASES}"
        )
    return basis  # type: ignore[return-value]


def factor_wedge_values(
    benchmark: Any, r: str, f: str, a: str, basis: VaSubsidyBasis = "gams"
) -> tuple[float, float]:
    """(impuesto, subsidio) en valor: (FTRV, -FBEP) con gams, (FTRV, +FBEP) con gempack."""
    sign = 1.0 if check_basis(basis) == "gempack" else -1.0
    ftrv = float(benchmark.ftrv.get((r, f, a), 0.0) or 0.0)
    fbep = float(benchmark.fbep.get((r, f, a), 0.0) or 0.0)
    return ftrv, sign * fbep


def factor_wedge_rates(
    benchmark: Any, r: str, f: str, a: str, basis: VaSubsidyBasis = "gams"
) -> tuple[float, float]:
    """(fcttx, fctts) del factor (r, f, a); (0, 0) si EVFB <= 0."""
    evfb = float(benchmark.evfb.get((r, f, a), 0.0) or 0.0)
    if evfb <= 0.0:
        return 0.0, 0.0
    tx, ts = factor_wedge_values(benchmark, r, f, a, basis)
    return tx / evfb, ts / evfb


def factor_wedge_rate(
    benchmark: Any, r: str, f: str, a: str, basis: VaSubsidyBasis = "gams"
) -> float:
    """fcttx + fctts del factor (r, f, a); 0 si EVFB <= 0."""
    evfb = float(benchmark.evfb.get((r, f, a), 0.0) or 0.0)
    if evfb <= 0.0:
        return 0.0
    tx, ts = factor_wedge_values(benchmark, r, f, a, basis)
    return (tx + ts) / evfb
