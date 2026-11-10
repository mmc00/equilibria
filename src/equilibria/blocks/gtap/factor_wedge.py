"""The factor tax/subsidy wedge (r, f, a): the one source.

pfa = pf*(1 + fcttx + fctts). The data satisfy EVFP = EVFB + FTRV + FBEP, with
the subsidy FBEP <= 0. Each ``va_subsidy_basis`` reproduces one GAMS reference
(#187, #191, #193):

- "gams" (default): the official getData.gms (``fbep=zero``): fbep = 0 and
  ftrv = EVFP - EVFB, one net tax that holds the subsidy; fctts = 0.
- "gempack": the HAR's headers (``fbep=header``): fcttx = FTRV/EVFB, with the
  FTRV header (0 where the data subsidise with no tax), and fctts = +FBEP/EVFB.

Both give the data's wedge, 1 + fcttx + fctts = EVFP/EVFB. Before #191, "gams"
took fctts = -FBEP/EVFB (cal.gms's sign on the HAR's negative FBEP), which
booked the subsidy as a tax.
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


def _get(table: Any, key: tuple[str, str, str]) -> float:
    return float((table or {}).get(key, 0.0) or 0.0)


def factor_wedge_values(
    benchmark: Any, r: str, f: str, a: str, basis: VaSubsidyBasis = "gams"
) -> tuple[float, float]:
    """(tax, subsidy) in value: (EVFP - EVFB, 0) with gams, (FTRV, FBEP) with gempack."""
    key = (r, f, a)
    evfp, evfb = _get(benchmark.vfm, key), _get(benchmark.evfb, key)
    fbep = _get(benchmark.fbep, key)
    if check_basis(basis) == "gams":
        return evfp - evfb, 0.0
    # the FTRV header where the data have one; else the identity, which is 0
    # where the cell is only subsidised (the loader's EVFP - EVFB holds FBEP)
    har = getattr(benchmark, "ftrv_har", None) or {}
    ftrv = _get(har, key) if key in har else evfp - evfb - fbep
    return ftrv, fbep


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


def _kappa(taxes: Any, r: str, f: str, a: str) -> float:
    if taxes is None:
        return 0.0
    kappa = float(taxes.kappaf_activity.get((r, f, a), 0.0) or 0.0)
    if kappa == 0.0:
        kappa = float(taxes.kappaf.get((r, f), 0.0) or 0.0)
    return kappa


def pf_bench(taxes: Any, r: str, f: str, a: str) -> float:
    """Benchmark factor price pf = 1/(1 - kappaf) (GTAPCalibratedShares calibration)."""
    return max(1.0 / max(1.0 - _kappa(taxes, r, f, a), 1e-12), 1e-12)


def pfa_bench(
    benchmark: Any,
    taxes: Any,
    r: str,
    f: str,
    a: str,
    basis: VaSubsidyBasis = "gams",
) -> float:
    """Benchmark tax-inclusive factor price pfa = pf*(1+fctts+fcttx) (cal.gms), the
    price ``af`` is calibrated with (``calibrate_from_benchmark``)."""
    fcttx, fctts = (
        factor_wedge_rates(benchmark, r, f, a, basis)
        if benchmark is not None
        else (0.0, 0.0)
    )
    return pf_bench(taxes, r, f, a) * max(1.0 + fctts + fcttx, 1e-12)
