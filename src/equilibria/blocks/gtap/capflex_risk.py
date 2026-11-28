"""capFlex's ``risk[r]`` at the benchmark, from the calibrated data (no model, no solve).

GAMS ``cal.gms:675-684`` computes it from the initial levels::

    arent = krat * sum(a, (1-kappaf)*pf*xf) / sum(a, xf)      (capital)
    kapEnd = (1-depr)*kstock + xi
    rorc = arent/pi - fdepr
    rore = rorc * (kstock/kapEnd)**RoRFlex
    rorg = sum(r, rore*pi*(xi - depr*kstock)) / sum(r, pi*(xi - depr*kstock))
    risk = rorg / rore

At the benchmark every price is 1 and ``pf*(1-kappaf)*xf = EVOS``, ``xf = EVOS``
(``cal.gms:141-149``), so ``arent = krat = sum(a, EVOS(cap))/VKB``,
``fdepr = depr = VDEP/VKB`` and ``xi`` is the investment purchases
``sum(i, VDIP + VMIP)``. GEMPACK has no ``risk`` level: with ``RORDELTA = 1``
(``gtapv7.tab:2487-2490``) ``rore(r) = rorg + cgdslack(r)`` in % changes keeps the
benchmark ratio, which is this value.
"""

from __future__ import annotations

import math
from typing import Any

_CAPITAL = ("capital", "cap", "k", "kap")


def capflex_risk_from_data(params: Any) -> dict[str, float]:
    """``{region: rorg/rore(r)}`` at the benchmark; ``ValueError`` naming the region
    when its ``rore`` is not finite or is zero (never a fallback to 1)."""
    sets = params.sets
    b = params.benchmark
    rorflex = params.elasticities.rorflex
    caps = [f for f in sets.f if str(f).lower() in _CAPITAL]
    if not caps:
        raise ValueError(f"capFlex risk: no capital factor among {list(sets.f)}")

    rore: dict[str, float] = {}
    netinv: dict[str, float] = {}
    for r in sets.r:
        kstock = float(b.vkb.get(r, 0.0))
        if kstock <= 0.0:
            raise ValueError(f"capFlex risk: region {r} has no capital stock (VKB)")
        evos = sum(float(b.evos.get((r, f, a), 0.0)) for f in caps for a in sets.a)
        depr = float(b.vdep.get(r, 0.0)) / kstock
        xi = sum(
            float(b.vdip.get((r, i), 0.0)) + float(b.vmip.get((r, i), 0.0))
            for i in sets.i
        )
        kap_end = (1.0 - depr) * kstock + xi
        rorc = evos / kstock - depr
        value = rorc * (kstock / kap_end) ** float(rorflex[r])
        if not math.isfinite(value) or abs(value) <= 1e-12:
            raise ValueError(
                f"capFlex risk: region {r} has rore = {value!r} at the benchmark"
            )
        rore[r] = value
        netinv[r] = xi - depr * kstock
    rorg = sum(rore[r] * netinv[r] for r in rore) / sum(netinv.values())
    return {r: rorg / rore[r] for r in rore}


def stamp_capflex_risk(params: Any, closure: Any) -> None:
    """Set ``params._capflex_risk`` from the data under capFlex, every time (so a
    params object reused after its data changed, e.g. a RoRFlex sweep, never keeps an
    earlier run's value), and remove it under any other closure (so it cannot move the
    capFix cache keys or results). Both backends stamp before any cache key."""
    if getattr(closure, "savf_flag", "capFix") != "capFlex":
        if hasattr(params, "_capflex_risk"):
            del params._capflex_risk
        return
    params._capflex_risk = capflex_risk_from_data(params)
