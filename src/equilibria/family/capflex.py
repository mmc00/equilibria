"""capFlex's ``risk[r]`` (``rorg/rore(r)`` at the benchmark) with no Pyomo model.

The value comes from the calibrated data as GAMS ``cal.gms:680-684``
(``blocks.gtap.capflex_risk``), the same functions the Pyomo route calls, so both
stamp the same numbers. No GAMS GDX is needed."""

from __future__ import annotations

from typing import Any


def capflex_risk(params: Any) -> dict[str, float]:
    """``{region: rorg/rore(r)}`` at the benchmark; ``ValueError`` naming a region
    that has no rate of return."""
    # imported here: the equilibria.blocks.gtap package loads Pyomo, the family route must not
    from equilibria.blocks.gtap.capflex_risk import capflex_risk_from_data

    return capflex_risk_from_data(params)


def stamp_capflex_risk(params: Any, closure: Any) -> None:
    """``blocks.gtap.capflex_risk.stamp_capflex_risk`` (imported on call, as above)."""
    from equilibria.blocks.gtap import capflex_risk as _risk

    _risk.stamp_capflex_risk(params, closure)
