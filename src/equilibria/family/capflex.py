"""capFlex's ``risk[r]`` (``rorg/rore(r)`` at the benchmark) with no Pyomo model.

The value comes from the calibrated data as GAMS ``cal.gms:680-684``
(``blocks.gtap.capflex_risk``), the same function the Pyomo route calls, so both
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
    """Set ``params._capflex_risk`` under capFlex (before any cache key is taken, so
    capFix and capFlex records never share a key). Under any other closure it removes a
    ``_capflex_risk`` left by an earlier capFlex run, so that run's value cannot move the
    capFix keys or results."""
    if getattr(closure, "savf_flag", "capFix") != "capFlex":
        if hasattr(params, "_capflex_risk"):
            del params._capflex_risk
        return
    risk = capflex_risk(params)
    held = getattr(params, "_capflex_risk", None)
    if held is not None and held != risk:
        raise ValueError(
            f"params._capflex_risk is stale: {held} differs from the data's {risk}"
        )
    params._capflex_risk = risk
