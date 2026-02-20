"""Canonical post-transform safeguards for solver variables."""

from __future__ import annotations

from typing import Any

from equilibria.templates.pep_model_equations import PEPModelVariables


def rebuild_tax_detail_from_rates(
    vars: PEPModelVariables,
    sets: dict[str, list[str]],
    params: dict[str, Any],
    *,
    include_tip: bool = True,
) -> None:
    """Rebuild detailed tax-payment variables from ad-valorem policy rates.

    This is the canonical reconstruction used after array->variables conversion
    and in initialization routines that need deterministic tax detail.
    """
    for labor in sets.get("L", []):
        for sector in sets.get("J", []):
            key = (labor, sector)
            ttiw = params.get("ttiw", {}).get(key, 0.0)
            vars.TIW[key] = ttiw * vars.W.get(labor, 1.0) * vars.LD.get(key, 0.0)

    for capital in sets.get("K", []):
        for sector in sets.get("J", []):
            key = (capital, sector)
            ttik = params.get("ttik", {}).get(key, 0.0)
            vars.TIK[key] = ttik * vars.R.get(key, 1.0) * vars.KD.get(key, 0.0)

    if include_tip:
        for sector in sets.get("J", []):
            ttip = params.get("ttip", {}).get(sector, 0.0)
            vars.TIP[sector] = ttip * vars.PP.get(sector, 0.0) * vars.XST.get(sector, 0.0)

