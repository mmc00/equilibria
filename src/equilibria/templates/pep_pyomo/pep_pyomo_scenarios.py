"""Counterfactual scenarios for the PEP model — the `** 6.4 Simulations` block of
PEP-1-1_v2_1.gms, applied to the CALIBRATED STATE before build_pep_model.

GAMS applies a scenario by fixing a benchmark-derived rate (e.g. `ttix.fx(i)=ttixO(i)*0.75`)
right before the solve. Since the Pyomo port derives every model rate from the calibrated
state's `*O` benchmarks (PEPParams reads `state.trade['ttixO']` etc.), the faithful analog
is to scale that benchmark IN THE STATE, then build — the model then carries the shocked
rate everywhere GAMS's fixed variable would. This keeps the builder scenario-agnostic (the
scenario is external, exactly as in the .gms).

SIM1 (the one uncommented simulation in the reference .gms): a 25% cut to the export-tax
rate — `ttix.fx(i) = ttixO(i)*0.75`. (The comment says "all indirect tax rates" but only
the ttix line is active; GAMS is the source of truth, so only ttix is scaled.)
"""
from __future__ import annotations
from typing import Any


def apply_sim1_export_tax_cut(state: Any, factor: float = 0.75) -> Any:
    """Scale the export-tax benchmark `ttixO` in-place by `factor` (default 0.75 = −25%,
    the reference SIM1). Mutates and returns `state` so PEPParams derives the shocked ttix.
    Idempotent only per fresh calibration — call on a newly calibrated state, not twice."""
    trade = getattr(state, "trade", None)
    if not isinstance(trade, dict) or "ttixO" not in trade:
        raise ValueError("state.trade['ttixO'] not found — cannot apply the SIM1 export-tax shock")
    for i in list(trade["ttixO"]):
        trade["ttixO"][i] = trade["ttixO"][i] * factor
    return state
