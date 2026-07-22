"""Build a ``GTAPVariableSnapshot`` populated with GAMS-literal altertax values.

The output is passed to ``GTAPSolver.apply_solution_hint`` to warm-start the
altertax solve with values that exactly mirror comp_altertax.gms:14469-14730.

Calibration outputs from ``calibration_sequence`` are normalised here to match
each Pyomo variable's index shape (single-region vars indexed by ``r`` rather
than ``(r,)``).
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from equilibria.templates.gtap.altertax.calibration_sequence import (
    compute_altertax_initial_values,
)
from equilibria.templates.gtap.gtap_parameters import GTAPParameters
from equilibria.templates.gtap.gtap_parity_pipeline import GTAPVariableSnapshot


def _unwrap_single(d: dict[tuple, float]) -> dict[str, float]:
    """Convert ``{("r",): v}`` to ``{"r": v}`` for vars indexed by a single set."""
    out: dict[str, float] = {}
    for k, v in d.items():
        if isinstance(k, tuple) and len(k) == 1:
            out[k[0]] = v
        elif isinstance(k, str):
            out[k] = v
    return out


def _drop_agent_dim(d: dict[tuple, float]) -> dict[str, float]:
    """Convert ``{("r","hhd"): v}`` to ``{"r": v}`` for welfare vars."""
    out: dict[str, float] = {}
    for k, v in d.items():
        if isinstance(k, tuple) and len(k) >= 1:
            out[k[0]] = v
    return out


def build_altertax_warm_start_snapshot(
    params: GTAPParameters,
    baseline: GTAPVariableSnapshot | None = None,
) -> GTAPVariableSnapshot:
    """Build a warm-start snapshot for the altertax solve.

    If ``baseline`` is supplied, all dict-valued fields are shallow-copied from
    it first; altertax-recomputed fields then overwrite. ``params.taxes`` is
    mutated in-place by ``compute_factor_tax_wedges`` (fctts/fcttx) so the
    model build itself picks up the new wedges -- the snapshot only carries
    the variable initial *levels*.
    """
    kwargs: dict[str, Any] = {}
    if baseline is not None:
        for f in fields(GTAPVariableSnapshot):
            val = getattr(baseline, f.name, None)
            if isinstance(val, dict):
                kwargs[f.name] = dict(val)
            elif val is not None:
                kwargs[f.name] = val

    cached = getattr(params, "_altertax_initial_values", None)
    vals = cached if cached is not None else compute_altertax_initial_values(params)

    # H15 fix (2026-05-22): the side-effect that mutates taxes.fctts_rate /
    # fcttx_rate to the GAMS formula (-FBEP/(pf*xf), FTRV/(pf*xf)) shifts the
    # wedge in eq_pfaeq/eq_pfyeq to large positive values (e.g. EU_28 Land
    # a_agricultur: fctts+fcttx=+0.42), while baseline xf/pf were calibrated
    # for rtf~-0.38. Without overwriting the warm-start factor levels, PATH
    # starts far from equilibrium in agricultural sectors and stalls
    # (residual ~0.23 plateau, eq_ug[SSA] cascade with abs_res=14.28).
    # Overwrite the factor side AND demand side with GAMS-literal levels so
    # equations and warm-start are mutually consistent.
    # H15.b (2026-05-22): full snapshot override regressed 87.08% → 79.33%.
    # Quantities (xf, xft, demand-side) from GAMS-literal computation desync
    # against baseline xa/pa/pd/pm/x/xs (which DON'T get recomputed) → PATH
    # stalls worse. Quirurgic alternative: only overwrite **factor PRICES**
    # (pf, pft) which the wedge mutation directly demands new values for.
    # Quantities are left to PATH to recompute via eq_xfeq from new prices.
    if "pf" in vals:
        kwargs["pf"] = dict(vals["pf"])
    if "pft" in vals:
        kwargs["pft"] = dict(vals["pft"])
    # H26 (2026-05-22): under H21 (all-factors-promoted), eq_pfaeq and
    # eq_pfyeq are now ACTIVE for all (r,f,a) — not just NatRes. PATH
    # needs sensible initial values for pfa/pfy, not the baseline ones
    # (which were computed from baseline pf and baseline wedge). Use the
    # GAMS-literal calibration values from compute_altertax_initial_values.
    if "pfa" in vals:
        kwargs["pfa"] = dict(vals["pfa"])
    if "pfy" in vals:
        kwargs["pfy"] = dict(vals["pfy"])

    # H55 (2026-05-25, REVERTED): injecting `va` from _altertax_initial_values
    # regressed 86.64% → 79.32%. Same H15.b mechanism: quantity-side override
    # without xa/pa/pd/pm/x sync cascades into desync. va field on
    # GTAPVariableSnapshot kept for future use; not consumed by solver hint.
    # The eq_xfeq residual at init (max_abs +3.43e-02, 19 cells > 1e-3)
    # cannot be closed by injecting va alone — needs joint xf/va/pva/pfa
    # reconciliation (H55 option 3 mini-cal sub-MCP, untested).

    return GTAPVariableSnapshot(**kwargs)
