"""SAM filtering for the GTAP benchmark (port of CGEBox gtapAgg/filter*.gms).

Removes economically insignificant flows and re-balances the SAM by LP, before
calibration, to reduce numerical dispersion in bilateral trade (the qxs residual;
see memory/project_gtap_qxs_bilateral_trade_residual_2026_08_21).

Design: docs/superpowers/specs/2026-08-23-gtap-sam-filtering-design.md
Field map (equilibria uses p/b decomposition, NOT CGEBox aggregate names):
docs/superpowers/plans/notes-filtering-recon.md
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FilterConfig:
    """Filtering knobs (defaults mirror CGEBox gtapAgg/filter.gms).

    rel_tol/abs_tol: a flow is flagged for removal when it is below
        max(abs_tol, rel_tol * sector_total). n_steps: iterative rounds ramping
        the tolerance 10%->100% (filter.gms:455-466). keep_*: preserve macro
        totals during the LP re-balance. exc_secs/exc_regs: exclude from filtering.
    """

    rel_tol: float = 1e-5
    abs_tol: float = 1e-6
    n_steps: int = 6
    keep_gdp: bool = True
    keep_factor_income: bool = True
    keep_intermediate: bool = False
    min_cost_share_va: float = 0.0
    max_seed_cost_share: float = 1.0
    exc_secs: tuple[str, ...] = ()
    exc_regs: tuple[str, ...] = ()
