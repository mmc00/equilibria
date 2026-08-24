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


def flag_small_flows(bench, sets, rel_tol, abs_tol, field_map):
    """Flag benchmark cells whose value is economically insignificant.

    A cell is flagged when ``abs(value) < max(abs_tol, rel_tol * sector_total)``,
    where ``sector_total`` is computed per-cell by ``field_map[field_name](bench, key)``
    (mirrors CGEBox filter.gms:579-610 relative + absolute thresholds). Pure: does
    NOT mutate ``bench``.

    Args:
        bench: GTAPBenchmarkValues (or stand-in) holding the flow dicts.
        sets: GTAPSets (unused here; kept for signature symmetry with callers).
        rel_tol: relative tolerance (fraction of the sector total).
        abs_tol: absolute floor.
        field_map: dict {field_name: sector_total_fn(bench, key) -> float} listing
            which benchmark fields to scan and how to size each cell's sector.

    Returns:
        dict {field_name: set[key]} of cells flagged for removal.
    """
    flagged: dict[str, set[tuple]] = {}
    for field_name, sector_total_fn in field_map.items():
        d = getattr(bench, field_name, None)
        if not d:
            continue
        marked: set[tuple] = set()
        for key, val in d.items():
            cut = max(abs_tol, rel_tol * sector_total_fn(bench, key))
            if abs(val) < cut:
                marked.add(key)
        flagged[field_name] = marked
    return flagged
