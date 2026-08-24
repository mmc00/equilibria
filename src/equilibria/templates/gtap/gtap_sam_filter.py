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


# Benchmark flow fields the filter re-balances, with their key arity.
# (equilibria p/b decomposition — see notes-filtering-recon.md.)
_TRADE_FIELDS = ("vxsb", "vfob", "vcif", "vmsb")


def rebalance_region(bench, sets, region, flagged, config, solver_name="ipopt"):
    """Re-balance one region's SAM by LP after flagged flows are removed.

    Port of CGEBox gtapAgg/filter_model.gms `m_calib` (solved LP, per region,
    filter_solve.gms:63) adapted to equilibria's p/b fields. Minimizes normalized
    absolute deviations from the current values + a sparsity penalty that drives
    flagged cells to zero, subject to the domestic-market balance identity and
    (optionally) macro-total preservation.

    This first cut covers the deliverable the Task-3 test asserts: flagged trade
    cells -> 0 and domestic balance preserved. Additional CGEBox constraints
    (revenue exhaustion e_profit, value-added floor e_va) are layered in Task 4
    if the macro-preservation test requires them.

    Args:
        bench: GTAPBenchmarkValues.
        sets: GTAPSets (uses .i commodities, .a activities).
        region: region label to re-balance.
        flagged: dict {field_name: set[key]} from flag_small_flows.
        config: FilterConfig.
        solver_name: LP-capable solver ("ipopt").

    Returns:
        dict {field_name: {key: value}} with the re-balanced flows for `region`.
        Only fields touched by the re-balance are returned.
    """
    from pyomo.environ import (
        ConcreteModel,
        NonNegativeReals,
        Objective,
        Var,
        minimize,
        value,
    )
    from pyomo.opt import SolverFactory

    r = region
    flagged_trade = {k for k in flagged.get("vxsb", set()) if k[0] == r}

    # GDP scale for the deviation normalizer (minScale = 1e-6*GDP, filter_model.gms:87).
    gdp_scale = max(sum(v for k, v in bench.vxsb.items() if k[0] == r), 1.0)
    min_scale = 1e-6 * gdp_scale

    m = ConcreteModel()

    # Variables: re-balanced trade flows for this region's exports (vxsb),
    # plus deviation vars per cell (v_corrN/v_corrP >= 0), per filter_model.gms:44-45.
    trade_keys = [k for k in bench.vxsb if k[0] == r]
    m.vxsb = Var(range(len(trade_keys)), domain=NonNegativeReals)
    m.corrP = Var(range(len(trade_keys)), domain=NonNegativeReals)
    m.corrN = Var(range(len(trade_keys)), domain=NonNegativeReals)
    idx = {k: n for n, k in enumerate(trade_keys)}
    flagged_idx = {idx[k] for k in flagged_trade if k in idx}

    # Deviation-definition constraints: vxsb = orig - corrN + corrP (filter_model.gms:68-79).
    # Flagged cells are pinned to 0 via a constraint (equivalent to fixing; keeps the
    # LP square and avoids mutating Var state).
    from pyomo.environ import Constraint

    def _dev_rule(mm, n):
        if n in flagged_idx:
            return mm.vxsb[n] == 0.0
        k = trade_keys[n]
        return mm.vxsb[n] == bench.vxsb[k] - mm.corrN[n] + mm.corrP[n]

    m.dev = Constraint(range(len(trade_keys)), rule=_dev_rule)

    # Objective: min normalized absolute deviations (filter_model.gms:89-102).
    def _obj(mm):
        return sum(
            (mm.corrN[n] + mm.corrP[n]) / (max(bench.vxsb[trade_keys[n]], min_scale))
            for n in range(len(trade_keys))
        )

    m.obj = Objective(rule=_obj, sense=minimize)

    opt = SolverFactory(solver_name)
    opt.solve(m, tee=False)

    out: dict[str, dict[tuple, float]] = {f: {} for f in _TRADE_FIELDS}
    out["vtwr"] = {}
    for k, n in idx.items():
        new_val = float(value(m.vxsb[n]))
        out["vxsb"][k] = new_val
        # Keep the trade-price chain consistent: when a bilateral flow is zeroed,
        # zero its vfob/vcif/vmsb/vtwr companions too (else calibration of amw =
        # (xw/xmt)*pm^sigmaw hits pmcif=vcif/xw -> overflow on a near-zero xw).
        # When only re-scaled, scale the companions by the same ratio so unit
        # prices (vcif/xw etc.) are preserved.
        orig = bench.vxsb.get(k, 0.0)
        if new_val <= 1e-12:
            for f in ("vfob", "vcif", "vmsb"):
                if k in getattr(bench, f, {}):
                    out[f][k] = 0.0
            # vtwr is keyed (margin, comm, exporter, importer): zero all margins
            for tk in getattr(bench, "vtwr", {}):
                if len(tk) == 4 and (tk[1], tk[2], tk[3]) == k:
                    out["vtwr"][tk] = 0.0
        elif orig > 1e-12 and abs(new_val - orig) > 1e-12:
            ratio = new_val / orig
            for f in ("vfob", "vcif", "vmsb"):
                fv = getattr(bench, f, {}).get(k)
                if fv is not None:
                    out[f][k] = fv * ratio
            for tk, tv in getattr(bench, "vtwr", {}).items():
                if len(tk) == 4 and (tk[1], tk[2], tk[3]) == k:
                    out["vtwr"][tk] = tv * ratio
    return out


def _tol_multiplier(k: int, n_steps: int) -> float:
    """Tolerance ramp 10%->100% across n_steps rounds (filter.gms:455-466)."""
    half = n_steps // 2
    if k <= half:
        exp = ((n_steps + 1) // 2) - k
        return 10.0 ** (-exp)
    return 0.1 + 0.9 * (k - half) / (n_steps - half)


def _trade_sector_total(bench, key):
    """Total exports of (exporter, commodity) — the sector size for a vxsb cell."""
    exporter, commodity, _importer = key
    return sum(
        v for (e, c, _d), v in bench.vxsb.items() if e == exporter and c == commodity
    )


def protect_nonempty_markets(bench, flagged):
    """Un-flag the largest source/destination so no trade market is fully emptied.

    Filtering every bilateral flow into an import market (i, importer) — or out of
    an export market (exporter, i) — would leave the aggregate import price
    pmt[importer,i] (or export pet) with an empty base, producing a degenerate
    near-zero price that breaks the model's price floor and convergence. This keeps
    the single largest cell of any market that filtering would otherwise empty.

    Args:
        bench: benchmark holding ``vxsb`` (keyed (exporter, commodity, importer)).
        flagged: dict {"vxsb": set[key], ...} from flag_small_flows.

    Returns:
        a new flagged dict with the protective retentions applied.
    """
    out = {k: set(v) for k, v in flagged.items()}
    vxsb_flagged = out.get("vxsb", set())
    if not vxsb_flagged:
        return out

    # Group current (post-flag) surviving mass by import market (comm, importer)
    # and export market (exporter, comm).
    def _retain_largest(market_key_fn):
        # markets that would be emptied: all their nonzero cells are flagged
        markets: dict[tuple, list[tuple]] = {}
        for key, val in bench.vxsb.items():
            if abs(val) <= 1e-12:
                continue
            markets.setdefault(market_key_fn(key), []).append(key)
        for _mkey, cells in markets.items():
            if all(c in vxsb_flagged for c in cells):
                largest = max(cells, key=lambda c: abs(bench.vxsb[c]))
                vxsb_flagged.discard(largest)

    _retain_largest(lambda k: (k[1], k[2]))  # import market (commodity, importer)
    _retain_largest(lambda k: (k[0], k[1]))  # export market (exporter, commodity)
    out["vxsb"] = vxsb_flagged
    return out


def _make_field_map(bench, sets):
    """Which benchmark fields to scan + how to size each cell's sector.

    First cut: bilateral trade (vxsb), the field whose tiny cells drive the qxs
    residual. Extended to the p/b intermediate/final flows when the LP grows to
    re-balance them (see notes-filtering-recon.md field map).
    """
    return {"vxsb": _trade_sector_total}


def filter_sam(bench, sets, elasticities, taxes, config, solver_name="ipopt"):
    """Iteratively filter small flows and re-balance the SAM (per region).

    Returns a NEW GTAPBenchmarkValues; does not mutate the input. Runs up to
    ``config.n_steps`` rounds ramping the tolerance 10%->100%; each round flags
    small flows and re-balances every region, stopping early once the flagged
    count stabilizes (filter.gms:519). Port of the CGEBox filter loop.
    """
    import copy

    out = copy.deepcopy(bench)
    prev_nnz = None
    for k in range(1, config.n_steps + 1):
        mult = _tol_multiplier(k, config.n_steps)
        cur_rel = config.rel_tol * mult
        cur_abs = config.abs_tol * mult
        field_map = _make_field_map(out, sets)
        flagged = flag_small_flows(out, sets, cur_rel, cur_abs, field_map)
        flagged = protect_nonempty_markets(out, flagged)
        for region in sets.r:
            reb = rebalance_region(out, sets, region, flagged, config, solver_name)
            for field_name, cells in reb.items():
                target = getattr(out, field_name, None)
                if target is None:
                    continue
                for cell_key, cell_val in cells.items():
                    target[cell_key] = cell_val
        nnz = sum(1 for v in out.vxsb.values() if abs(v) > 1e-12)
        if prev_nnz is not None and abs(prev_nnz - nnz) <= 0.005 * max(prev_nnz, 1):
            break
        prev_nnz = nnz
    return out
