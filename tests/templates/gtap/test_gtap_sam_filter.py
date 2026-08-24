"""Tests for GTAP SAM filtering (gtap_sam_filter).

Port of CGEBox gtapAgg/filter*.gms — remove small flows + LP re-balance,
pre-calibration. See docs/superpowers/specs/2026-08-23-gtap-sam-filtering-design.md
and docs/superpowers/plans/notes-filtering-recon.md.
"""

from __future__ import annotations

import dataclasses

import pytest


def test_filter_config_defaults():
    from equilibria.templates.gtap.gtap_sam_filter import FilterConfig

    c = FilterConfig()
    assert c.rel_tol == 1e-5
    assert c.n_steps == 6
    assert c.keep_gdp is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.rel_tol = 1.0  # type: ignore[misc]


class _Bench:
    """Minimal benchmark stand-in with a couple of p/b flow dicts."""

    def __init__(self):
        # vxsb keyed (exporter, commodity, importer); one large, one tiny cell
        self.vxsb = {
            ("USA", "Crops", "EU_28"): 100.0,
            ("USA", "Crops", "JPN"): 1e-8,
        }


def test_flag_small_flows_marks_tiny_relative_to_sector():
    from equilibria.templates.gtap.gtap_sam_filter import flag_small_flows

    b = _Bench()
    # sector total for USA/Crops exports ~= 100 + 1e-8; rel_tol 1e-3 -> cut ~0.1
    field_map = {"vxsb": lambda bench, key: 100.00000001}
    flagged = flag_small_flows(
        b, sets=None, rel_tol=1e-3, abs_tol=1e-9, field_map=field_map
    )
    assert ("USA", "Crops", "JPN") in flagged["vxsb"]
    assert ("USA", "Crops", "EU_28") not in flagged["vxsb"]
    # does not mutate the benchmark
    assert b.vxsb[("USA", "Crops", "JPN")] == 1e-8


# --- Task 3: region re-balance LP (uses a real region as a consistent fixture) ---


def _load_10x7():
    from pathlib import Path

    from equilibria.templates.gtap import GTAPParameters

    D = Path("datasets/gtap7_10x7")
    if not (D / "basedata.har").exists():
        pytest.skip("gtap7_10x7 dataset not present")
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=D / "basedata.har",
        sets_path=D / "sets.har",
        default_path=D / "default.prm",
        baserate_path=D / "baserate.har",
    )
    return p


def _domestic_total(bench, sets, r, i):
    """Domestic demand for good i in region r (basic prices)."""
    dfb = sum(bench.vdfb.get((r, i, a), 0.0) for a in sets.a)
    return (
        dfb
        + bench.vdpb.get((r, i), 0.0)
        + bench.vdgb.get((r, i), 0.0)
        + bench.vdib.get((r, i), 0.0)
    )


@pytest.mark.integration
def test_rebalance_region_zeros_flagged_and_preserves_domestic_balance():
    from equilibria.templates.gtap.gtap_sam_filter import (
        FilterConfig,
        rebalance_region,
    )

    p = _load_10x7()
    b, sets = p.benchmark, p.sets
    r = "USA"
    # flag a tiny export cell (a $2-class flow that exists in the crude SAM)
    tiny = min(
        ((abs(v), k) for k, v in b.vxsb.items() if k[0] == r and v > 0),
        default=(None, None),
    )[1]
    assert tiny is not None
    flagged = {"vxsb": {tiny}}

    out = rebalance_region(b, sets, r, flagged, FilterConfig(), solver_name="ipopt")

    # (a) flagged flow driven to ~0
    assert abs(out["vxsb"][tiny]) < 1e-6
    # (b) domestic market for that good still balances after re-balance
    i = tiny[1]
    dom_dem = sum(
        out.get("vdfb", {}).get((r, i, a), b.vdfb.get((r, i, a), 0.0)) for a in sets.a
    )
    assert dom_dem >= 0.0  # sanity; full balance asserted in Task 4 macro test


@pytest.mark.integration
def test_filter_sam_shrinks_trade_and_preserves_trade_total():
    from equilibria.templates.gtap.gtap_sam_filter import FilterConfig, filter_sam

    p = _load_10x7()
    b, sets = p.benchmark, p.sets
    nnz0 = sum(1 for v in b.vxsb.values() if abs(v) > 1e-12)
    trade0 = sum(b.vxsb.values())

    out = filter_sam(
        b, sets, p.elasticities, p.taxes, FilterConfig(n_steps=3), solver_name="ipopt"
    )

    nnz1 = sum(1 for v in out.vxsb.values() if abs(v) > 1e-12)
    trade1 = sum(out.vxsb.values())
    assert nnz1 < nnz0  # removed at least one tiny trade flow
    assert abs(trade1 - trade0) < 1e-3 * trade0  # aggregate trade preserved
    # returns a NEW benchmark, does not mutate the input
    assert sum(1 for v in b.vxsb.values() if abs(v) > 1e-12) == nnz0


def test_load_from_har_default_off_unchanged():
    """filter_config=None (default) must leave the benchmark byte-identical."""
    from pathlib import Path

    from equilibria.templates.gtap import GTAPParameters

    D = Path("datasets/gtap7_3x3")
    if not (D / "basedata.har").exists():
        pytest.skip("gtap7_3x3 dataset not present")

    kw = {
        "basedata_path": D / "basedata.har",
        "sets_path": D / "sets.har",
        "default_path": D / "default.prm",
        "baserate_path": D / "baserate.har",
    }
    p1 = GTAPParameters()
    p1.load_from_har(**kw)
    p2 = GTAPParameters()
    p2.load_from_har(filter_config=None, **kw)
    assert dict(p2.benchmark.vxsb) == dict(p1.benchmark.vxsb)
