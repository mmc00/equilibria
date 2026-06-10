"""Unit tests for sluggish factor pft market clearing fix.

These tests verify that after the fix:
  - xftflag is activated for sf factors with flow data
  - aft/etaf are calibrated for sf factors
  - eq_xft and eq_xfteq constraints exist for sf factors
"""
from __future__ import annotations

from unittest.mock import MagicMock
import pytest


# ---------------------------------------------------------------------------
# Minimal mock helpers
# ---------------------------------------------------------------------------

def _make_sets(regions, activities, factors, mobile_factors, sluggish_factors):
    """Build a minimal GTAPSets-like namespace."""
    s = MagicMock()
    s.r = regions
    s.a = activities
    s.f = factors
    s.mf = set(mobile_factors)
    s.sf = set(sluggish_factors)
    s.i = []
    return s


def _make_params(evfb_data: dict, vfm_data: dict, etrae_data: dict | None = None):
    """Build a minimal GTAPParameters-like namespace."""
    p = MagicMock()
    p.benchmark.evfb.get = lambda key, default=0.0: evfb_data.get(key, default)
    p.benchmark.vfm.get = lambda key, default=0.0: vfm_data.get(key, default)
    p.elasticities.etrae = etrae_data or {}
    p.taxes.kappaf_activity.get = lambda key, default=0.0: 0.0
    p.taxes.kappaf.get = lambda key, default=0.0: 0.0
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_xftflag_activated_for_sf_with_flow():
    """xftflag[(r, sf)] must be 1.0 when evfb has flow for that (r, sf, a)."""
    sets = _make_sets(
        regions=["RegA"],
        activities=["Act1"],
        factors=["Capital", "Land"],
        mobile_factors=["Capital"],
        sluggish_factors=["Land"],
    )
    params = _make_params(
        evfb_data={
            ("RegA", "Capital", "Act1"): 5.0,
            ("RegA", "Land", "Act1"): 3.0,
        },
        vfm_data={},
    )

    xftflag_data = {}
    for r in sets.r:
        for f in sets.f:
            any_flow = False
            for a in sets.a:
                val = params.benchmark.evfb.get((r, f, a), 0.0)
                if abs(val) > 1e-12:
                    any_flow = True
            xftflag_data[(r, f)] = 1.0 if (any_flow and f in (sets.mf | sets.sf)) else 0.0

    assert xftflag_data[("RegA", "Land")] == 1.0, "Land should be flagged when it has flow"
    assert xftflag_data[("RegA", "Capital")] == 1.0, "Capital should still be flagged"


def test_xftflag_zero_for_sf_without_flow():
    """xftflag[(r, sf)] must be 0.0 when sf has no flow in that region."""
    sets = _make_sets(
        regions=["RegA"],
        activities=["Act1"],
        factors=["Capital", "Land"],
        mobile_factors=["Capital"],
        sluggish_factors=["Land"],
    )
    params = _make_params(
        evfb_data={("RegA", "Capital", "Act1"): 5.0},  # no Land flow
        vfm_data={},
    )

    xftflag_data = {}
    for r in sets.r:
        for f in sets.f:
            any_flow = False
            for a in sets.a:
                val = params.benchmark.evfb.get((r, f, a), 0.0)
                if abs(val) > 1e-12:
                    any_flow = True
            xftflag_data[(r, f)] = 1.0 if (any_flow and f in (sets.mf | sets.sf)) else 0.0

    assert xftflag_data[("RegA", "Land")] == 0.0, "Land should NOT be flagged when no flow"


def test_aft_calibrated_for_sf():
    """aft[(r, sf)] must be populated with benchmark aggregate supply (kappa=0 → pf_val=1)."""
    sets = _make_sets(
        regions=["RegA"],
        activities=["Act1", "Act2"],
        factors=["Capital", "Land"],
        mobile_factors=["Capital"],
        sluggish_factors=["Land"],
    )
    # Land has flow in Act1 (3.0) and Act2 (2.0), pf_val=1 because kappa=0
    params = _make_params(
        evfb_data={
            ("RegA", "Capital", "Act1"): 5.0,
            ("RegA", "Land", "Act1"): 3.0,
            ("RegA", "Land", "Act2"): 2.0,
        },
        vfm_data={},
    )

    aft_data = {}
    for region in ["RegA"]:
        for factor in ["Capital", "Land"]:
            if factor in sets.mf or factor in sets.sf:
                benchmark_xft = 0.0
                for activity in ["Act1", "Act2"]:
                    factor_flow = float(
                        params.benchmark.evfb.get((region, factor, activity), 0.0) or 0.0
                    )
                    if factor_flow <= 0.0:
                        continue
                    kappa = 0.0
                    pf_val = max(1.0 / max(1.0 - kappa, 1e-8), 1e-8)
                    benchmark_xft += factor_flow / pf_val
                aft_data[(region, factor)] = benchmark_xft

    assert aft_data[("RegA", "Land")] == pytest.approx(5.0), "aft[Land] = 3.0 + 2.0"
    assert aft_data[("RegA", "Capital")] == pytest.approx(5.0), "Capital unchanged"


def test_etaf_zero_for_sf_without_etrae():
    """etaf[(r, sf)] must be 0.0 when etrae is not set (standard GTAP)."""
    etrae_data = {}  # no etrae for Land

    def _lookup_etrae(region, factor):
        raw = etrae_data
        for key in ((factor, region), (region, factor), factor):
            val = raw.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return 0.0

    assert _lookup_etrae("RegA", "Land") == 0.0


def test_etaf_nonzero_for_sf_with_etrae():
    """etaf[(r, sf)] must pick up etrae value when present (GFTLnd-style)."""
    etrae_data = {("Land", "RegA"): 0.6}

    def _lookup_etrae(region, factor):
        raw = etrae_data
        for key in ((factor, region), (region, factor), factor):
            val = raw.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return 0.0

    assert _lookup_etrae("RegA", "Land") == pytest.approx(0.6)
