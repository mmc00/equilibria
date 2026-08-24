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
