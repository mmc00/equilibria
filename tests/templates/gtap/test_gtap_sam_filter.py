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
