"""The capFlex ``risk`` oracle covers Burfisher's dataset and agrees with itself."""

from __future__ import annotations

import math

import pytest
from tests.templates.gtap._capflex_risk_oracle import RISK_GDX, RISK_TWIN, TWIN_REL


def test_oracle_covers_burfisher_and_several_sizes() -> None:
    assert "nus333" in RISK_GDX and "nus333" in RISK_TWIN
    assert len(RISK_GDX) >= 4


@pytest.mark.parametrize("dataset", sorted(RISK_GDX))
def test_every_region_has_a_finite_positive_risk(dataset: str) -> None:
    _, risk = RISK_GDX[dataset]
    assert risk and all(math.isfinite(v) and v > 0.0 for v in risk.values())


@pytest.mark.parametrize("dataset", sorted(RISK_TWIN))
def test_the_twin_is_close_to_gams_but_not_equal(dataset: str) -> None:
    """The twin is a solve: within its tolerance of GAMS, never bit for bit."""
    _, gams = RISK_GDX[dataset]
    twin = RISK_TWIN[dataset]
    assert set(twin) == set(gams)
    rel = max(abs(twin[r] - gams[r]) / gams[r] for r in gams)
    assert 0.0 < rel < TWIN_REL
