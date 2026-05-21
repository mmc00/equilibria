"""Tests for the v6.2 calibration helpers (derived tax rates and aggregates)."""

from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.templates.gtap_v62 import (
    GTAPv62Parameters,
    GTAPv62Sets,
    derive_calibration,
)


BOOK3X3_DIR = Path("C:/runGTAP375/BOOK3X3")


def _rungtap_available() -> bool:
    return all(
        (BOOK3X3_DIR / fname).exists()
        for fname in ("SETS.HAR", "basedata.har", "Default.prm")
    )


pytestmark = pytest.mark.skipif(
    not _rungtap_available(),
    reason="RunGTAP v6.2 dataset BOOK3X3 not available",
)


@pytest.fixture
def sets() -> GTAPv62Sets:
    s = GTAPv62Sets()
    s.load_from_har(
        BOOK3X3_DIR / "SETS.HAR",
        default_path=BOOK3X3_DIR / "Default.prm",
    )
    return s


@pytest.fixture
def params(sets: GTAPv62Sets) -> GTAPv62Parameters:
    p = GTAPv62Parameters()
    p.load_from_har(
        basedata_path=BOOK3X3_DIR / "basedata.har",
        default_prm_path=BOOK3X3_DIR / "Default.prm",
        sets=sets,
    )
    return p


def test_derive_calibration_detects_tariff(sets, params) -> None:
    """Exp1a baseline has ~37% US→EU food tariff hidden in the SAM."""
    c = derive_calibration(sets, params)
    rate = c.tms.get(("food", "USA", "EU"), 0.0)
    # The benchmark tariff is around 36.89% on US→EU food (V/W - 1).
    # Exp1a is a 10% cut of the *power* of this tariff.
    assert 0.30 < rate < 0.45


def test_derived_rates_default_to_zero(sets, params) -> None:
    """When V*M is zero (no flow), the derived rate is 0."""
    c = derive_calibration(sets, params)
    # No factor tax in BOOK3X3 (clean dataset)
    assert c.tf.get(("Labor", "food", "USA"), 0.0) == 0.0
    # No household domestic tax
    assert c.tpd.get(("food", "USA"), 0.0) == 0.0


def test_vom_positive_for_active_sectors(sets, params) -> None:
    """VOM (output) is strictly positive for sectors with non-zero output."""
    c = derive_calibration(sets, params)
    for r in sets.r:
        for i in sets.i:
            assert c.vom.get((i, r), 0.0) > 0.0, (
                f"VOM({i},{r}) should be positive in BOOK3X3"
            )
        for cg in sets.cgds:
            assert c.vom.get((cg, r), 0.0) > 0.0, (
                f"VOM({cg},{r}) (capital goods) should be positive"
            )


def test_evom_matches_evoa(sets, params) -> None:
    """Factor income (EVOM) is read from EVOA header."""
    c = derive_calibration(sets, params)
    b = params.benchmark
    for f in sets.f:
        for r in sets.r:
            assert c.evom[(f, r)] == b.evoa.get((f, r), 0.0)


def test_vds_plus_exports_equals_vom_for_trad_comm(sets, params) -> None:
    """VOM(i,r) = VDS(i,r) + sum_s VXMD(i,r,s) + VST(i,r) for traded commodities."""
    c = derive_calibration(sets, params)
    b = params.benchmark
    for r in sets.r:
        for i in sets.i:
            exports = sum(b.vxmd.get((i, r, rp), 0.0) for rp in sets.r if rp != r)
            margin = b.vst.get((i, r), 0.0)
            balance = c.vds[(i, r)] + exports + margin
            assert abs(balance - c.vom[(i, r)]) < 1e-3, (
                f"Output balance check failed for ({i},{r}): "
                f"VDS+exports+margin={balance:.4f} vs VOM={c.vom[(i,r)]:.4f}"
            )


def test_export_tax_calibration_recovers_known_cells(sets, params) -> None:
    """Spot-check derived export tax rates against known BOOK3X3 cells.

    The 1997 GTAP database underlying BOOK3X3 contains:
    - EU→ROW food: a positive export tax (~30%) from CAP-era subsidies
      stored at producer-price levels.
    - Most other trade cells: near zero.

    These are derived from VXMD/VXWD ratios and should be reproducible.
    """
    c = derive_calibration(sets, params)
    # EU→ROW food: known sizable rate from CAP-era data
    rate_eu_row_food = c.txs.get(("food", "EU", "ROW"), 0.0)
    assert 0.10 < rate_eu_row_food < 0.50, (
        f"Expected ~30% EU→ROW food export tax, got {rate_eu_row_food:.4f}"
    )

    # All keys exist (no missing flows)
    expected_keys = {
        (i, s, d) for i in sets.i for s in sets.r for d in sets.r if s != d
    }
    have = {k for k in c.txs.keys() if k[1] != k[2]}  # off-diagonal
    assert expected_keys.issubset(have), (
        f"Missing trade-flow keys in derived txs: {expected_keys - have}"
    )
