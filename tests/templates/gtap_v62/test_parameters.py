"""Tests for GTAPv62Parameters — v6.2 benchmark + elasticity loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.templates.gtap_v62 import (
    GTAPv62BenchmarkValues,
    GTAPv62Elasticities,
    GTAPv62Parameters,
    GTAPv62Sets,
)


BOOK3X3_DIR = Path("C:/runGTAP375/BOOK3X3")
BOOK3X3_SETS = BOOK3X3_DIR / "SETS.HAR"
BOOK3X3_BASE = BOOK3X3_DIR / "basedata.har"
BOOK3X3_DEFAULT = BOOK3X3_DIR / "Default.prm"


def _rungtap_available() -> bool:
    return all(p.exists() for p in (BOOK3X3_SETS, BOOK3X3_BASE, BOOK3X3_DEFAULT))


pytestmark = pytest.mark.skipif(
    not _rungtap_available(),
    reason="RunGTAP v6.2 dataset BOOK3X3 not available",
)


@pytest.fixture
def sets() -> GTAPv62Sets:
    s = GTAPv62Sets()
    s.load_from_har(BOOK3X3_SETS, default_path=BOOK3X3_DEFAULT)
    return s


@pytest.fixture
def params(sets: GTAPv62Sets) -> GTAPv62Parameters:
    p = GTAPv62Parameters()
    p.load_from_har(
        basedata_path=BOOK3X3_BASE,
        default_prm_path=BOOK3X3_DEFAULT,
        sets=sets,
    )
    return p


def test_loads_elasticities(params: GTAPv62Parameters) -> None:
    """All v6.2 elasticity headers are populated from Default.prm."""
    e = params.elasticities
    assert isinstance(e, GTAPv62Elasticities)
    assert set(e.esubd.keys()) == {"food", "mnfcs", "svces"}
    assert set(e.esubm.keys()) == {"food", "mnfcs", "svces"}
    # ESBT/ESBV are indexed over PROD_COMM = TRAD_COMM ∪ CGDS_COMM
    assert set(e.esubt.keys()) == {"food", "mnfcs", "svces", "CGDS"}
    assert set(e.esubva.keys()) == {"food", "mnfcs", "svces", "CGDS"}
    # ETRE indexed over factors
    assert set(e.etrae.keys()) == {"Land", "Labor", "Capital"}
    # RFLX indexed over regions
    assert set(e.rorflex.keys()) == {"USA", "EU", "ROW"}


def test_elasticity_values_match_dataset(params: GTAPv62Parameters) -> None:
    """Spot-check known BOOK3X3 elasticity values."""
    e = params.elasticities
    # From inspection of Default.prm:
    assert e.esubd["food"] == pytest.approx(2.399, abs=1e-2)
    assert e.esubm["food"] == pytest.approx(4.639, abs=1e-2)
    assert e.esubva["food"] == pytest.approx(0.789, abs=1e-2)
    # ETRE = -1.0 is the GEMPACK signal for "infinite" / mobile
    assert e.etrae["Land"] == pytest.approx(-1.0)


def test_loads_factor_benchmark(params: GTAPv62Parameters) -> None:
    """VFM/EVFA load with (factor, sector, region) keying."""
    b = params.benchmark
    assert isinstance(b, GTAPv62BenchmarkValues)
    # 3 factors × 4 sectors (PROD_COMM) × 3 regions = 36 cells
    assert len(b.vfm) == 36
    assert len(b.evfa) == 36
    # Keys are 3-tuples
    sample_key = next(iter(b.vfm.keys()))
    assert len(sample_key) == 3
    # Land is used in food but not CGDS
    assert ("Land", "food", "USA") in b.vfm
    assert b.vfm[("Land", "CGDS", "USA")] == 0.0


def test_loads_intermediate_benchmark(params: GTAPv62Parameters) -> None:
    """VDFA/VIFA load with (commodity, sector, region) keying."""
    b = params.benchmark
    # 3 commodities × 4 sectors (PROD_COMM) × 3 regions = 36 cells
    assert len(b.vdfa) == 36
    assert len(b.vifa) == 36
    assert ("food", "food", "USA") in b.vdfa


def test_loads_trade_benchmark(params: GTAPv62Parameters) -> None:
    """VXMD/VIMS/VXWD/VIWS load with (commodity, source, destination) keying."""
    b = params.benchmark
    # 3 commodities × 3 sources × 3 destinations = 27 cells (incl. diagonal which GTAP sets to 0)
    assert len(b.vxmd) == 27
    assert len(b.vims) == 27
    assert len(b.vxwd) == 27
    assert len(b.viws) == 27


def test_loads_margin_benchmark(params: GTAPv62Parameters) -> None:
    """VST/VTWR margins load correctly."""
    b = params.benchmark
    # VST: 1 margin × 3 regions = 3 cells
    assert len(b.vst) == 3
    # VTWR: 1 margin × 3 commodities × 3 sources × 3 destinations = 27
    assert len(b.vtwr) == 27


def test_loads_region_aggregates(params: GTAPv62Parameters) -> None:
    """1-d region series load with plain string keys."""
    b = params.benchmark
    assert set(b.vkb.keys()) == {"USA", "EU", "ROW"}
    assert set(b.vdep.keys()) == {"USA", "EU", "ROW"}
    assert set(b.save.keys()) == {"USA", "EU", "ROW"}
    # VKB (initial capital) must be positive for every region
    assert all(v > 0 for v in b.vkb.values())


def test_validate_book3x3_clean(params: GTAPv62Parameters, sets: GTAPv62Sets) -> None:
    """A fully-loaded BOOK3X3 should pass parameter validation."""
    is_valid, errors = params.validate(sets)
    assert is_valid, f"Unexpected validation errors: {errors}"
