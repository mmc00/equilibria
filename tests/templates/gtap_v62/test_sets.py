"""Tests for GTAPv62Sets — v6.2 set loading and structure."""

from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.templates.gtap_v62 import GTAPv62Sets


BOOK3X3_DIR = Path("C:/runGTAP375/BOOK3X3")
BOOK3X3_SETS = BOOK3X3_DIR / "SETS.HAR"
BOOK3X3_DEFAULT = BOOK3X3_DIR / "Default.prm"


def _rungtap_available() -> bool:
    return BOOK3X3_SETS.exists() and BOOK3X3_DEFAULT.exists()


pytestmark = pytest.mark.skipif(
    not _rungtap_available(),
    reason="RunGTAP v6.2 dataset BOOK3X3 not available on this host",
)


def test_loads_book3x3_sets() -> None:
    """v6.2 SETS.HAR with classic H1/H2/H6/H9 headers loads correctly."""
    sets = GTAPv62Sets()
    sets.load_from_har(BOOK3X3_SETS, default_path=BOOK3X3_DEFAULT)

    assert sets.r == ["USA", "EU", "ROW"]
    assert sets.i == ["food", "mnfcs", "svces"]
    assert sets.cgds == ["CGDS"]
    assert sets.f == ["Land", "Labor", "Capital"]
    assert sets.marg == ["svces"]


def test_factor_mobility_partition_book3x3() -> None:
    """BOOK3X3's SLUG=[3,1,1] heuristic → Land=sluggish, Labor/Cap=mobile."""
    sets = GTAPv62Sets()
    sets.load_from_har(BOOK3X3_SETS, default_path=BOOK3X3_DEFAULT)

    assert sets.sf == ["Land"]
    assert sets.mf == ["Labor", "Capital"]
    # No factor is in both partitions
    assert not set(sets.mf) & set(sets.sf)
    # Union equals f
    assert set(sets.mf) | set(sets.sf) == set(sets.f)


def test_derived_sets_book3x3() -> None:
    """PROD_COMM, DEMD_COMM, NSAV_COMM follow v6.2 definitions."""
    sets = GTAPv62Sets()
    sets.load_from_har(BOOK3X3_SETS, default_path=BOOK3X3_DEFAULT)

    # PROD_COMM = TRAD_COMM ∪ CGDS_COMM
    assert sets.prod_comm == ["food", "mnfcs", "svces", "CGDS"]
    # DEMD_COMM = ENDW_COMM ∪ TRAD_COMM
    assert sets.demd_comm == ["Land", "Labor", "Capital", "food", "mnfcs", "svces"]
    # NSAV_COMM = DEMD_COMM ∪ CGDS_COMM
    assert sets.nsav_comm == ["Land", "Labor", "Capital", "food", "mnfcs", "svces", "CGDS"]


def test_activity_alias_returns_commodities() -> None:
    """v6.2 has no separate activities set; ``sets.a`` aliases ``sets.i``."""
    sets = GTAPv62Sets()
    sets.load_from_har(BOOK3X3_SETS, default_path=BOOK3X3_DEFAULT)
    assert sets.a == sets.i
    assert sets.is_diagonal is True


def test_validate_book3x3_no_errors() -> None:
    """A correctly loaded BOOK3X3 should validate clean."""
    sets = GTAPv62Sets()
    sets.load_from_har(BOOK3X3_SETS, default_path=BOOK3X3_DEFAULT)
    is_valid, errors = sets.validate()
    assert is_valid, f"Unexpected validation errors: {errors}"


def test_count_properties() -> None:
    """Convenience counters report the right cardinalities for BOOK3X3."""
    sets = GTAPv62Sets()
    sets.load_from_har(BOOK3X3_SETS, default_path=BOOK3X3_DEFAULT)
    assert sets.n_regions == 3
    assert sets.n_commodities == 3
    assert sets.n_factors == 3
    assert sets.n_mobile_factors == 2
    assert sets.n_sluggish_factors == 1


def test_missing_default_prm_uses_hint_fallback(tmp_path: Path) -> None:
    """Without GTAPPARM, factor mobility falls back to name hints."""
    sets = GTAPv62Sets()
    # Pass a non-existent path → loader skips SLUG and uses hint fallback
    sets.load_from_har(BOOK3X3_SETS, default_path=tmp_path / "no-such.prm")

    # The hint fallback marks 'Land' as sluggish (matches canonical
    # GTAP textbook convention).
    assert "Land" in sets.sf
    assert "Labor" in sets.mf and "Capital" in sets.mf


def test_repr_describes_structure() -> None:
    sets = GTAPv62Sets()
    sets.load_from_har(BOOK3X3_SETS, default_path=BOOK3X3_DEFAULT)
    rep = repr(sets)
    assert "BOOK3X3" in rep
    assert "3r" in rep and "3i" in rep and "3f" in rep
