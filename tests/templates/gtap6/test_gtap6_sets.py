"""GTAP6Sets loads datasets/gtap6_3x3 correctly."""

from __future__ import annotations

from pathlib import Path

from equilibria.templates.gtap6.gtap6_sets import GTAP6Sets

DATASET = Path(__file__).resolve().parents[3] / "datasets" / "gtap6_3x3"


def test_load_from_har_gtap6_3x3():
    sets = GTAP6Sets()
    sets.load_from_har(DATASET / "sets.har", default_path=DATASET / "default.prm")

    assert sets.r == ["USA", "EU", "ROW"] or len(sets.r) == 3
    assert len(sets.i) == 3
    assert sets.a == sets.i  # alias property, no ACT/COMM split
    assert sets.is_diagonal is True
    assert len(sets.f) >= 1
    is_valid, errors = sets.validate()
    assert is_valid, errors


def test_mobile_sluggish_partition_covers_all_factors():
    sets = GTAP6Sets()
    sets.load_from_har(DATASET / "sets.har", default_path=DATASET / "default.prm")

    assert set(sets.mf) | set(sets.sf) == set(sets.f)
    assert set(sets.mf) & set(sets.sf) == set()
