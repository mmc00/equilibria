"""Tests for SAM (Social Accounting Matrix) class."""

import pandas as pd
import pytest

from equilibria.babel import SAM


class TestSAM:
    """Interface checks for the Babel SAM wrapper."""

    @pytest.fixture
    def sample_sam_data(self):
        """Create sample SAM data."""
        accounts = ["sec1", "sec2", "labor", "capital", "hh"]
        data = [
            [10, 20, 0, 0, 30],
            [15, 25, 0, 0, 35],
            [25, 45, 0, 0, 0],
            [10, 20, 0, 0, 0],
            [0, 0, 70, 30, 0],
        ]
        return pd.DataFrame(data, index=accounts, columns=accounts)

    def test_sam_creation_from_dataframe(self, sample_sam_data):
        sam = SAM.from_dataframe(sample_sam_data, name="TestSAM")
        assert sam.name == "TestSAM"
        assert sam.data.shape == (5, 5)
        assert "AC" in sam.sets

    def test_sam_summary_and_serialization(self, sample_sam_data):
        sam = SAM.from_dataframe(sample_sam_data, name="Test")
        summary = sam.summary()
        serial = sam.to_dict()

        assert summary["name"] == "Test"
        assert summary["shape"] == (5, 5)
        assert "total_value" in summary
        assert serial["name"] == "Test"
        assert "sets" in serial

    def test_sam_extract_sets(self, sample_sam_data):
        sam = SAM.from_dataframe(sample_sam_data)
        mapping = {
            "J": ["sec"],
            "I": ["labor", "capital"],
            "H": ["hh"],
        }
        sam.extract_sets(mapping)

        assert "J" in sam.sets
        assert "I" in sam.sets
        assert "H" in sam.sets
        assert sam.sets["J"] == ["sec1", "sec2"]

    def test_sam_repr(self, sample_sam_data):
        sam = SAM.from_dataframe(sample_sam_data, name="Test")
        repr_str = repr(sam)
        assert "Test" in repr_str
        assert "5x5" in repr_str
