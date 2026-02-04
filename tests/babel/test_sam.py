"""Tests for SAM (Social Accounting Matrix) class."""

import numpy as np
import pandas as pd
import pytest

from equilibria.babel import SAM


class TestSAM:
    """Tests for SAM class."""

    @pytest.fixture
    def sample_sam_data(self):
        """Create sample SAM data."""
        accounts = ["sec1", "sec2", "labor", "capital", "hh"]
        data = np.array(
            [
                [10, 20, 0, 0, 30],
                [15, 25, 0, 0, 35],
                [25, 45, 0, 0, 0],
                [10, 20, 0, 0, 0],
                [0, 0, 70, 30, 0],
            ],
            dtype=float,
        )
        return pd.DataFrame(data, index=accounts, columns=accounts)

    @pytest.fixture
    def balanced_sam_data(self):
        """Create perfectly balanced SAM data."""
        accounts = ["sec1", "sec2", "hh"]
        # Create balanced matrix (rows sum = columns sum)
        data = np.array(
            [
                [10, 20, 30],
                [15, 25, 40],
                [25, 45, 0],
            ],
            dtype=float,
        )
        return pd.DataFrame(data, index=accounts, columns=accounts)

    def test_sam_creation_from_dataframe(self, sample_sam_data):
        """Test creating SAM from DataFrame."""
        sam = SAM.from_dataframe(sample_sam_data, name="TestSAM")
        assert sam.name == "TestSAM"
        assert sam.data.shape == (5, 5)
        assert "AC" in sam.sets

    def test_sam_must_be_square(self):
        """Test that SAM must be square."""
        data = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]))
        with pytest.raises(ValueError):
            SAM.from_dataframe(data)

    def test_sam_validation_balanced(self, balanced_sam_data):
        """Test validation of balanced SAM."""
        sam = SAM.from_dataframe(balanced_sam_data)
        validation = sam.check_balance(tolerance=1e-6)

        assert validation["is_balanced"] is True
        assert validation["max_difference"] <= 1e-6
        assert len(validation["unbalanced_accounts"]) == 0

    def test_sam_validation_unbalanced(self, sample_sam_data):
        """Test validation of unbalanced SAM."""
        sam = SAM.from_dataframe(sample_sam_data)
        validation = sam.check_balance(tolerance=1e-6)

        assert validation["is_balanced"] is False
        assert validation["max_difference"] > 1e-6
        assert len(validation["unbalanced_accounts"]) > 0

    def test_sam_get_submatrix(self, sample_sam_data):
        """Test extracting submatrices."""
        sam = SAM.from_dataframe(sample_sam_data)
        sectors = ["sec1", "sec2"]

        sub = sam.get_submatrix(sectors, sectors)
        assert sub.shape == (2, 2)
        assert sub.loc["sec1", "sec1"] == 10
        assert sub.loc["sec2", "sec2"] == 25

    def test_sam_extract_sets(self, sample_sam_data):
        """Test extracting sets from accounts."""
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

    def test_sam_summary(self, sample_sam_data):
        """Test summary generation."""
        sam = SAM.from_dataframe(sample_sam_data, name="Test")
        summary = sam.summary()

        assert summary["name"] == "Test"
        assert summary["shape"] == (5, 5)
        assert summary["accounts"] == 5
        assert "total_value" in summary

    def test_sam_to_dict(self, sample_sam_data):
        """Test serialization to dict."""
        sam = SAM.from_dataframe(sample_sam_data, name="Test")
        d = sam.to_dict()

        assert d["name"] == "Test"
        assert "data" in d
        assert "sets" in d

    def test_sam_balance_ras(self, sample_sam_data):
        """Test RAS balancing."""
        # Create slight imbalance
        unbalanced_data = sample_sam_data.copy()
        unbalanced_data.iloc[0, 1] += 5
        unbalanced_sam = SAM.from_dataframe(unbalanced_data)

        # Balance it
        balanced = unbalanced_sam.balance(method="ras")

        # Check that it's more balanced
        val_before = unbalanced_sam.check_balance()["max_difference"]
        val_after = balanced.check_balance()["max_difference"]
        assert val_after < val_before

    def test_sam_repr(self, sample_sam_data):
        """Test string representation."""
        sam = SAM.from_dataframe(sample_sam_data, name="Test")
        repr_str = repr(sam)

        assert "Test" in repr_str
        assert "5x5" in repr_str
