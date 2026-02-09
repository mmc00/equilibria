"""
Test GDX decoder against CSV ground truth.

This test validates that the GDX decoder correctly reads all parameter values
from SAM-V2_0.gdx and matches the expected values from the CSV fixture.
"""

import csv
from pathlib import Path

import pytest

from equilibria.babel.gdx.reader import read_gdx, read_data_sections
from equilibria.babel.gdx.decoder import decode_parameter_delta


class TestGDXDecoderAgainstCSV:
    """Test GDX decoder accuracy against CSV ground truth."""

    @pytest.fixture
    def gdx_file(self):
        """Path to SAM-V2_0.gdx fixture."""
        return Path(__file__).parent / "fixtures" / "SAM-V2_0.gdx"

    @pytest.fixture
    def csv_file(self):
        """Path to expected values CSV fixture."""
        return Path(__file__).parent / "fixtures" / "sam_v2_0_expected.csv"

    @pytest.fixture
    def expected_values(self, csv_file):
        """Load expected values from CSV fixture."""
        values = {}
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 5:
                    key = (row[0], row[1], row[2], row[3])
                    values[key] = float(row[4])
        return values

    @pytest.fixture
    def decoded_values(self, gdx_file):
        """Decode values from GDX file."""
        # Read GDX metadata
        gdx_data = read_gdx(gdx_file)
        
        # Read raw bytes
        raw_data = gdx_file.read_bytes()
        data_sections = read_data_sections(raw_data)
        
        # Get SAM section
        _, section = data_sections[0]
        
        # Decode parameter
        elements = gdx_data['elements']
        return decode_parameter_delta(section, elements, 4)

    def test_decode_all_records(self, decoded_values, expected_values):
        """Test that all 196 records are decoded."""
        assert len(decoded_values) == 196
        assert len(expected_values) == 196

    def test_all_keys_match(self, decoded_values, expected_values):
        """Test that all keys from CSV are present in decoded values."""
        missing_keys = set(expected_values.keys()) - set(decoded_values.keys())
        assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"

    def test_no_extra_keys(self, decoded_values, expected_values):
        """Test that decoded values don't have extra keys not in CSV."""
        extra_keys = set(decoded_values.keys()) - set(expected_values.keys())
        assert len(extra_keys) == 0, f"Extra keys: {extra_keys}"

    def test_values_match_within_tolerance(self, decoded_values, expected_values):
        """Test that all values match within tolerance."""
        tolerance = 0.01
        mismatches = []
        
        for key, expected_val in expected_values.items():
            decoded_val = decoded_values[key]
            if abs(expected_val - decoded_val) > tolerance:
                mismatches.append({
                    'key': key,
                    'expected': expected_val,
                    'decoded': decoded_val,
                    'diff': abs(expected_val - decoded_val)
                })
        
        assert len(mismatches) == 0, f"Value mismatches: {mismatches[:5]}"

    def test_specific_records(self, decoded_values, expected_values):
        """Test specific known records."""
        test_cases = [
            (('L', 'USK', 'J', 'AGR'), 10002.0),
            (('L', 'USK', 'J', 'IND'), 2289.0),
            (('L', 'USK', 'J', 'ADM'), 3006.0),
            (('L', 'USK', 'OTH', 'TOT'), 15297.0),
            (('L', 'SK', 'J', 'AGR'), 910.0),
        ]
        
        for key, expected_val in test_cases:
            assert key in decoded_values, f"Key {key} not found"
            assert abs(decoded_values[key] - expected_val) < 0.01, \
                f"Value mismatch for {key}: expected {expected_val}, got {decoded_values[key]}"

    def test_first_record(self, decoded_values):
        """Test first record has correct key and value."""
        # First record should be ('L', 'USK', 'J', 'AGR') = 10002.0
        assert ('L', 'USK', 'J', 'AGR') in decoded_values
        assert abs(decoded_values[('L', 'USK', 'J', 'AGR')] - 10002.0) < 0.01

    def test_last_record(self, decoded_values):
        """Test last record exists and has valid value."""
        # Just verify we have all 196 records and last one has a value
        assert len(decoded_values) == 196
        # Values can be negative in SAM matrix (e.g., taxes, transfers)
        # Just check they are valid numbers (not NaN, not infinity)
        for key, value in decoded_values.items():
            assert value == value, f"NaN value for {key}"  # Check not NaN
            assert abs(value) < 1e15, f"Extreme value for {key}: {value}"

    def test_value_ranges(self, decoded_values):
        """Test that values are within reasonable ranges for SAM matrix."""
        values = list(decoded_values.values())
        
        # Check min/max values
        min_val = min(values)
        max_val = max(values)
        
        # SAM values should be reasonable (not NaN, not infinity)
        assert all(v == v for v in values), "Found NaN values"
        assert all(abs(v) < 1e15 for v in values), "Found extremely large values"
        
        # Log range for debugging
        print(f"Value range: {min_val} to {max_val}")

    def test_100_percent_accuracy(self, decoded_values, expected_values):
        """Comprehensive test achieving 100% accuracy."""
        # Check counts
        assert len(decoded_values) == 196
        assert len(expected_values) == 196
        
        # Check all keys match
        assert set(decoded_values.keys()) == set(expected_values.keys())
        
        # Check all values match
        tolerance = 0.01
        all_match = all(
            abs(decoded_values[k] - expected_values[k]) <= tolerance
            for k in expected_values.keys()
        )
        assert all_match, "Not all values match within tolerance"
        
        # Calculate match statistics
        match_count = sum(
            1 for k in expected_values.keys()
            if abs(decoded_values[k] - expected_values[k]) <= tolerance
        )
        match_rate = match_count / len(expected_values)
        
        assert match_rate == 1.0, f"Match rate is {match_rate}, expected 1.0"
        print(f"✅ 100% accuracy achieved: {match_count}/{len(expected_values)} records match")
