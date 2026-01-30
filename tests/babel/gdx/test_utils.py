"""
equilibria/tests/babel/gdx/test_utils.py - Tests for GDX utility functions.

Unit tests for equilibria.babel.gdx.utils module.

Tests cover:
- Constant mappings (ENDIANNESS_MAP, PLATFORM_MAP, SYMBOL_TYPE_MAP)
- Attribute indices
- Special value constants
- decode_special_value() function
- encode_special_value() function
"""

from __future__ import annotations

import math

from equilibria.babel.gdx.utils import (
    ATTR_LEVEL,
    ATTR_LOWER,
    ATTR_MARGINAL,
    ATTR_SCALE,
    ATTR_UPPER,
    ENDIANNESS_MAP,
    GDX_EPS,
    GDX_MINF,
    GDX_NA,
    GDX_PINF,
    GDX_UNDEF,
    PLATFORM_MAP,
    SYMBOL_TYPE_MAP,
    decode_special_value,
    encode_special_value,
)


class TestEndianessMap:
    """Tests for ENDIANNESS_MAP constant."""

    def test_little_endian_mapping(self) -> None:
        """Should map 0x01 to 'little'."""
        assert ENDIANNESS_MAP[b"\x01"] == "little"

    def test_big_endian_mapping(self) -> None:
        """Should map 0x02 to 'big'."""
        assert ENDIANNESS_MAP[b"\x02"] == "big"

    def test_only_two_mappings(self) -> None:
        """Should have exactly 2 mappings."""
        assert len(ENDIANNESS_MAP) == 2


class TestPlatformMap:
    """Tests for PLATFORM_MAP constant."""

    def test_windows_mapping(self) -> None:
        """Should map 0x01 to 'Windows'."""
        assert PLATFORM_MAP[b"\x01"] == "Windows"

    def test_linux_mapping(self) -> None:
        """Should map 0x02 to 'Linux'."""
        assert PLATFORM_MAP[b"\x02"] == "Linux"

    def test_macos_mapping(self) -> None:
        """Should map 0x03 to 'macOS'."""
        assert PLATFORM_MAP[b"\x03"] == "macOS"

    def test_three_mappings(self) -> None:
        """Should have exactly 3 platform mappings."""
        assert len(PLATFORM_MAP) == 3


class TestSymbolTypeMap:
    """Tests for SYMBOL_TYPE_MAP constant."""

    def test_set_mapping(self) -> None:
        """Should map 0 to 'set'."""
        assert SYMBOL_TYPE_MAP[0] == "set"

    def test_parameter_mapping(self) -> None:
        """Should map 1 to 'parameter'."""
        assert SYMBOL_TYPE_MAP[1] == "parameter"

    def test_variable_mapping(self) -> None:
        """Should map 2 to 'variable'."""
        assert SYMBOL_TYPE_MAP[2] == "variable"

    def test_equation_mapping(self) -> None:
        """Should map 3 to 'equation'."""
        assert SYMBOL_TYPE_MAP[3] == "equation"

    def test_alias_mapping(self) -> None:
        """Should map 4 to 'alias'."""
        assert SYMBOL_TYPE_MAP[4] == "alias"

    def test_five_symbol_types(self) -> None:
        """Should have exactly 5 symbol type mappings."""
        assert len(SYMBOL_TYPE_MAP) == 5


class TestAttributeIndices:
    """Tests for variable/equation attribute index constants."""

    def test_level_index(self) -> None:
        """Level should be at index 0."""
        assert ATTR_LEVEL == 0

    def test_marginal_index(self) -> None:
        """Marginal should be at index 1."""
        assert ATTR_MARGINAL == 1

    def test_lower_index(self) -> None:
        """Lower bound should be at index 2."""
        assert ATTR_LOWER == 2

    def test_upper_index(self) -> None:
        """Upper bound should be at index 3."""
        assert ATTR_UPPER == 3

    def test_scale_index(self) -> None:
        """Scale should be at index 4."""
        assert ATTR_SCALE == 4

    def test_indices_are_sequential(self) -> None:
        """Indices should be sequential 0-4."""
        indices: list[int] = [
            ATTR_LEVEL,
            ATTR_MARGINAL,
            ATTR_LOWER,
            ATTR_UPPER,
            ATTR_SCALE,
        ]
        assert indices == [0, 1, 2, 3, 4]


class TestSpecialValueConstants:
    """Tests for GDX special value constants."""

    def test_gdx_undef_is_nan(self) -> None:
        """GDX_UNDEF should be NaN."""
        assert math.isnan(GDX_UNDEF)

    def test_gdx_na_is_nan(self) -> None:
        """GDX_NA should be NaN."""
        assert math.isnan(GDX_NA)

    def test_gdx_pinf_is_positive_infinity(self) -> None:
        """GDX_PINF should be positive infinity."""
        assert float("inf") == GDX_PINF
        assert math.isinf(GDX_PINF)
        assert GDX_PINF > 0

    def test_gdx_minf_is_negative_infinity(self) -> None:
        """GDX_MINF should be negative infinity."""
        assert float("-inf") == GDX_MINF
        assert math.isinf(GDX_MINF)
        assert GDX_MINF < 0

    def test_gdx_eps_is_very_small(self) -> None:
        """GDX_EPS should be a very small positive number."""
        assert GDX_EPS > 0
        assert GDX_EPS < 1e-100


class TestDecodeSpecialValue:
    """Tests for decode_special_value() function."""

    def test_decode_regular_value(self) -> None:
        """Should pass through regular values unchanged."""
        assert decode_special_value(1.0) == 1.0
        assert decode_special_value(-5.5) == -5.5
        assert decode_special_value(0.0) == 0.0

    def test_decode_infinity(self) -> None:
        """Should handle infinity values."""
        assert decode_special_value(float("inf")) == float("inf")
        assert decode_special_value(float("-inf")) == float("-inf")

    def test_decode_nan(self) -> None:
        """Should handle NaN values."""
        result: float = decode_special_value(float("nan"))
        assert math.isnan(result)


class TestEncodeSpecialValue:
    """Tests for encode_special_value() function."""

    def test_encode_regular_value(self) -> None:
        """Should pass through regular values unchanged."""
        assert encode_special_value(1.0) == 1.0
        assert encode_special_value(-5.5) == -5.5
        assert encode_special_value(0.0) == 0.0

    def test_encode_infinity(self) -> None:
        """Should handle infinity values."""
        assert encode_special_value(float("inf")) == float("inf")
        assert encode_special_value(float("-inf")) == float("-inf")

    def test_encode_nan(self) -> None:
        """Should handle NaN values."""
        result: float = encode_special_value(float("nan"))
        assert math.isnan(result)


class TestAttributeIndexUsage:
    """Tests demonstrating attribute index usage pattern."""

    def test_extract_values_from_tuple(self) -> None:
        """Should use indices to extract values from variable/equation tuple."""
        # Typical variable record: (level, marginal, lower, upper, scale)
        record: tuple[float, float, float, float, float] = (
            100.0,
            0.5,
            0.0,
            float("inf"),
            1.0,
        )

        assert record[ATTR_LEVEL] == 100.0
        assert record[ATTR_MARGINAL] == 0.5
        assert record[ATTR_LOWER] == 0.0
        assert record[ATTR_UPPER] == float("inf")
        assert record[ATTR_SCALE] == 1.0
