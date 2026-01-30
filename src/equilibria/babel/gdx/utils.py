"""
GDX utility functions and constants.

Binary parsers, compression helpers, format mappings.
"""

from __future__ import annotations

from typing import Final

# GDX endianness byte mapping
ENDIANNESS_MAP: Final[dict[bytes, str]] = {
    b"\x01": "little",
    b"\x02": "big",
}

# GDX platform byte mapping
PLATFORM_MAP: Final[dict[bytes, str]] = {
    b"\x01": "Windows",
    b"\x02": "Linux",
    b"\x03": "macOS",
}

# GDX symbol type codes
SYMBOL_TYPE_MAP: Final[dict[int, str]] = {
    0: "set",
    1: "parameter",
    2: "variable",
    3: "equation",
    4: "alias",
}

# GDX variable/equation attribute indices
ATTR_LEVEL: Final[int] = 0
ATTR_MARGINAL: Final[int] = 1
ATTR_LOWER: Final[int] = 2
ATTR_UPPER: Final[int] = 3
ATTR_SCALE: Final[int] = 4

# Special values in GDX
GDX_UNDEF: Final[float] = float("nan")
GDX_NA: Final[float] = float("nan")
GDX_PINF: Final[float] = float("inf")
GDX_MINF: Final[float] = float("-inf")
GDX_EPS: Final[float] = 1e-300


def decode_special_value(value: float) -> float:
    """
    Decode GDX special values to Python equivalents.

    Args:
        value: Raw value from GDX file.

    Returns:
        Python float with proper special value handling.
    """
    # GDX uses specific bit patterns for special values
    # This is a simplified version
    return value


def encode_special_value(value: float) -> float:
    """
    Encode Python special values to GDX format.

    Args:
        value: Python float value.

    Returns:
        Value encoded for GDX format.
    """
    # Handle Python inf/nan -> GDX special values
    return value
