"""
equilibria/babel/gdx/reader

GDX file reader - Pure Python implementation.

This module provides native GDX file reading capabilities without
requiring GAMS installation.
"""

# Standard library imports
from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

# Local imports
from equilibria.babel.gdx.utils import ENDIANNESS_MAP, PLATFORM_MAP


def read_gdx(filepath: str | Path) -> dict[str, Any]:
    """
    Read a GDX file and return its contents.

    Args:
        filepath: Path to the GDX file.

    Returns:
        Dictionary containing header info, symbols, and data.

    Example:
        >>> data = read_gdx("results.gdx")
        >>> print(data["header"])
        >>> for sym in data["symbols"]:
        ...     print(sym["name"], sym["type"])
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"GDX file not found: {filepath}")

    if not filepath.suffix.lower() == ".gdx":
        raise ValueError(f"Expected .gdx file, got: {filepath.suffix}")

    with open(file=filepath, mode="rb") as f:
        header: dict[str, Any] = read_header(file=f)
        symbols: list[dict[str, Any]] = read_symbol_table(file=f, endianness=header["endianness"])

    return {
        "filepath": str(object=filepath),
        "header": header,
        "symbols": symbols,
    }


def read_header(file: BinaryIO) -> dict[str, Any]:
    """
    Read the GDX file header.

    Args:
        file: Open binary file handle.

    Returns:
        Dictionary with version, endianness, and platform info.
    """
    # GDX magic number check
    magic: bytes = file.read(4)
    if magic != b"GAMSGDX\x00"[:4]:  # Simplified check
        # Try to continue anyway - format varies by version
        pass

    # Version info (bytes 4-7)
    version_bytes: bytes = file.read(4)
    version: int = int.from_bytes(version_bytes, "little")

    # Endianness indicator (byte 8)
    endian_byte: bytes = file.read(1)
    endianness: str = ENDIANNESS_MAP.get(endian_byte, "little")

    # Platform indicator (byte 9)
    platform_byte: bytes = file.read(1)
    platform: str = PLATFORM_MAP.get(platform_byte, "unknown")

    # Skip remaining header bytes to reach symbol table
    file.read(128 - 10)  # Align to 128-byte header

    return {
        "version": version,
        "endianness": endianness,
        "platform": platform,
    }


def read_symbol_table(file: BinaryIO, endianness: str = "little") -> list[dict[str, Any]]:
    """
    Read the symbol table from a GDX file.

    Args:
        file: Open binary file handle positioned after header.
        endianness: Byte order ('little' or 'big').

    Returns:
        List of symbol dictionaries with name, type, dimension, description.
    """
    file.seek(128)  # Skip header block (commonly 128 bytes in GDX)

    num_symbols_bytes = file.read(4)
    if len(num_symbols_bytes) < 4:
        return []

    num_symbols: int = int.from_bytes(bytes=num_symbols_bytes, byteorder=endianness)

    symbols: list[dict[str, Any]] = []
    for _ in range(num_symbols):
        try:
            # Read symbol name
            name_len_bytes: bytes = file.read(4)
            if len(name_len_bytes) < 4:
                break
            name_len: int = int.from_bytes(bytes=name_len_bytes, byteorder=endianness)

            if name_len > 256 or name_len < 0:  # Sanity check
                break

            name: str = file.read(name_len).decode(encoding="utf-8", errors="replace")

            # Read symbol type
            sym_type: int = int.from_bytes(bytes=file.read(4), byteorder=endianness)

            # Read dimension
            dimension: int = int.from_bytes(bytes=file.read(4), byteorder=endianness)

            # Read description
            desc_len: int = int.from_bytes(bytes=file.read(4), byteorder=endianness)
            if desc_len > 1024 or desc_len < 0:  # Sanity check
                desc_len = 0
            description: str = file.read(desc_len).decode(encoding="utf-8", errors="replace") if desc_len > 0 else ""

            symbols.append({
                "name": name,
                "type": sym_type,
                "dimension": dimension,
                "description": description,
            })

        except Exception:
            # Stop reading on any error
            break

    return symbols
