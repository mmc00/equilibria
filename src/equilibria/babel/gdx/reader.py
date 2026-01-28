"""
equilibria/babel/gdx/reader

GDX file reader - Pure Python implementation.

This module provides native GDX file reading capabilities without
requiring GAMS installation.

GDX Format Notes (Version 7):
- First ~19 bytes: checksum/metadata
- Bytes 19-25: Magic "GAMSGDX"
- Bytes 26-29: Version number (little-endian uint32)
- Following: Library info string, then data sections
- "_SYMB_" marker: Symbol table
- "_UEL_" marker: Unique Element List (set elements)
- "_SETT_" marker: Settings

Symbol table entry structure (after _SYMB_ marker + 4-byte count):
- 1 byte: name length
- N bytes: name (ASCII)
- 1 byte: type flag (0x01=set, 0x20=alias, 0x3f/0x66=parameter)
- 25 bytes: metadata
  - byte 7: dimension count
  - bytes 16-19: record count (little-endian uint32)
- 1 byte: description length
- M bytes: description
- 6 bytes: padding
"""

# Standard library imports
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO

# GDX format constants
GDX_MAGIC: bytes = b"GAMSGDX"
GDX_SYMB_MARKER: bytes = b"_SYMB_"
GDX_UEL_MARKER: bytes = b"_UEL_"

# Symbol type codes
SYMBOL_TYPE_NAMES: dict[int, str] = {
    0: "set",
    1: "parameter",
    2: "variable",
    3: "equation",
    4: "alias",
}


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
        data: bytes = f.read()

    header: dict[str, Any] = read_header_from_bytes(data)
    symbols: list[dict[str, Any]] = read_symbol_table_from_bytes(data)

    return {
        "filepath": str(filepath),
        "header": header,
        "symbols": symbols,
    }


def read_header(file: BinaryIO) -> dict[str, Any]:
    """
    Read the GDX file header from a file handle.

    Args:
        file: Open binary file handle.

    Returns:
        Dictionary with version, endianness, platform, and producer info.
    """
    file.seek(0)
    data: bytes = file.read(512)
    return read_header_from_bytes(data)


def read_header_from_bytes(data: bytes) -> dict[str, Any]:
    """
    Read the GDX file header from bytes.

    Args:
        data: Raw bytes from GDX file.

    Returns:
        Dictionary with version, endianness, platform, and producer info.
    """
    magic_pos: int = data.find(GDX_MAGIC)
    if magic_pos == -1:
        return {
            "version": 0,
            "endianness": "little",
            "platform": "unknown",
            "producer": "",
        }

    version_pos: int = magic_pos + len(GDX_MAGIC)
    version: int = struct.unpack_from("<I", data, version_pos)[0]

    # Extract producer string (after version, skip nulls)
    producer_start: int = version_pos + 4
    while producer_start < len(data) and data[producer_start] == 0:
        producer_start += 1

    producer_end: int = producer_start
    null_count: int = 0
    while producer_end < min(len(data), producer_start + 200):
        if data[producer_end] == 0:
            null_count += 1
            if null_count >= 3:
                break
        else:
            null_count = 0
        producer_end += 1

    producer: str = (
        data[producer_start:producer_end]
        .decode("utf-8", errors="replace")
        .strip("\x00")
    )

    return {
        "version": version,
        "endianness": "little",
        "platform": _detect_platform(producer),
        "producer": producer,
    }


def _detect_platform(producer: str) -> str:
    """Detect platform from producer string."""
    producer_lower: str = producer.lower()
    if "macos" in producer_lower or "darwin" in producer_lower:
        return "macOS"
    elif "linux" in producer_lower:
        return "Linux"
    elif "win" in producer_lower:
        return "Windows"
    return "unknown"


def read_symbol_table(
    file: BinaryIO, endianness: str = "little"
) -> list[dict[str, Any]]:  # noqa: ARG001
    """
    Read the symbol table from a GDX file handle.

    Args:
        file: Open binary file handle.
        endianness: Byte order (ignored, GDX is always little-endian).

    Returns:
        List of symbol dictionaries.
    """
    file.seek(0)
    data: bytes = file.read()
    return read_symbol_table_from_bytes(data)


def read_symbol_table_from_bytes(data: bytes) -> list[dict[str, Any]]:
    """
    Read the symbol table from GDX bytes.

    Args:
        data: Raw bytes from GDX file.

    Returns:
        List of symbol dictionaries with name, type, dimension, records, description.
    """
    symb_pos: int = data.find(GDX_SYMB_MARKER)
    if symb_pos == -1:
        return []

    pos: int = symb_pos + len(GDX_SYMB_MARKER)

    if pos + 4 > len(data):
        return []

    num_symbols: int = struct.unpack_from("<I", data, pos)[0]
    pos += 4

    symbols: list[dict[str, Any]] = []

    for _ in range(num_symbols):
        if pos + 35 > len(data):  # Minimum symbol size
            break

        try:
            # Name length (1 byte)
            name_len: int = data[pos]
            pos += 1

            if name_len == 0 or name_len > 64 or pos + name_len > len(data):
                break

            # Name
            name: str = data[pos : pos + name_len].decode("utf-8", errors="replace")
            pos += name_len

            # Type flag (1 byte)
            type_flag: int = data[pos]
            pos += 1

            # Metadata (25 bytes)
            # - byte 7: dimension count
            # - bytes 16-19: record count
            if pos + 25 > len(data):
                break

            dimension: int = data[pos + 7]
            records: int = struct.unpack_from("<I", data, pos + 16)[0]
            pos += 25

            # Description length (1 byte)
            desc_len: int = data[pos]
            pos += 1

            # Description
            description: str = ""
            if 0 < desc_len < 200 and pos + desc_len <= len(data):
                description = data[pos : pos + desc_len].decode(
                    "utf-8", errors="replace"
                )
                pos += desc_len

            # Padding (6 bytes)
            pos += 6

            # Map type flag to type code
            if type_flag == 0x01:
                sym_type = 0  # set
            elif type_flag == 0x20:
                sym_type = 4  # alias
            elif type_flag in (0x3F, 0x66):
                sym_type = 1  # parameter
            elif type_flag in (0x40, 0x67):
                sym_type = 2  # variable
            elif type_flag in (0x41, 0x68):
                sym_type = 3  # equation
            else:
                sym_type = type_flag

            symbols.append(
                {
                    "name": name,
                    "type": sym_type,
                    "type_name": SYMBOL_TYPE_NAMES.get(
                        sym_type, f"unknown({type_flag:#x})"
                    ),
                    "dimension": dimension,
                    "records": records,
                    "description": description,
                }
            )

        except (IndexError, struct.error):
            break

    return symbols
