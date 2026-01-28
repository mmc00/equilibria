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
- "_DOMS_" marker: Domain definitions
- "_DATA_" marker: Data records (one per symbol)

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
GDX_DOMS_MARKER: bytes = b"_DOMS_"
GDX_DATA_MARKER: bytes = b"_DATA_"

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
        Dictionary containing header info, symbols, elements, and domains.

    Example:
        >>> data = read_gdx("results.gdx")
        >>> print(data["header"])
        >>> for sym in data["symbols"]:
        ...     print(sym["name"], sym["type"])
        >>> print(data["elements"])  # Unique Element List
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
    elements: list[str] = read_uel_from_bytes(data)
    domains: list[str] = read_domains_from_bytes(data)

    return {
        "filepath": str(filepath),
        "header": header,
        "symbols": symbols,
        "elements": elements,
        "domains": domains,
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
    file: BinaryIO,
    endianness: str = "little",  # noqa: ARG001
) -> list[dict[str, Any]]:
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
            # GDX type flags are complex - they encode symbol index and type
            # These mappings are based on observed patterns
            if type_flag == 0x01:
                sym_type = 0  # set (first set in file)
            elif type_flag in (0x20, 0x22, 0x45):
                # Could be alias or additional set
                # 0x20 is typically alias, 0x22/0x45 are sets
                sym_type = 4 if type_flag == 0x20 else 0
            elif type_flag in (0x3F, 0x64, 0x66, 0x6E):
                sym_type = 1  # parameter
            elif type_flag in (0x40, 0x48, 0x63, 0x67, 0xFD):
                sym_type = 2  # variable
            elif type_flag in (0x41, 0x68, 0x7E, 0xD9):
                sym_type = 3  # equation
            else:
                # Unknown - keep raw value for debugging
                sym_type = type_flag

            symbols.append(
                {
                    "name": name,
                    "type": sym_type,
                    "type_name": SYMBOL_TYPE_NAMES.get(
                        sym_type, f"unknown({type_flag:#x})"
                    ),
                    "type_flag": type_flag,  # Raw flag for debugging
                    "dimension": dimension,
                    "records": records,
                    "description": description,
                }
            )

        except (IndexError, struct.error):
            break

    return symbols


def read_uel_from_bytes(data: bytes) -> list[str]:
    """
    Read the Unique Element List (UEL) from GDX bytes.

    The UEL contains all unique set elements in the GDX file.

    Args:
        data: Raw bytes from GDX file.

    Returns:
        List of element names (strings).

    Example:
        >>> elements = read_uel_from_bytes(gdx_bytes)
        >>> print(elements)
        ['agr', 'mfg', 'srv', 'food', 'goods', 'services']
    """
    uel_pos: int = data.find(GDX_UEL_MARKER)
    if uel_pos == -1:
        return []

    pos: int = uel_pos + len(GDX_UEL_MARKER)  # After '_UEL_' (5 bytes)

    if pos + 4 > len(data):
        return []

    num_elements: int = struct.unpack_from("<I", data, pos)[0]
    pos += 4

    elements: list[str] = []

    for _ in range(num_elements):
        if pos >= len(data):
            break

        elem_len: int = data[pos]
        pos += 1

        if elem_len == 0 or elem_len > 64 or pos + elem_len > len(data):
            break

        element: str = data[pos : pos + elem_len].decode("utf-8", errors="replace")
        pos += elem_len
        elements.append(element)

    return elements


def read_domains_from_bytes(data: bytes) -> list[str]:
    """
    Read domain definitions from GDX bytes.

    Args:
        data: Raw bytes from GDX file.

    Returns:
        List of domain set names.

    Example:
        >>> domains = read_domains_from_bytes(gdx_bytes)
        >>> print(domains)
        ['i', 'j']
    """
    doms_pos: int = data.find(GDX_DOMS_MARKER)
    if doms_pos == -1:
        return []

    pos: int = doms_pos + len(GDX_DOMS_MARKER)

    if pos + 4 > len(data):
        return []

    num_domains: int = struct.unpack_from("<I", data, pos)[0]
    pos += 4

    domains: list[str] = []

    for _ in range(num_domains):
        if pos >= len(data):
            break

        dom_len: int = data[pos]
        pos += 1

        if dom_len == 0 or dom_len > 64 or pos + dom_len > len(data):
            break

        domain: str = data[pos : pos + dom_len].decode("utf-8", errors="replace")
        pos += dom_len
        domains.append(domain)

    return domains


def get_symbol(gdx_data: dict[str, Any], name: str) -> dict[str, Any] | None:
    """
    Get a symbol by name from GDX data.

    Args:
        gdx_data: Result from read_gdx().
        name: Symbol name to find.

    Returns:
        Symbol dictionary or None if not found.

    Example:
        >>> data = read_gdx("model.gdx")
        >>> sym = get_symbol(data, "price")
        >>> print(sym["dimension"], sym["records"])
    """
    for sym in gdx_data.get("symbols", []):
        if sym["name"] == name:
            return sym
    return None


def get_sets(gdx_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Get all set symbols from GDX data.

    Args:
        gdx_data: Result from read_gdx().

    Returns:
        List of set symbol dictionaries.
    """
    return [s for s in gdx_data.get("symbols", []) if s["type"] == 0]


def get_parameters(gdx_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Get all parameter symbols from GDX data.

    Args:
        gdx_data: Result from read_gdx().

    Returns:
        List of parameter symbol dictionaries.
    """
    return [s for s in gdx_data.get("symbols", []) if s["type"] == 1]


def get_variables(gdx_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Get all variable symbols from GDX data.

    Args:
        gdx_data: Result from read_gdx().

    Returns:
        List of variable symbol dictionaries.
    """
    return [s for s in gdx_data.get("symbols", []) if s["type"] == 2]


def get_equations(gdx_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Get all equation symbols from GDX data.

    Args:
        gdx_data: Result from read_gdx().

    Returns:
        List of equation symbol dictionaries.
    """
    return [s for s in gdx_data.get("symbols", []) if s["type"] == 3]
