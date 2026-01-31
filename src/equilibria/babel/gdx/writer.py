"""
GDX file writer - Pure Python implementation.

This module provides native GDX file writing capabilities.
"""

from __future__ import annotations

import struct
import platform
from pathlib import Path
from typing import Any
from datetime import datetime

from equilibria.babel.gdx.symbols import SymbolBase, Set, Parameter, Variable, Equation

# GDX format constants
GDX_MAGIC: bytes = b"GAMSGDX"
GDX_VERSION: int = 7

# Markers
MARKER_SYMB: bytes = b"_SYMB_"
MARKER_UEL: bytes = b"_UEL_"
MARKER_SETT: bytes = b"_SETT_"
MARKER_DOMS: bytes = b"_DOMS_"
MARKER_DATA: bytes = b"_DATA_"
MARKER_ACRO: bytes = b"_ACRO_"

# Symbol type flags (for writing)
TYPE_FLAGS: dict[str, int] = {
    "set": 0x01,
    "parameter": 0x3F,
    "variable": 0x40,
    "equation": 0x41,
    "alias": 0x20,
}


def write_gdx(
    filepath: str | Path,
    symbols: list[SymbolBase],
    *,
    version: int = 7,
    compress: bool = False,
) -> None:
    """
    Write symbols to a GDX file.

    Args:
        filepath: Output path for the GDX file.
        symbols: List of symbols to write.
        version: GDX format version (default: 7).
        compress: Whether to compress data (default: False).

    Example:
        >>> from equilibria.babel.gdx.symbols import Parameter
        >>> from equilibria.babel.gdx.writer import write_gdx
        >>> param = Parameter(
        ...     name="price",
        ...     sym_type="parameter",
        ...     dimensions=1,
        ...     description="Market prices",
        ...     domain=["i"],
        ...     records=[(["agr"], 1.0), (["mfg"], 1.2)]
        ... )
        >>> write_gdx("output.gdx", [param])
    """
    filepath = Path(filepath)
    
    if version != 7:
        raise ValueError(f"Only GDX version 7 is supported, got {version}")
    
    # Build GDX binary data
    gdx_bytes = _build_gdx_binary(symbols, compress)
    
    # Write to file
    with open(filepath, 'wb') as f:
        f.write(gdx_bytes)


def _build_gdx_binary(symbols: list[SymbolBase], compress: bool) -> bytes:
    """Build complete GDX binary data."""
    parts: list[bytes] = []
    
    # 1. Header
    parts.append(_write_header())
    
    # 2. Build UEL (Unique Element List) from all symbols
    uel = _build_uel(symbols)
    
    # 3. Build domains list
    domains = _build_domains(symbols)
    
    # 4. Symbol table
    parts.append(_write_symbol_table(symbols, uel))
    
    # 5. Settings (empty for now)
    parts.append(_write_settings())
    
    # 6. UEL
    parts.append(_write_uel(uel))
    
    # 7. ACRO (empty for now)
    parts.append(_write_acro())
    
    # 8. Domains
    parts.append(_write_domains(domains))
    
    # 9. Data sections (one per symbol)
    for symbol in symbols:
        parts.append(_write_data_section(symbol, uel, compress))
    
    return b"".join(parts)


def _write_header() -> bytes:
    """Write GDX file header."""
    parts: list[bytes] = []
    
    # Magic checksum/metadata (19 bytes)
    # This is a simplified version - real GDX has complex checksums
    parts.append(b"4xV4-DT\xe9!\t@{")
    
    # Magic string "GAMSGDX"
    parts.append(GDX_MAGIC)
    
    # Padding
    parts.append(b"\x00" * 7)
    
    # Version number (little-endian uint32)
    parts.append(struct.pack("<I", GDX_VERSION))
    
    # Producer string
    producer = f"equilibria Python GDX Writer {datetime.now().strftime('%a %b %d %H:%M:%S %Y')}"
    producer_bytes = producer.encode('ascii')
    parts.append(struct.pack("B", len(producer_bytes)))
    parts.append(producer_bytes)
    
    # GAMS info string
    gams_info = f"Python {platform.python_version()} {platform.system()}"
    gams_bytes = gams_info.encode('ascii')
    parts.append(struct.pack("B", len(gams_bytes)))
    parts.append(gams_bytes)
    
    return b"".join(parts)


def _build_uel(symbols: list[SymbolBase]) -> list[str]:
    """Build Unique Element List from all symbols."""
    uel_set: set[str] = set()
    
    for symbol in symbols:
        if isinstance(symbol, Set):
            # Add set elements
            for element_list in symbol.elements:
                uel_set.update(element_list)
        elif isinstance(symbol, Parameter):
            # Add parameter indices
            for keys, _ in symbol.records:
                uel_set.update(keys)
        elif isinstance(symbol, (Variable, Equation)):
            # Add variable/equation indices
            for keys, _ in symbol.records:
                uel_set.update(keys)
    
    # Sort for consistent output
    return sorted(uel_set)


def _build_domains(symbols: list[SymbolBase]) -> list[str]:
    """Build domains list from symbols."""
    domains_set: set[str] = set()
    
    for symbol in symbols:
        if symbol.domain:
            domains_set.update(symbol.domain)
    
    return sorted(domains_set)


def _write_symbol_table(symbols: list[SymbolBase], uel: list[str]) -> bytes:
    """Write symbol table section."""
    parts: list[bytes] = []
    
    # Marker
    parts.append(MARKER_SYMB)
    
    # Number of symbols (4 bytes, little-endian)
    parts.append(struct.pack("<I", len(symbols)))
    
    # Write each symbol entry
    for symbol in symbols:
        parts.append(_write_symbol_entry(symbol, uel))
    
    parts.append(MARKER_SYMB)
    
    return b"".join(parts)


def _write_symbol_entry(symbol: SymbolBase, uel: list[str]) -> bytes:
    """Write a single symbol table entry."""
    parts: list[bytes] = []
    
    # Name length + name
    name_bytes = symbol.name.encode('ascii')
    parts.append(struct.pack("B", len(name_bytes)))
    parts.append(name_bytes)
    
    # Type flag
    type_flag = TYPE_FLAGS.get(symbol.sym_type, 0x3F)
    parts.append(struct.pack("B", type_flag))
    
    # Metadata (25 bytes)
    metadata = bytearray(25)
    # Byte 7: dimension
    metadata[7] = symbol.dimensions
    # Bytes 16-19: record count (little-endian)
    record_count = _get_record_count(symbol)
    struct.pack_into("<I", metadata, 16, record_count)
    parts.append(bytes(metadata))
    
    # Description length + description
    desc_bytes = symbol.description.encode('ascii') if symbol.description else b""
    parts.append(struct.pack("B", len(desc_bytes)))
    parts.append(desc_bytes)
    
    # Padding (6 bytes)
    parts.append(b"\x00" * 6)
    
    return b"".join(parts)


def _get_record_count(symbol: SymbolBase) -> int:
    """Get number of records in a symbol."""
    if isinstance(symbol, Set):
        return len(symbol.elements)
    elif isinstance(symbol, Parameter):
        return len(symbol.records)
    elif isinstance(symbol, (Variable, Equation)):
        return len(symbol.records)
    return 0


def _write_settings() -> bytes:
    """Write settings section (empty for basic implementation)."""
    return MARKER_SETT + b"\x00\x00\x00\x00" + MARKER_SETT


def _write_uel(uel: list[str]) -> bytes:
    """Write Unique Element List section."""
    parts: list[bytes] = []
    
    # Marker
    parts.append(MARKER_UEL)
    
    # Number of elements
    parts.append(struct.pack("<I", len(uel)))
    
    # Write each element
    for element in uel:
        elem_bytes = element.encode('ascii')
        parts.append(struct.pack("B", len(elem_bytes)))
        parts.append(elem_bytes)
    
    parts.append(MARKER_UEL)
    
    return b"".join(parts)


def _write_acro() -> bytes:
    """Write ACRO section (empty for basic implementation)."""
    return MARKER_ACRO + b"\x00\x00\x00\x00" + MARKER_ACRO


def _write_domains(domains: list[str]) -> bytes:
    """Write domains section."""
    parts: list[bytes] = []
    
    # Marker
    parts.append(MARKER_DOMS)
    
    # Write domain names
    for domain in domains:
        domain_bytes = domain.encode('ascii')
        parts.append(struct.pack("B", len(domain_bytes)))
        parts.append(domain_bytes)
    
    # End markers
    parts.append(b"\x00" * 4)
    parts.append(MARKER_DOMS)
    
    return b"".join(parts)


def _write_data_section(symbol: SymbolBase, uel: list[str], compress: bool) -> bytes:
    """Write data section for a symbol."""
    parts: list[bytes] = []
    
    # Data marker
    parts.append(MARKER_DATA)
    
    # Header (19 bytes of metadata)
    parts.append(b"\x01" + b"\xff" * 10 + b"\x00" * 4 + b"\xff" * 4)
    
    # Write data based on symbol type
    if isinstance(symbol, Set):
        parts.append(_write_set_data(symbol, uel))
    elif isinstance(symbol, Parameter):
        parts.append(_write_parameter_data(symbol, uel, compress))
    elif isinstance(symbol, (Variable, Equation)):
        parts.append(_write_variable_data(symbol, uel))
    
    return b"".join(parts)


def _write_set_data(symbol: Set, uel: list[str]) -> bytes:
    """Write set element data."""
    if not symbol.elements:
        return b""
    
    parts: list[bytes] = []
    
    for element_list in symbol.elements:
        # For 1D sets
        if len(element_list) == 1:
            idx = uel.index(element_list[0]) + 1  # 1-based indexing
            parts.append(b"\x02" + struct.pack("B", idx))
    
    return b"".join(parts)


def _write_parameter_data(symbol: Parameter, uel: list[str], compress: bool) -> bytes:
    """Write parameter data."""
    if not symbol.records:
        return b""
    
    parts: list[bytes] = []
    
    for keys, value in symbol.records:
        # Write indices
        for key in keys:
            idx = uel.index(key) + 1  # 1-based indexing
            parts.append(struct.pack("B", idx))
        
        # Write value (double)
        parts.append(b"\x0a")  # Double marker
        parts.append(struct.pack("<d", value))
    
    return b"".join(parts)


def _write_variable_data(symbol: Variable | Equation, uel: list[str]) -> bytes:
    """Write variable/equation data (5 attributes per record)."""
    if not symbol.records:
        return b""
    
    parts: list[bytes] = []
    
    for keys, values in symbol.records:
        # Write indices
        for key in keys:
            idx = uel.index(key) + 1  # 1-based indexing
            parts.append(struct.pack("B", idx))
        
        # Write 5 doubles: level, marginal, lower, upper, scale
        level, marginal, lower, upper, scale = values
        parts.append(b"\x0a" + struct.pack("<d", level))
        parts.append(b"\x0a" + struct.pack("<d", marginal))
        parts.append(b"\x0a" + struct.pack("<d", lower))
        parts.append(b"\x0a" + struct.pack("<d", upper))
        parts.append(b"\x0a" + struct.pack("<d", scale))
    
    return b"".join(parts)
