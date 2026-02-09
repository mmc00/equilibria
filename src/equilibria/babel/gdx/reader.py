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

# Import delta decoder for parameters
from equilibria.babel.gdx.decoder import decode_parameter_delta

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

    # Refine unknown symbol types (-1) by inspecting data sections
    _refine_symbol_types(symbols, data)

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


def _refine_symbol_types(symbols: list[dict[str, Any]], data: bytes) -> None:
    """
    Refine unknown symbol types by inspecting data sections.
    
    Some type_flags (like 0x3F) are ambiguous and can represent either
    sets or parameters. Sets have only index tuples, while parameters
    have numeric values. This function examines the data section to
    distinguish between them.
    
    Args:
        symbols: List of symbol dicts (modified in-place).
        data: Raw GDX file bytes.
    """
    data_sections = read_data_sections(data)
    
    for idx, symbol in enumerate(symbols):
        if symbol["type"] != -1:  # Only process unknown types
            continue
            
        # Get this symbol's data section
        if idx >= len(data_sections):
            # No data section - default to parameter
            symbol["type"] = 1
            symbol["type_name"] = "parameter"
            continue
            
        _, section = data_sections[idx]
        
        # Check if data section contains float values
        # Sets only have index tuples (small integers)
        # Parameters have doubles (8-byte floats)
        has_floats = _data_section_has_floats(section)
        
        if has_floats:
            symbol["type"] = 1  # parameter
            symbol["type_name"] = "parameter"
        else:
            symbol["type"] = 0  # set
            symbol["type_name"] = "set"


def _data_section_has_floats(section: bytes) -> bool:
    """
    Check if a data section contains float values.
    
    Returns True if floats detected (parameter), False otherwise (set).
    """
    if len(section) < 20:
        return False
        
    # Look for RECORD_DOUBLE marker (0x0A) which indicates float data
    # Parameters have this, sets don't
    pos = 0
    while pos < len(section) - 8:
        if section[pos] == 0x0A:  # RECORD_DOUBLE
            return True
        pos += 1
    
    return False


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

            # Variable padding: skip zeros until we find next symbol or end
            # Minimum padding is 6 bytes, but can be more for parameters with domains
            min_padding: int = 6
            pos += min_padding

            # Skip any additional zero bytes (domain info padding)
            while pos < len(data) and data[pos] == 0:
                pos += 1

            # Map type flag to type code
            # For sets: type_flag is usually 0x01 or 0x20 (alias)
            # For parameters: type_flag varies
            # NOTE: 0x01 with high dimension (>4) can be ambiguous
            # NOTE: 0x3F is ambiguous - can be either set or parameter
            # Sets don't have numeric values, parameters do
            if type_flag == 0x01 and dimension > 4:
                # High-dimensional sparse parameters often use 0x01
                # Mark as unknown and refine later based on data section
                sym_type = -1  # unknown
            elif type_flag == 0x01 and dimension > 0:
                sym_type = 0  # set
            elif type_flag in (0x01, 0x20, 0x22, 0x45):
                sym_type = 4 if type_flag == 0x20 else 0
            elif type_flag in (0x3F, 0x64, 0x66, 0x6E):
                # 0x3F can be either set or parameter
                # Default to parameter unless proven otherwise
                # In real GDX files, this is refined by checking data sections
                # For standalone symbol table reading, default to parameter
                sym_type = 1  # parameter
            elif type_flag in (0x40, 0x48, 0x63, 0x67, 0xFD):
                sym_type = 2  # variable
            elif type_flag in (0x41, 0x68, 0x7E, 0xD9):
                sym_type = 3  # equation
            else:
                # Heuristic: if dimension > 0 and not a known set flag, likely parameter
                if dimension > 0 and type_flag not in (0x01, 0x20):
                    sym_type = -1  # unknown
                else:
                    sym_type = type_flag

            symbols.append(
                {
                    "name": name,
                    "type": sym_type,
                    "type_name": SYMBOL_TYPE_NAMES.get(
                        sym_type, f"unknown({type_flag:#x})"
                    ),
                    "type_flag": type_flag,
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


# =============================================================================
# Data Reading Functions
# =============================================================================

# Record type markers in _DATA_ sections
RECORD_DOUBLE: int = 0x0A  # Double value follows (8 bytes)
RECORD_ROW_START: int = 0x01  # New row/dimension block
RECORD_RECORD: int = 0x02  # Data record marker
RECORD_CONTINUE: int = 0x03  # Continue in same row


def read_data_sections(data: bytes) -> list[tuple[int, bytes]]:
    """
    Find all _DATA_ sections in GDX bytes.

    Args:
        data: Raw bytes from GDX file.

    Returns:
        List of (symbol_index, section_bytes) tuples.
    """
    sections: list[tuple[int, bytes]] = []
    positions: list[int] = []

    # Find all _DATA_ markers
    pos: int = 0
    while True:
        pos = data.find(GDX_DATA_MARKER, pos)
        if pos == -1:
            break
        positions.append(pos)
        pos += 6

    # Extract each section
    for i, start in enumerate(positions):
        end: int = positions[i + 1] if i + 1 < len(positions) else len(data)

        # Find end by looking for next marker
        for marker in [GDX_SYMB_MARKER, GDX_UEL_MARKER, GDX_DOMS_MARKER]:
            marker_pos: int = data.find(marker, start + 6)
            if marker_pos != -1 and marker_pos < end:
                end = marker_pos

        section: bytes = data[start:end]

        # Symbol index is typically indicated after header
        # Using position in file as proxy for now
        sections.append((i, section))

    return sections


def read_parameter_values(
    gdx_data: dict[str, Any],
    symbol_name: str,
) -> dict[tuple[str, ...], float]:
    """
    Read parameter values from GDX data.

    Extracts the actual numeric values for a parameter symbol.
    Uses the UEL (Unique Element List) to map indices to element names.

    Note: GDX files may use data compression. This function reads
    explicitly stored values; compressed/delta-encoded values may
    not be fully decoded.

    Args:
        gdx_data: Result from read_gdx().
        symbol_name: Name of the parameter to read.

    Returns:
        Dictionary mapping index tuples to values.
        For 1D parameter: {("agr",): 1.5, ("mfg",): 2.0, ...}
        For 2D parameter: {("agr", "food"): 100.0, ...}

    Example:
        >>> data = read_gdx("model.gdx")
        >>> prices = read_parameter_values(data, "price")
        >>> print(prices)
        {('agr',): 1.5, ('mfg',): 2.0, ('srv',): 2.5}
    """
    symbol = get_symbol(gdx_data, symbol_name)
    if symbol is None:
        raise ValueError(f"Symbol '{symbol_name}' not found in GDX data")

    if symbol["type"] != 1:  # Not a parameter
        raise ValueError(
            f"Symbol '{symbol_name}' is not a parameter (type={symbol['type_name']})"
        )

    filepath: str = gdx_data.get("filepath", "")
    if not filepath:
        raise ValueError("GDX data missing filepath - cannot read raw bytes")

    # Read raw bytes
    raw_data: bytes = Path(filepath).read_bytes()

    # Find the _DATA_ section for this symbol
    symbols: list[dict[str, Any]] = gdx_data["symbols"]
    symbol_index: int = -1
    for i, sym in enumerate(symbols):
        if sym["name"] == symbol_name:
            symbol_index = i
            break

    if symbol_index == -1:
        return {}

    # Get data sections
    data_sections = read_data_sections(raw_data)
    if symbol_index >= len(data_sections):
        return {}

    _, section = data_sections[symbol_index]
    elements: list[str] = gdx_data.get("elements", [])
    dimension: int = symbol["dimension"]
    expected_records: int = symbol.get("records", 0)

    # Calculate domain offsets based on set definitions
    # Each set in the GDX occupies a contiguous range in the UEL
    domain_offsets: list[int] = _calculate_domain_offsets(gdx_data, dimension)

    return _decode_parameter_section(
        section, elements, dimension, domain_offsets, expected_records
    )


def _calculate_domain_offsets(
    gdx_data: dict[str, Any],
    dimension: int,
) -> list[int]:
    """
    Calculate UEL offsets for each dimension based on set order.

    In GDX files, sets are stored sequentially in the UEL.
    For a parameter over (i, j), we need to know:
    - i's offset (usually 0)
    - j's offset (usually len(i))

    Args:
        gdx_data: Result from read_gdx().
        dimension: Number of dimensions.

    Returns:
        List of offsets for each dimension.
    """
    if dimension == 0:
        return []

    # Get all sets in order
    sets: list[dict[str, Any]] = get_sets(gdx_data)

    # Build offsets based on set sizes
    offsets: list[int] = []
    current_offset: int = 0

    for i in range(dimension):
        if i < len(sets):
            offsets.append(current_offset)
            current_offset += sets[i].get("records", 0)
        else:
            offsets.append(current_offset)

    return offsets


def _decode_parameter_section(
    section: bytes,
    elements: list[str],
    dimension: int,
    domain_offsets: list[int] | None = None,
    expected_records: int = 0,
) -> dict[tuple[str, ...], float]:
    """
    Decode a _DATA_ section for a parameter.

    Handles GDX compression where arithmetic sequences have values
    compressed using interpolation/extrapolation patterns.

    Args:
        section: Raw bytes of the _DATA_ section.
        elements: UEL elements list.
        dimension: Parameter dimension.
        domain_offsets: Offset in UEL for each dimension.
        expected_records: Expected number of records (from symbol table).

    Returns:
        Dictionary mapping index tuples to values.
    """
    values: dict[tuple[str, ...], float] = {}

    if len(section) < 20:
        return values

    if domain_offsets is None:
        domain_offsets = [0] * dimension

    if dimension == 1:
        return _decode_1d_parameter(section, elements, domain_offsets, expected_records)
    elif dimension == 2:
        return _decode_2d_parameter(section, elements, domain_offsets, expected_records)
    else:
        # Fallback for higher dimensions
        return _decode_simple_parameter(section, elements, dimension, domain_offsets)


def _detect_sequence_type(values: list[tuple[int, float]]) -> tuple[str, float]:
    """
    Detect whether a sequence is arithmetic or geometric progression.
    
    Args:
        values: List of (index, value) tuples from the data stream.
    
    Returns:
        Tuple of (progression_type, parameter) where:
        - progression_type is "arithmetic" or "geometric"
        - parameter is the delta (arithmetic) or ratio (geometric)
    
    This function uses heuristics since GDX format doesn't explicitly
    encode the progression type.
    """
    import math
    
    if len(values) < 2:
        return ("arithmetic", 0.0)
    
    # Extract first two values to calculate parameters
    idx1, val1 = values[0]
    idx2, val2 = values[1]
    gap = idx2 - idx1
    
    if gap == 0:
        return ("arithmetic", 0.0)
    
    # Calculate both parameters
    delta = (val2 - val1) / gap
    
    if val1 != 0 and abs(val1) > 1e-10:
        ratio = (val2 / val1) ** (1.0 / gap)
    else:
        ratio = 1.0
    
    # Case 1: Three or more values - use variance analysis
    if len(values) >= 3:
        # Test arithmetic: check if deltas are consistent
        deltas = []
        for i in range(len(values) - 1):
            idx_a, val_a = values[i]
            idx_b, val_b = values[i + 1]
            gap_local = idx_b - idx_a
            if gap_local > 0:
                deltas.append((val_b - val_a) / gap_local)
        
        # Test geometric: check if ratios are consistent
        ratios = []
        for i in range(len(values) - 1):
            idx_a, val_a = values[i]
            idx_b, val_b = values[i + 1]
            gap_local = idx_b - idx_a
            if val_a != 0 and abs(val_a) > 1e-10 and gap_local > 0:
                ratios.append((val_b / val_a) ** (1.0 / gap_local))
        
        if deltas and ratios:
            # Calculate coefficient of variation for both
            mean_delta = sum(deltas) / len(deltas)
            variance_delta = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
            cv_delta = (variance_delta ** 0.5) / abs(mean_delta) if mean_delta != 0 else float('inf')
            
            mean_ratio = sum(ratios) / len(ratios)
            variance_ratio = sum((r - mean_ratio) ** 2 for r in ratios) / len(ratios)
            cv_ratio = (variance_ratio ** 0.5) / abs(mean_ratio) if mean_ratio != 0 else float('inf')
            
            # Lower CV indicates better fit
            if cv_delta < cv_ratio * 0.5:  # Arithmetic is clearly better
                return ("arithmetic", mean_delta)
            elif cv_ratio < cv_delta * 0.5:  # Geometric is clearly better
                return ("geometric", mean_ratio)
            else:
                # Both fit similarly, use ratio proximity to 1 as tie-breaker
                if abs(mean_ratio - 1.0) < 0.1:
                    return ("arithmetic", mean_delta)
                else:
                    return ("geometric", mean_ratio)
    
    # Case 2: Only two values - use enhanced heuristics
    if len(values) == 2:
        # RULE 1: Exact powers of 2 → geometric
        if abs(ratio - round(ratio)) < 0.01 and round(ratio) >= 2:
            return ("geometric", ratio)
        
        # RULE 2: Check if val2 = val1 * 2^n (exact power)
        if val1 > 0 and val2 > 0:
            try:
                log_ratio = math.log2(val2 / val1)
                if abs(log_ratio - round(log_ratio)) < 0.01:
                    return ("geometric", ratio)
            except:
                pass
        
        # RULE 3: Large values with large delta → arithmetic
        if abs(val1) >= 50:
            avg_val = (abs(val1) + abs(val2)) / 2
            if abs(delta) >= avg_val * 0.5:
                return ("arithmetic", delta)
        
        # RULE 4: Ratio near 1 → arithmetic
        if 0.85 < ratio < 1.15:
            return ("arithmetic", delta)
        
        # RULE 5: Small values with significant ratio → geometric
        if abs(val1) < 50 and (ratio > 1.3 or ratio < 0.77):
            return ("geometric", ratio)
        
        # RULE 6: Large total change but moderate ratio → arithmetic
        rel_change_ratio = abs((val2 - val1) / val1) if val1 != 0 else 0
        if rel_change_ratio > 1.0 and ratio < 1.6:
            return ("arithmetic", delta)
        
        # RULE 7: Significant ratio → geometric
        if ratio > 1.2:
            return ("geometric", ratio)
    
    # Default fallback
    return ("arithmetic", delta)


def _decode_1d_parameter(
    section: bytes,
    elements: list[str],
    domain_offsets: list[int],
    expected_records: int,
) -> dict[tuple[str, ...], float]:
    """Decode a 1D parameter with compression support."""
    values: dict[tuple[str, ...], float] = {}

    if len(section) < 20:
        return values

    # Parse the byte stream to extract values and skip markers in order
    stream: list[tuple[str, float | None]] = []

    pos: int = 19  # Skip header

    while pos < len(section) - 2:
        byte: int = section[pos]

        # Row start marker: 01 XX 00 00 00 (skip for 1D)
        if (
            byte == RECORD_ROW_START
            and pos + 5 <= len(section)
            and section[pos + 2] == 0x00
            and section[pos + 3] == 0x00
            and section[pos + 4] == 0x00
        ):
            pos += 5
            continue

        # Block size marker
        if byte == 0x06 and pos + 1 < len(section):
            pos += 1
            continue

        # Compression marker: 02 09 = ONE interpolated value
        if byte == 0x02 and pos + 1 < len(section) and section[pos + 1] == 0x09:
            stream.append(("skip", None))
            pos += 2
            continue

        # Record type 02 followed by double marker 0a
        if (
            byte == 0x02
            and pos + 10 <= len(section)
            and section[pos + 1] == RECORD_DOUBLE
        ):
            try:
                value: float = struct.unpack_from("<d", section, pos + 2)[0]
                if -1e15 < value < 1e15 and value == value:
                    stream.append(("value", value))
            except struct.error:
                pass
            pos += 10
            continue

        # Standalone double marker 0a
        if byte == RECORD_DOUBLE and pos + 9 <= len(section):
            try:
                value = struct.unpack_from("<d", section, pos + 1)[0]
                if -1e15 < value < 1e15 and value == value:
                    stream.append(("value", value))
            except struct.error:
                pass
            pos += 9
            continue

        # Skip marker 03 (sparse)
        if byte == RECORD_CONTINUE:
            stream.append(("skip", None))
            pos += 1
            continue

        pos += 1

    # Count values and skips
    stored_values: list[float] = [
        v for t, v in stream if t == "value" and v is not None
    ]
    num_stored: int = len(stored_values)
    num_skips_in_stream: int = sum(1 for t, _ in stream if t == "skip")

    if expected_records == 0:
        expected_records = num_stored + num_skips_in_stream

    # Build logical-to-value mapping
    all_values: dict[int, float] = {}

    if num_skips_in_stream == 0 and num_stored == expected_records:
        # No compression - direct mapping
        for i, val in enumerate(stored_values):
            all_values[i] = val
    elif num_stored >= 2:
        # Compression detected
        stream_represents: int = num_stored + num_skips_in_stream
        leading_missing: int = expected_records - stream_represents

        logical_idx: int = leading_missing

        for t, v in stream:
            if t == "value" and v is not None:
                all_values[logical_idx] = v
            logical_idx += 1

        # Detect progression type (arithmetic vs geometric) and calculate parameters
        stored_logical_indices: list[int] = sorted(all_values.keys())
        progression_type: str = "arithmetic"
        delta: float = 0.0
        ratio: float = 1.0
        
        if len(stored_logical_indices) >= 2:
            # Collect sample values for detection
            sample_values: list[tuple[int, float]] = [
                (idx, all_values[idx]) for idx in stored_logical_indices[:min(3, len(stored_logical_indices))]
            ]
            
            # Detect progression type using heuristics
            progression_type, param = _detect_sequence_type(sample_values)
            
            if progression_type == "arithmetic":
                delta = param
            else:  # geometric
                ratio = param

        # Fill missing values using detected interpolation method
        for i in range(expected_records):
            if i not in all_values:
                prev_idx: int | None = None
                next_idx: int | None = None
                for k in stored_logical_indices:
                    if k < i:
                        prev_idx = k
                    elif k > i and next_idx is None:
                        next_idx = k
                        break

                if prev_idx is not None and next_idx is not None:
                    prev_val: float = all_values[prev_idx]
                    next_val: float = all_values[next_idx]
                    gap: int = next_idx - prev_idx
                    
                    # Apply appropriate interpolation based on detected type
                    if progression_type == "arithmetic":
                        interp_delta: float = (next_val - prev_val) / gap
                        all_values[i] = prev_val + (i - prev_idx) * interp_delta
                    else:  # geometric
                        if prev_val != 0 and abs(prev_val) > 1e-10:
                            interp_ratio: float = (next_val / prev_val) ** (1.0 / gap)
                            all_values[i] = prev_val * (interp_ratio ** (i - prev_idx))
                        else:
                            # Fallback to arithmetic if division by zero
                            interp_delta = (next_val - prev_val) / gap
                            all_values[i] = prev_val + (i - prev_idx) * interp_delta
                            
                elif prev_idx is not None:
                    # Extrapolate forward
                    if progression_type == "arithmetic":
                        all_values[i] = all_values[prev_idx] + (i - prev_idx) * delta
                    else:  # geometric
                        all_values[i] = all_values[prev_idx] * (ratio ** (i - prev_idx))
                        
                elif next_idx is not None:
                    # Extrapolate backward
                    if progression_type == "arithmetic":
                        all_values[i] = all_values[next_idx] - (next_idx - i) * delta
                    else:  # geometric
                        all_values[i] = all_values[next_idx] / (ratio ** (next_idx - i))
    else:
        for i, val in enumerate(stored_values):
            all_values[i] = val

    # Convert indices to element tuples
    for idx, val in all_values.items():
        if idx < 0 or idx >= expected_records:
            continue

        index_tuple: tuple[str, ...] = _build_index_tuple_with_offsets(
            [idx], elements, 1, domain_offsets
        )
        if index_tuple:
            values[index_tuple] = val

    return values


def _decode_2d_parameter(
    section: bytes,
    elements: list[str],
    domain_offsets: list[int],
    expected_records: int,  # noqa: ARG001
) -> dict[tuple[str, ...], float]:
    """Decode a 2D parameter (row-major storage)."""
    values: dict[tuple[str, ...], float] = {}

    if len(section) < 20:
        return values

    pos: int = 19  # Skip header
    current_row: int = 0
    col_idx: int = 0

    while pos < len(section) - 2:
        byte: int = section[pos]

        # Row start marker: 01 XX 00 00 00
        if (
            byte == RECORD_ROW_START
            and pos + 5 <= len(section)
            and section[pos + 2] == 0x00
            and section[pos + 3] == 0x00
            and section[pos + 4] == 0x00
        ):
            current_row = section[pos + 1] - 1  # 1-indexed
            col_idx = 0
            pos += 5
            continue

        # Skip 4-byte blocks
        if (
            byte in (0x04, 0x06, 0x08)
            and pos + 4 <= len(section)
            and section[pos + 1 : pos + 4] == b"\x00\x00\x00"
        ):
            pos += 4
            continue

        # Double value: 0a + 8 bytes
        if byte == RECORD_DOUBLE and pos + 9 <= len(section):
            try:
                value: float = struct.unpack_from("<d", section, pos + 1)[0]
                if -1e15 < value < 1e15 and value == value:
                    index_tuple: tuple[str, ...] = _build_index_tuple_with_offsets(
                        [current_row, col_idx], elements, 2, domain_offsets
                    )
                    if index_tuple:
                        values[index_tuple] = value
                    col_idx += 1
            except struct.error:
                pass
            pos += 9
            continue

        pos += 1

    return values


def _decode_simple_parameter(
    section: bytes,
    elements: list[str],
    dimension: int,
    domain_offsets: list[int],
) -> dict[tuple[str, ...], float]:
    """
    Decoder for higher-dimensional parameters (3D, 4D, etc.).
    
    Format for 3D+ parameters follows similar pattern to sets:
    - First row marker: 01 <dim1> 00 00 00 <dim2_int32> <dim3_int32> ... 0a <double>
    - Continuation values in same "slice": 0a <double>
    - New slice: 01 <dim1> 00 00 00 <dim2_int32> <dim3_int32> ... 0a <double>
    
    The structure is essentially row-major where the last dimension varies fastest.
    
    KNOWN ISSUE - Parameter Delta Encoding:
    This decoder assumes parameters use the same delta encoding format as sets,
    but real-world GDX files (e.g., SAM-V2_0.gdx) use a DIFFERENT compression
    scheme for parameters. The current implementation fails to decode parameters
    with complex delta compression.
    
    WHAT WORKS:
    - Sets (2D, 3D, 4D) with delta encoding: ✓ Fully working
    - Simple parameters without compression: ✓ Working
    
    WHAT DOESN'T WORK:
    - Parameters with delta compression (like SAM-V2_0.gdx): ✗ Returns 0 values
    
    INVESTIGATION NEEDED:
    The SAM-V2_0.gdx file has 196 parameter records that should decode to tuples
    like ('AG', 'USK', 'J', 'AGR') = 1500.0, but the decoder returns empty.
    
    Root cause: The decoder expects pattern "01 XX 00 00 00" at position 11,
    but actual data has different byte patterns (e.g., "01 00 00 00 1B...").
    
    The actual format appears to be:
    - Position 11: 01 (record start)
    - Position 12: 00 (dim1 = 0, not AG=7 as expected)
    - Positions 13+: Compressed/delta-encoded indices
    
    The delta encoding for parameters is DIFFERENT from sets and requires
    reverse-engineering the specific compression algorithm used by GAMS.
    
    RECOMMENDED FIX:
    1. Analyze the binary structure of SAM-V2_0.gdx more deeply
    2. Identify the specific delta encoding pattern for parameters
    3. Implement a separate decoder for parameter delta compression
    4. Test against the CSV reference file (gdx_values.csv)
    
    See GitHub issue for detailed binary analysis.
    """
    values: dict[tuple[str, ...], float] = {}

    if len(section) < 20 or dimension < 3:
        # Fallback to simple extraction for edge cases
        return _decode_simple_parameter_fallback(section, elements, domain_offsets)

    # Try the new delta decoder first
    # This handles compressed parameters like SAM-V2_0.gdx
    try:
        values = decode_parameter_delta(section, elements, dimension)
        if len(values) > 0:
            return values
    except Exception:
        # If delta decoder fails, fall back to old method
        pass

    # Legacy decoder (for simple parameters without compression)
    pos: int = 11  # Skip header: _DATA_ (6) + dimension (1) + record_count (4)
    current_indices: list[int] = []  # Current tuple indices (0-based UEL indices)

    while pos < len(section) - 1:
        byte: int = section[pos]
        
        # Pattern 1: Full tuple specification: 01 <dim1> 00 00 00 <int32>*N 0a <double>
        if (byte == RECORD_ROW_START and 
            pos + 4 < len(section) and
            section[pos + 2] == 0x00 and
            section[pos + 3] == 0x00 and
            section[pos + 4] == 0x00):
            
            dim1_idx = section[pos + 1]
            
            if dim1_idx < 1 or dim1_idx > len(elements):
                pos += 1
                continue
            
            # For 3D: read 2 int32s (dim2, dim3)
            # For 4D: read 3 int32s (dim2, dim3, dim4)
            # For ND: read (N-1) int32s
            num_int32s = dimension - 1
            header_size = 5 + (num_int32s * 4)  # 01 XX 00 00 00 + int32s
            
            # Check if we have space for header + marker + double
            if pos + header_size + 9 > len(section):
                pos += 1
                continue
            
            try:
                # Build UEL indices - ALL are 1-based and need conversion to 0-based
                uel_indices = [dim1_idx - 1]
                
                # Read int32 values for remaining dimensions
                # These are also 1-based UEL indices (need to subtract 1)
                for i in range(num_int32s):
                    offset = pos + 5 + (i * 4)
                    uel_idx_1based = struct.unpack_from("<I", section, offset)[0]
                    
                    if uel_idx_1based < 1 or uel_idx_1based > len(elements):
                        raise ValueError("Invalid UEL index")
                    
                    uel_indices.append(uel_idx_1based - 1)
                
                # Check for double marker after indices
                double_marker_pos = pos + header_size
                if double_marker_pos >= len(section) or section[double_marker_pos] != RECORD_DOUBLE:
                    pos += 1
                    continue
                
                # Read the double value
                value = struct.unpack_from("<d", section, double_marker_pos + 1)[0]
                
                # Build tuple directly from UEL indices
                if all(0 <= idx < len(elements) for idx in uel_indices):
                    index_tuple = tuple(elements[idx] for idx in uel_indices)
                    values[index_tuple] = value
                    current_indices = uel_indices
                    pos += header_size + 9  # Skip header + 0a + double
                    continue
                    
            except (struct.error, ValueError):
                pass
        
        # Pattern 2: Partial tuple update: 02 <new_dim2_byte> 00 00 00 <int32>*(N-2) 0a <double>
        # This pattern updates dimension 2 (and possibly later dims) while keeping dim1 same
        elif (byte == 0x02 and 
              len(current_indices) == dimension and
              pos + 5 <= len(section)):
            
            try:
                # Read new value for dimension 2 (1-based byte)
                new_dim2_1based = section[pos + 1]
                
                if new_dim2_1based < 1 or new_dim2_1based > len(elements):
                    pos += 1
                    continue
                
                # For 3D: need 1 more int32 (dim3)
                # For 4D: need 2 more int32s (dim3, dim4)
                num_additional_int32s = dimension - 2
                partial_header_size = 5 + (num_additional_int32s * 4)  # 02 XX 00 00 00 + int32s
                
                if pos + partial_header_size + 9 > len(section):
                    pos += 1
                    continue
                
                # Build new indices: keep dim1, update dim2 and remaining
                new_indices = [current_indices[0], new_dim2_1based - 1]
                
                # Read int32s for remaining dimensions
                for i in range(num_additional_int32s):
                    offset = pos + 5 + (i * 4)
                    uel_idx_1based = struct.unpack_from("<I", section, offset)[0]
                    
                    if uel_idx_1based < 1 or uel_idx_1based > len(elements):
                        raise ValueError("Invalid UEL index")
                    
                    new_indices.append(uel_idx_1based - 1)
                
                # Check for double marker
                double_marker_pos = pos + partial_header_size
                if double_marker_pos >= len(section) or section[double_marker_pos] != RECORD_DOUBLE:
                    pos += 1
                    continue
                
                # Read value
                value = struct.unpack_from("<d", section, double_marker_pos + 1)[0]
                
                # Store value
                if all(0 <= idx < len(elements) for idx in new_indices):
                    index_tuple = tuple(elements[idx] for idx in new_indices)
                    values[index_tuple] = value
                    current_indices = new_indices
                    pos += partial_header_size + 9
                    continue
                    
            except (struct.error, ValueError):
                pass
        
        # Pattern 3: Update last N-2 dimensions: 03 <int32> <int32> ... 0a <double>
        # Keep dim1 and dim2, update remaining dimensions
        # For 4D: 03 <dim3_int32> <dim4_int32> 0a <double>
        elif (byte == 0x03 and 
              len(current_indices) == dimension and
              dimension >= 3 and
              pos + 1 <= len(section)):
            
            try:
                # For 4D: need 2 int32s (dim3, dim4)
                # For 5D: need 3 int32s (dim3, dim4, dim5)
                num_update_int32s = dimension - 2
                pattern3_size = 1 + (num_update_int32s * 4)  # 03 + int32s
                
                if pos + pattern3_size + 9 > len(section):
                    pos += 1
                    continue
                
                # Build new indices: keep dim1 and dim2, update remaining
                new_indices = current_indices[:2].copy()
                
                # Read int32s for dimensions 3+
                for i in range(num_update_int32s):
                    offset = pos + 1 + (i * 4)
                    uel_idx_1based = struct.unpack_from("<I", section, offset)[0]
                    
                    if uel_idx_1based < 1 or uel_idx_1based > len(elements):
                        raise ValueError("Invalid UEL index")
                    
                    new_indices.append(uel_idx_1based - 1)
                
                # Check for double marker
                double_marker_pos = pos + pattern3_size
                if double_marker_pos >= len(section) or section[double_marker_pos] != RECORD_DOUBLE:
                    pos += 1
                    continue
                
                # Read value
                value = struct.unpack_from("<d", section, double_marker_pos + 1)[0]
                
                # Store value
                if all(0 <= idx < len(elements) for idx in new_indices):
                    index_tuple = tuple(elements[idx] for idx in new_indices)
                    values[index_tuple] = value
                    current_indices = new_indices
                    pos += pattern3_size + 9
                    continue
                    
            except (struct.error, ValueError):
                pass
        
        # Pattern 4: Update last N-3 dimensions: 04 <int32> <int32> ... 0a <double>
        # Keep dim1, dim2, and dim3, update remaining dimensions
        # For 5D: 04 <dim4_int32> <dim5_int32> 0a <double>
        # For 6D: 04 <dim4_int32> <dim5_int32> <dim6_int32> 0a <double>
        elif (byte == 0x04 and 
              len(current_indices) == dimension and
              dimension >= 4 and
              pos + 1 <= len(section)):
            
            try:
                # For 5D: need 2 int32s (dim4, dim5)
                # For 6D: need 3 int32s (dim4, dim5, dim6)
                num_update_int32s = dimension - 3
                pattern4_size = 1 + (num_update_int32s * 4)  # 04 + int32s
                
                if pos + pattern4_size + 9 > len(section):
                    pos += 1
                    continue
                
                # Build new indices: keep dim1, dim2, dim3, update remaining
                new_indices = current_indices[:3].copy()
                
                # Read int32s for dimensions 4+
                for i in range(num_update_int32s):
                    offset = pos + 1 + (i * 4)
                    uel_idx_1based = struct.unpack_from("<I", section, offset)[0]
                    
                    if uel_idx_1based < 1 or uel_idx_1based > len(elements):
                        raise ValueError("Invalid UEL index")
                    
                    new_indices.append(uel_idx_1based - 1)
                
                # Check for double marker
                double_marker_pos = pos + pattern4_size
                if double_marker_pos >= len(section) or section[double_marker_pos] != RECORD_DOUBLE:
                    pos += 1
                    continue
                
                # Read value
                value = struct.unpack_from("<d", section, double_marker_pos + 1)[0]
                
                # Store value
                if all(0 <= idx < len(elements) for idx in new_indices):
                    index_tuple = tuple(elements[idx] for idx in new_indices)
                    values[index_tuple] = value
                    current_indices = new_indices
                    pos += pattern4_size + 9
                    continue
                    
            except (struct.error, ValueError):
                pass
        
        # Pattern 5: Update last N-4 dimensions: 05 <int32> <int32> ... 0a <double>
        # Keep dim1, dim2, dim3, and dim4, update remaining dimensions
        # For 6D: 05 <dim5_int32> <dim6_int32> 0a <double>
        # For 7D: 05 <dim5_int32> <dim6_int32> <dim7_int32> 0a <double>
        elif (byte == 0x05 and 
              len(current_indices) == dimension and
              dimension >= 5 and
              pos + 1 <= len(section)):
            
            try:
                # For 6D: need 2 int32s (dim5, dim6)
                # For 7D: need 3 int32s (dim5, dim6, dim7)
                num_update_int32s = dimension - 4
                pattern5_size = 1 + (num_update_int32s * 4)  # 05 + int32s
                
                if pos + pattern5_size + 9 > len(section):
                    pos += 1
                    continue
                
                # Build new indices: keep dim1-4, update remaining
                new_indices = current_indices[:4].copy()
                
                # Read int32s for dimensions 5+
                for i in range(num_update_int32s):
                    offset = pos + 1 + (i * 4)
                    uel_idx_1based = struct.unpack_from("<I", section, offset)[0]
                    
                    if uel_idx_1based < 1 or uel_idx_1based > len(elements):
                        raise ValueError("Invalid UEL index")
                    
                    new_indices.append(uel_idx_1based - 1)
                
                # Check for double marker
                double_marker_pos = pos + pattern5_size
                if double_marker_pos >= len(section) or section[double_marker_pos] != RECORD_DOUBLE:
                    pos += 1
                    continue
                
                # Read value
                value = struct.unpack_from("<d", section, double_marker_pos + 1)[0]
                
                # Store value
                if all(0 <= idx < len(elements) for idx in new_indices):
                    index_tuple = tuple(elements[idx] for idx in new_indices)
                    values[index_tuple] = value
                    current_indices = new_indices
                    pos += pattern5_size + 9
                    continue
                    
            except (struct.error, ValueError):
                pass
                
                # Check for double marker after indices
                double_marker_pos = pos + header_size
                if double_marker_pos >= len(section) or section[double_marker_pos] != RECORD_DOUBLE:
                    pos += 1
                    continue
                
                # Read the double value
                value = struct.unpack_from("<d", section, double_marker_pos + 1)[0]
                
                # Build tuple directly from UEL indices
                if all(0 <= idx < len(elements) for idx in uel_indices):
                    index_tuple = tuple(elements[idx] for idx in uel_indices)
                    values[index_tuple] = value
                    current_indices = uel_indices
                    pos += header_size + 9  # Skip header + 0a + double
                    continue
                    
            except (struct.error, ValueError):
                pass
        
        # Look for continuation values: 0a <double> (increments last dimension)
        elif (len(current_indices) == dimension and
              byte == RECORD_DOUBLE and
              pos + 9 <= len(section)):
            
            try:
                value = struct.unpack_from("<d", section, pos + 1)[0]
                
                # Create new tuple with incremented last dimension
                new_uel_indices = current_indices.copy()
                new_uel_indices[-1] += 1
                
                if all(idx < len(elements) for idx in new_uel_indices):
                    index_tuple = tuple(elements[idx] for idx in new_uel_indices)
                    values[index_tuple] = value
                    current_indices = new_uel_indices
                    pos += 9
                    continue
                    
            except struct.error:
                pass
        
        pos += 1

    return values


def _decode_simple_parameter_fallback(
    section: bytes,
    elements: list[str],
    domain_offsets: list[int],
) -> dict[tuple[str, ...], float]:
    """Simple fallback decoder - extracts all doubles sequentially."""
    values: dict[tuple[str, ...], float] = {}

    if len(section) < 20:
        return values

    # Simple extraction of all double values
    pos: int = 19
    idx: int = 0

    while pos < len(section) - 8:
        if section[pos] == RECORD_DOUBLE:
            try:
                value: float = struct.unpack_from("<d", section, pos + 1)[0]
                if -1e15 < value < 1e15 and value == value:
                    # Simple linear index mapping
                    rel_indices: list[int] = [idx]
                    index_tuple: tuple[str, ...] = _build_index_tuple_with_offsets(
                        rel_indices, elements, 1, domain_offsets
                    )
                    if index_tuple:
                        values[index_tuple] = value
                    idx += 1
            except struct.error:
                pass
            pos += 9
        else:
            pos += 1

    return values


def _build_index_tuple_with_offsets(
    rel_indices: list[int],
    elements: list[str],
    dimension: int,
    domain_offsets: list[int],
) -> tuple[str, ...]:
    """
    Build an index tuple from relative indices using domain offsets.

    Args:
        rel_indices: List of relative indices within each domain.
        elements: UEL elements list.
        dimension: Expected dimension.
        domain_offsets: Offset in UEL for each dimension.

    Returns:
        Tuple of element names, or empty tuple if invalid.
    """
    if dimension == 0:
        return ()

    result: list[str] = []
    for i in range(min(dimension, len(rel_indices))):
        # Apply domain offset to get absolute UEL index
        offset: int = domain_offsets[i] if i < len(domain_offsets) else 0
        abs_idx: int = rel_indices[i] + offset

        if 0 <= abs_idx < len(elements):
            result.append(elements[abs_idx])
        else:
            # Invalid index
            return ()

    return tuple(result)


# =============================================================================
# Variable and Equation Reading Functions
# =============================================================================


def read_variable_values(
    gdx_data: dict[str, Any],
    symbol_name: str,
) -> dict[tuple[str, ...], dict[str, float]]:
    """
    Read variable values from GDX data.

    Variables in GAMS have 5 attributes: level, marginal, lower, upper, scale.
    This function extracts the level values (solve results).

    Args:
        gdx_data: Result from read_gdx().
        symbol_name: Name of the variable to read.

    Returns:
        Dictionary mapping index tuples to attribute dicts.
        {("agr",): {"level": 100.0, "marginal": 0.0, "lower": 0.0,
                    "upper": inf, "scale": 1.0}, ...}

    Example:
        >>> data = read_gdx("results.gdx")
        >>> x_values = read_variable_values(data, "X")
        >>> print(x_values[("agr",)]["level"])
        100.0
    """
    symbol = get_symbol(gdx_data, symbol_name)
    if symbol is None:
        raise ValueError(f"Symbol '{symbol_name}' not found in GDX data")

    if symbol["type"] != 2:  # Not a variable
        raise ValueError(
            f"Symbol '{symbol_name}' is not a variable (type={symbol['type_name']})"
        )

    filepath: str = gdx_data.get("filepath", "")
    if not filepath:
        raise ValueError("GDX data missing filepath - cannot read raw bytes")

    # Read raw bytes
    raw_data: bytes = Path(filepath).read_bytes()

    # Find the _DATA_ section for this symbol
    symbols: list[dict[str, Any]] = gdx_data["symbols"]
    symbol_index: int = -1
    for i, sym in enumerate(symbols):
        if sym["name"] == symbol_name:
            symbol_index = i
            break

    if symbol_index == -1:
        return {}

    # Get data sections
    data_sections = read_data_sections(raw_data)
    if symbol_index >= len(data_sections):
        return {}

    _, section = data_sections[symbol_index]
    elements: list[str] = gdx_data.get("elements", [])
    dimension: int = symbol["dimension"]

    # Calculate domain offsets
    domain_offsets: list[int] = _calculate_domain_offsets(gdx_data, dimension)

    return _decode_variable_section(section, elements, dimension, domain_offsets)


def read_equation_values(
    gdx_data: dict[str, Any],
    symbol_name: str,
) -> dict[tuple[str, ...], dict[str, float]]:
    """
    Read equation values from GDX data.

    Equations have the same 5 attributes as variables: level, marginal, lower, upper, scale.
    The marginal value represents the dual value or shadow price.

    Args:
        gdx_data: Result from read_gdx().
        symbol_name: Name of the equation to read.

    Returns:
        Dictionary mapping index tuples to attribute dicts.

    Example:
        >>> data = read_gdx("results.gdx")
        >>> eq_values = read_equation_values(data, "eq_balance")
        >>> print(eq_values[("agr",)]["marginal"])
        1.5
    """
    symbol = get_symbol(gdx_data, symbol_name)
    if symbol is None:
        raise ValueError(f"Symbol '{symbol_name}' not found in GDX data")

    if symbol["type"] != 3:  # Not an equation
        raise ValueError(
            f"Symbol '{symbol_name}' is not an equation (type={symbol['type_name']})"
        )

    filepath: str = gdx_data.get("filepath", "")
    if not filepath:
        raise ValueError("GDX data missing filepath - cannot read raw bytes")

    # Read raw bytes
    raw_data: bytes = Path(filepath).read_bytes()

    # Find the _DATA_ section for this symbol
    symbols: list[dict[str, Any]] = gdx_data["symbols"]
    symbol_index: int = -1
    for i, sym in enumerate(symbols):
        if sym["name"] == symbol_name:
            symbol_index = i
            break

    if symbol_index == -1:
        return {}

    # Get data sections
    data_sections = read_data_sections(raw_data)
    if symbol_index >= len(data_sections):
        return {}

    _, section = data_sections[symbol_index]
    elements: list[str] = gdx_data.get("elements", [])
    dimension: int = symbol["dimension"]

    # Calculate domain offsets
    domain_offsets: list[int] = _calculate_domain_offsets(gdx_data, dimension)

    return _decode_variable_section(section, elements, dimension, domain_offsets)


def _decode_variable_section(
    section: bytes,
    elements: list[str],
    dimension: int,
    domain_offsets: list[int],
) -> dict[tuple[str, ...], dict[str, float]]:
    """
    Decode a _DATA_ section for a variable or equation.

    Variables and equations store 5 values per record:
    - level: Solution value
    - marginal: Reduced cost (variables) or dual value (equations)
    - lower: Lower bound
    - upper: Upper bound
    - scale: Scale factor

    Args:
        section: Raw bytes of the _DATA_ section.
        elements: UEL elements list.
        dimension: Symbol dimension.
        domain_offsets: Offset in UEL for each dimension.

    Returns:
        Dictionary mapping index tuples to attribute dicts.
    """
    values: dict[tuple[str, ...], dict[str, float]] = {}

    if len(section) < 20:
        return values

    pos: int = 19  # Skip header
    current_indices: list[int] = []

    while pos < len(section) - 40:  # Need at least 5 doubles (40 bytes)
        byte: int = section[pos]

        # Row start marker for multi-dimensional variables
        if (
            byte == RECORD_ROW_START
            and pos + 5 <= len(section)
            and section[pos + 2] == 0x00
            and section[pos + 3] == 0x00
            and section[pos + 4] == 0x00
        ):
            if dimension > 0:
                current_indices = [section[pos + 1] - 1]
            pos += 5
            continue

        # Skip 4-byte control blocks
        if (
            byte in (0x04, 0x06, 0x08)
            and pos + 4 <= len(section)
            and section[pos + 1 : pos + 4] == b"\x00\x00\x00"
        ):
            pos += 4
            continue

        # Look for sequence of 5 double values
        if byte == RECORD_DOUBLE and pos + 41 <= len(section):
            try:
                # Read 5 consecutive doubles
                level: float = struct.unpack_from("<d", section, pos + 1)[0]
                marginal: float = struct.unpack_from("<d", section, pos + 10)[0]
                lower: float = struct.unpack_from("<d", section, pos + 19)[0]
                upper: float = struct.unpack_from("<d", section, pos + 28)[0]
                scale: float = struct.unpack_from("<d", section, pos + 37)[0]

                # Verify we have valid markers between doubles
                if (
                    section[pos + 9] == RECORD_DOUBLE
                    and section[pos + 18] == RECORD_DOUBLE
                    and section[pos + 27] == RECORD_DOUBLE
                    and section[pos + 36] == RECORD_DOUBLE
                ):
                    # Build index tuple
                    if dimension == 0:
                        index_tuple: tuple[str, ...] = ()
                    elif dimension == 1:
                        idx: int = len(values)
                        index_tuple = _build_index_tuple_with_offsets(
                            [idx], elements, 1, domain_offsets
                        )
                    else:
                        # For multi-dimensional, use current_indices
                        col_idx: int = len(
                            [
                                k
                                for k in values.keys()
                                if len(k) > 0 and k[0] == elements[current_indices[0]]
                            ]
                        )
                        full_indices: list[int] = current_indices + [col_idx]
                        index_tuple = _build_index_tuple_with_offsets(
                            full_indices, elements, dimension, domain_offsets
                        )

                    if index_tuple is not None and index_tuple != ():
                        values[index_tuple] = {
                            "level": level,
                            "marginal": marginal,
                            "lower": lower,
                            "upper": upper,
                            "scale": scale,
                        }

                    pos += 45  # Skip past all 5 doubles
                    continue

            except struct.error:
                pass

        pos += 1

    return values


def read_set_elements(
    gdx_data: dict[str, Any],
    set_name: str,
) -> list[tuple[str, ...]]:
    """
    Read set elements from GDX data.

    Args:
        gdx_data: Result from read_gdx().
        set_name: Name of the set to read.

    Returns:
        List of element tuples. For 1D set: [("agr",), ("mfg",), ("srv",)]
        For 2D set: [("agr", "food"), ("mfg", "goods"), ...]

    Example:
        >>> data = read_gdx("model.gdx")
        >>> elements = read_set_elements(data, "i")
        >>> print(elements)
        [('agr',), ('mfg',), ('srv',)]
    """
    symbol = get_symbol(gdx_data, set_name)
    if symbol is None:
        raise ValueError(f"Symbol '{set_name}' not found in GDX data")

    if symbol["type"] != 0:  # Not a set
        raise ValueError(
            f"Symbol '{set_name}' is not a set (type={symbol['type_name']})"
        )

    filepath: str = gdx_data.get("filepath", "")
    if not filepath:
        raise ValueError("GDX data missing filepath - cannot read raw bytes")

    dimension: int = symbol["dimension"]
    records: int = symbol.get("records", 0)

    if dimension == 0 or records == 0:
        return []

    # Read raw bytes
    raw_data: bytes = Path(filepath).read_bytes()

    # Find the _DATA_ section for this symbol
    symbols: list[dict[str, Any]] = gdx_data["symbols"]
    symbol_index: int = -1
    for i, sym in enumerate(symbols):
        if sym["name"] == set_name:
            symbol_index = i
            break

    if symbol_index == -1:
        return []

    # Get data sections
    data_sections = read_data_sections(raw_data)
    if symbol_index >= len(data_sections):
        return []

    _, section = data_sections[symbol_index]
    elements: list[str] = gdx_data.get("elements", [])

    # Calculate domain offsets
    domain_offsets: list[int] = _calculate_domain_offsets(gdx_data, dimension)

    return _decode_set_section(section, elements, dimension, domain_offsets, records)


def _decode_set_section(
    section: bytes,
    elements: list[str],
    dimension: int,
    domain_offsets: list[int],
    expected_records: int,
) -> list[tuple[str, ...]]:
    """
    Decode a _DATA_ section for a set.

    Sets store index tuples. The format can be complex, so we use a pragmatic
    approach: for 1D sets, we take the first N elements from the UEL that
    correspond to the set's domain. For multi-dimensional sets, we parse the
    binary structure.

    Args:
        section: Raw bytes of the _DATA_ section.
        elements: UEL elements list.
        dimension: Set dimension (1, 2, 3, etc.).
        domain_offsets: Offset in UEL for each dimension.
        expected_records: Number of records expected in the set.

    Returns:
        List of index tuples representing set elements.
    """
    result: list[tuple[str, ...]] = []

    if len(section) < 20 or expected_records == 0:
        return result

    # For 1D sets, use a simple approach: take elements from the UEL
    # starting at the domain offset
    if dimension == 1 and len(domain_offsets) > 0:
        start_idx = domain_offsets[0]
        end_idx = start_idx + expected_records
        for i in range(start_idx, min(end_idx, len(elements))):
            result.append((elements[i],))
        return result

    # For multi-dimensional sets, parse the binary structure  
    # Format patterns discovered through reverse engineering:
    # 2D: 01 <dim1> 00 00 00 <dim2_int32> 05 [<delta+2> 05]...
    # 3D: 01 <dim1> 00 00 00 <dim2_int32> <dim3_int32> 05 [<delta+2> 05]...
    # 4D: 01 <dim1> 00 00 00 <dim2_int32> <dim3_int32> <dim4_int32> 05 [<delta+2> 05]...
    # First tuple has full int32s, subsequent tuples in same row use delta encoding
    pos: int = 19  # Skip header
    current_indices: list[int] = []  # Current tuple being built (0-based indices)

    while pos < len(section) - 1:
        # Look for tuple start: 01 <dim1> 00 00 00
        if (section[pos] == RECORD_ROW_START and 
            pos + 4 < len(section) and
            section[pos + 2] == 0x00 and
            section[pos + 3] == 0x00 and
            section[pos + 4] == 0x00):
            
            dim1_idx = section[pos + 1] - 1  # Convert from 1-based to 0-based
            
            if dim1_idx >= len(elements):
                pos += 1
                continue
            
            # For 2D: read 1 int32 (dim2)
            # For 3D: read 2 int32s (dim2, dim3)
            # For 4D: read 3 int32s (dim2, dim3, dim4)
            num_int32s = dimension - 1
            bytes_needed = 5 + (num_int32s * 4) + 1  # header(5) + int32s + marker(1)
            
            if pos + bytes_needed > len(section):
                pos += 1
                continue
            
            try:
                indices = [dim1_idx]
                
                # Read int32 values for remaining dimensions
                for i in range(num_int32s):
                    offset = pos + 5 + (i * 4)
                    dim_idx_1based = struct.unpack_from("<I", section, offset)[0]
                    
                    if dim_idx_1based < 1 or dim_idx_1based > len(elements):
                        raise ValueError("Invalid index")
                    
                    indices.append(dim_idx_1based - 1)
                
                # Check for marker after all int32s
                marker_pos = pos + 5 + (num_int32s * 4)
                marker = section[marker_pos]
                
                if marker not in (0x05, 0x06):
                    pos += 1
                    continue
                
                # Validate all indices and create tuple
                if all(idx < len(elements) for idx in indices):
                    tuple_elem = tuple(elements[idx] for idx in indices)
                    result.append(tuple_elem)
                    current_indices = indices
                    pos += bytes_needed
                    continue
                    
            except (struct.error, ValueError):
                pass
        
        # Look for additional tuples using delta encoding
        # Pattern: <delta_byte> 05 means update dimension(s)
        # The delta_byte encodes which dimension changes and by how much
        # For 2D: delta_byte >= 2 means increment last dim by (delta_byte - 2)
        # For 3D+: delta_byte indicates DimFrst (first changing dimension)
        # For 4D: [delta_byte] [byte] [int32_dim3] [int32_dim4] [padding] [marker]
        elif (len(current_indices) == dimension and
              pos + 1 < len(section)):
            
            # Check if this looks like a delta pattern
            # Either: section[pos + 1] is a marker (0x05/0x06) for 2D/3D
            # Or: dimension >= 4 and there's a marker at pos + 13 (4D pattern)
            is_delta_pattern = (
                section[pos + 1] in (0x05, 0x06) or  # 2D/3D pattern
                (dimension >= 4 and pos + 14 <= len(section) and section[pos + 13] in (0x05, 0x06))  # 4D pattern
            )
            
            if not is_delta_pattern:
                pos += 1
                continue
            
            delta_byte = section[pos]
            
            if dimension == 2 and delta_byte >= 2:
                # 2D: delta_byte - 2 = increment for last dimension
                delta = delta_byte - 2
                if delta >= 1:
                    new_indices = current_indices.copy()
                    new_last_dim_0based = new_indices[-1] + delta
                    
                    if new_last_dim_0based < len(elements):
                        new_indices[-1] = new_last_dim_0based
                        
                        if all(idx < len(elements) for idx in new_indices):
                            tuple_elem = tuple(elements[idx] for idx in new_indices)
                            result.append(tuple_elem)
                            current_indices = new_indices
                            pos += 2
                            continue
            
            elif dimension >= 3 and 1 <= delta_byte <= len(elements):
                # 3D+: delta_byte indicates the new value for a dimension
                # Try different interpretations based on context
                new_indices = current_indices.copy()
                
                # Check for 4D pattern: [delta_byte] [byte] [int32_dim3] [int32_dim4] [padding] [marker]
                # Total: 1 + 1 + 4 + 4 + 3 + 1 = 14 bytes
                if (dimension >= 4 and pos + 14 <= len(section) and 
                    section[pos + 13] in (0x05, 0x06)):
                    # 4D update: delta_byte + 1 = dim2, int32s for dim3 and dim4
                    try:
                        # delta_byte + 1 = new dim2 index (0-based)
                        new_dim2 = delta_byte + 1
                        if new_dim2 < len(elements):
                            new_indices[1] = new_dim2
                        
                        # Read int32s for dim3 and dim4 (big-endian, 1-based)
                        dim3_1based = struct.unpack_from(">I", section, pos + 2)[0]
                        dim4_1based = struct.unpack_from(">I", section, pos + 6)[0]
                        
                        new_dim3 = dim3_1based - 1
                        new_dim4 = dim4_1based - 1
                        
                        if new_dim3 < len(elements):
                            new_indices[2] = new_dim3
                        if new_dim4 < len(elements):
                            new_indices[3] = new_dim4
                        
                        if all(idx < len(elements) for idx in new_indices):
                            tuple_elem = tuple(elements[idx] for idx in new_indices)
                            result.append(tuple_elem)
                            current_indices = new_indices
                            pos += 14  # Skip full pattern
                            continue
                    except struct.error:
                        pass
                
                # Check if there's an int32 after the marker (complex update for 3D)
                # Pattern: [delta_byte] [marker] [int32_dim3 (4 bytes)] [padding (3 bytes)] [marker]
                # Total: 1 + 1 + 4 + 3 + 1 = 10 bytes
                if (pos + 10 <= len(section) and 
                    section[pos + 1] in (0x05, 0x06) and
                    section[pos + 9] in (0x05, 0x06)):
                    # Complex update: delta_byte for dim2, int32 for dim3
                    try:
                        # delta_byte + 2 = new dim2 index (0-based)
                        new_dim2 = delta_byte + 2
                        if new_dim2 < len(elements):
                            new_indices[1] = new_dim2
                        
                        # Read int32 for dim3 (big-endian, 1-based, so subtract 1)
                        dim3_1based = struct.unpack_from(">I", section, pos + 2)[0]
                        new_dim3 = dim3_1based - 1
                        if new_dim3 < len(elements):
                            new_indices[-1] = new_dim3
                        
                        if all(idx < len(elements) for idx in new_indices):
                            tuple_elem = tuple(elements[idx] for idx in new_indices)
                            result.append(tuple_elem)
                            current_indices = new_indices
                            pos += 10  # Skip delta, marker, int32, padding, marker
                            continue
                    except struct.error:
                        pass
                
                # Simple update: delta_byte + 2 = new last dim index (0-based)
                new_idx = delta_byte + 2
                if new_idx < len(elements):
                    new_indices[-1] = new_idx
                    
                    if all(idx < len(elements) for idx in new_indices):
                        tuple_elem = tuple(elements[idx] for idx in new_indices)
                        result.append(tuple_elem)
                        current_indices = new_indices
                        pos += 2
                        continue
        
        pos += 1

    # Remove duplicates while preserving order
    seen = set()
    unique_result = []
    for elem in result:
        if elem not in seen:
            seen.add(elem)
            unique_result.append(elem)
    
    return unique_result

