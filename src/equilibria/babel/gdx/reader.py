"""GDX file reader — pure-Python implementation.

Native GDX reading without requiring GAMS installation.

CLEAN-ROOM REIMPLEMENTATION of the GDX v7 wire format. Developed by
inspecting the binary layout of files produced by GAMS and reading public
documentation. Written WITHOUT consulting, copying, translating, or deriving
from the official GAMS GDX libraries (`gdxcclib`, `gdxapi`, the C/Java/.NET
SDKs), GAMS Transfer (Python/R/Matlab), or any other third-party GDX
implementation. The interoperability constants below (magic `GAMSGDX`, marker
strings, symbol-type codes) are the wire format itself, not copyrightable
expression. Distributed under MIT (see top-level NOTICE).

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

Symbol table entry structure (after _SYMB_ marker + 4-byte count). All
little-endian; let q = offset just past the name bytes:
- 1 byte:   name length
- N bytes:  name (ASCII)
- q+0  u64: data_offset (byte offset of this symbol's _DATA_ section)
- q+8  u32: dimension
- q+12 u8:  TYPE — 0=set 1=parameter 2=variable 3=equation 4=alias (decode directly)
- q+13 u8:  userinfo (var subtype / equ type / alias target symbol index)
- q+17 u32: record count
- q+26 u8:  description length
- q+27 D:   description (ASCII)
- tail at endf=q+27+D: endf+1 dom_flag — 0x00 → 6-byte tail (no domain);
  0x01 → 2 + 4*dim + 4 bytes (per-dimension u32 domain-set indices). The tail
  length is self-describing, so advancing to the next record is deterministic.
"""

# Standard library imports
from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any, BinaryIO

# Import delta decoder for parameters
from equilibria.babel.gdx.decoder import decode_parameter_delta, get_integer_size

# var/equ _DATA_ record: 5 value fields in fixed order, each one token byte
# (a parameter record has ONE). 0x0A = an 8-byte double follows; the others are
# value codes. Only codes SEEN in a real GDX are mapped (gtap7 3x3/10x7 vs
# gdxdump): 0x01 NA (mdtx0), 0x09 2.0 (years). An unmapped code is an error --
# guessing it, or stopping there, would return a wrong or truncated symbol.
_VALUE_FIELDS: tuple[str, ...] = ("level", "marginal", "lower", "upper", "scale")
_VALUE_SPECIAL: dict[int, float] = {
    0x01: float("nan"),  # NA
    0x02: float("inf"),
    0x03: float("-inf"),
    0x05: 0.0,
    0x06: 1.0,
    0x07: -1.0,
    0x09: 2.0,
}

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


# zlib stream first byte (CMF): low nibble 8 = deflate method, common GDX values.
_ZLIB_CMF_CANDIDATES: bytes = bytes([0x78])
_ZLIB_FLG_CANDIDATES: bytes = bytes([0x9C, 0x01, 0xDA, 0x5E, 0x1D])

# Tamano del bloque de compresion de GAMS: una seccion comprimida se parte en
# bloques de 32768 bytes DESCOMPRIMIDOS, cada uno su propio stream zlib precedido
# de 3 bytes de cabecera. Un bloque mas corto que esto es el ultimo de la seccion.
_GDX_ZLIB_BLOCK: int = 32768


def _decompress_gdx_sections(data: bytes) -> bytes:
    """
    Expand a zlib-per-section GDX (v7, above some internal size threshold) into
    an equivalent uncompressed byte layout that the marker-scanning readers
    below can parse unmodified.

    GAMS switches to compressing each section (_SYMB_, _UEL_, _DATA_, _DOMS_...)
    independently with raw zlib once the file crosses a size threshold — the
    section markers themselves become invisible to a plain byte search. Detect
    every zlib stream start (CMF/FLG header bytes), decompress each one, and
    splice the decompressed bytes back into the buffer at the same relative
    position (preserving any uncompressed spans verbatim). If no zlib streams
    are found, the input is returned unchanged (uncompressed GDX, the common
    case for small files) — this makes the pre-pass a no-op safe to always run.

    Args:
        data: Raw bytes of a .gdx file.

    Returns:
        Bytes with every detected zlib section replaced by its decompressed
        content. Byte-identical to `data` if no zlib streams were detected.
    """
    candidates: list[int] = []
    start = 0
    while True:
        p = data.find(_ZLIB_CMF_CANDIDATES, start)
        if p == -1:
            break
        if p + 1 < len(data) and data[p + 1] in _ZLIB_FLG_CANDIDATES:
            candidates.append(p)
        start = p + 1

    if not candidates:
        return data

    spans: list[tuple[int, int, bytes]] = []
    last_end = 0
    for p in candidates:
        if p < last_end:
            continue  # inside a previously-consumed stream
        try:
            decompressor = zlib.decompressobj()
            out = decompressor.decompress(data[p:])
            consumed = len(data) - p - len(decompressor.unused_data)
        except zlib.error:
            continue
        if len(out) < 4:
            continue
        spans.append((p, p + consumed, out))
        last_end = p + consumed

    if not spans:
        return data

    out_buf = bytearray()
    cursor = 0
    for span_start, span_end, decompressed in spans:
        out_buf += data[cursor:span_start]
        out_buf += decompressed
        cursor = span_end
    out_buf += data[cursor:]
    return bytes(out_buf)


_raw_cache: dict[str, bytes] = {}
_expanded_cache: dict[str, bytes] = {}


def _read_gdx_bytes(filepath: str) -> bytes:
    """
    Read a .gdx file's RAW (on-disk) bytes, with a small per-filepath cache
    (the value-reader entry points below each re-read the file independently,
    once per symbol lookup).

    NOTE: these are the raw bytes, NOT zlib-expanded — ``data_offset``
    (recorded per-symbol in the symbol table) is always relative to this raw
    layout, for both compressed and uncompressed GDX. Use
    ``_section_at_offset`` to resolve a symbol's _DATA_ section from these
    bytes (it handles the per-symbol zlib-stream case itself). Use
    ``_read_gdx_expanded`` instead when you need whole-file marker scanning
    (symbol table / UEL / domains).

    Args:
        filepath: Path to the GDX file (string key, as stored in gdx_data).

    Returns:
        Raw file bytes.
    """
    cached = _raw_cache.get(filepath)
    if cached is not None:
        return cached
    raw = Path(filepath).read_bytes()
    _raw_cache[filepath] = raw
    return raw


def _read_gdx_expanded(filepath: str) -> bytes:
    """
    Read a .gdx file and return it with every per-section zlib stream
    decompressed and spliced back in place, for whole-file marker scanning
    (symbol table / UEL / domains). See ``_decompress_gdx_sections``.

    NOT suitable for per-symbol _DATA_ lookups by ``data_offset`` — those
    offsets are relative to the RAW file, and splicing shifts everything
    after the first compressed span. Use ``_read_gdx_bytes`` +
    ``_section_at_offset`` for that.
    """
    cached = _expanded_cache.get(filepath)
    if cached is not None:
        return cached
    expanded = _decompress_gdx_sections(_read_gdx_bytes(filepath))
    _expanded_cache[filepath] = expanded
    return expanded


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

    filepath_str = str(filepath)
    raw: bytes = _read_gdx_bytes(filepath_str)

    header: dict[str, Any] = read_header_from_bytes(raw)
    data: bytes = _read_gdx_expanded(filepath_str)
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

    # GDX v7 _SYMB_ record (little-endian), parsed by its real layout — NOT by the
    # old fixed-offset + skip-zeros heuristic, which read the low byte of the u64
    # data_offset as a 1-byte "type_flag" and desynced after ~52 of 453 symbols.
    # Let q = offset just after the name:
    #   q+0  u64 data_offset   q+8  u32 dim   q+12 u8 TYPE (0=set 1=par 2=var 3=equ
    #   4=alias)   q+13 u8 userinfo   q+17 u32 nrec   q+26 u8 desc_len   q+27 desc.
    # Tail at endf=q+27+desc_len: endf+0 sep, endf+1 dom_flag (0=no domain → 6-byte
    # tail; 1=explicit → 2 + 4*dim + 4 bytes of u32 domain-set indices). Advancing by
    # the declared tail length is deterministic — no zero-skipping guess.
    for _ in range(num_symbols):
        try:
            name_len: int = data[pos]
            pos += 1
            if name_len == 0 or name_len > 64 or pos + name_len > len(data):
                break
            name: str = data[pos : pos + name_len].decode("utf-8", errors="replace")
            pos += name_len

            q: int = pos
            if q + 27 > len(data):
                break
            data_offset: int = struct.unpack_from("<Q", data, q)[0]
            dimension: int = struct.unpack_from("<I", data, q + 8)[0]
            sym_type: int = data[q + 12]
            userinfo: int = data[q + 13]
            records: int = struct.unpack_from("<I", data, q + 17)[0]
            desc_len: int = data[q + 26]

            description: str = ""
            if desc_len and q + 27 + desc_len <= len(data):
                description = data[q + 27 : q + 27 + desc_len].decode(
                    "utf-8", errors="replace"
                )
            endf: int = q + 27 + desc_len
            if endf + 2 > len(data):
                break

            dom_flag: int = data[endf + 1]
            domain_idx: list[int] = []
            if dom_flag == 0x01:
                tail = 2 + 4 * dimension + 4
                for d in range(dimension):
                    if endf + 2 + 4 * d + 4 <= len(data):
                        domain_idx.append(
                            struct.unpack_from("<I", data, endf + 2 + 4 * d)[0]
                        )
            else:
                tail = 6
            pos = endf + tail  # deterministic advance to the next record

            symbols.append(
                {
                    "name": name,
                    "type": sym_type,
                    "type_name": SYMBOL_TYPE_NAMES.get(
                        sym_type, f"unknown({sym_type})"
                    ),
                    # Back-compat: the old field was the low byte of data_offset.
                    "type_flag": data[q],
                    "userinfo": userinfo,
                    "dimension": dimension,
                    "records": records,
                    "description": description,
                    "data_offset": data_offset,
                    "domain_idx": domain_idx,
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
    domain_offsets: list[int] | None = None,
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
        domain_offsets: Optional list of UEL start offsets per dimension. When
            provided, these override the auto-computed offsets from set symbols.
            Use when the GDX has parameters but no set symbols (e.g. Results.gdx).
            Example: [0, 17] means dim0 starts at UEL[0], dim1 at UEL[17].

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

    # Same path as variables/equations: the symbol's own data_offset (positional
    # lookup is wrong -- a compressed GDX has fewer _DATA_ sections than symbols,
    # and every parameter past the last one came back EMPTY), then the record
    # decoder with one value field. Keys are absolute UEL indices, so
    # ``domain_offsets`` is not needed (kept for callers that still pass it).
    del domain_offsets
    raw_data: bytes = _read_gdx_bytes(filepath)
    section = _section_at_offset(raw_data, symbol["data_offset"])
    elements: list[str] = gdx_data.get("elements", [])
    records = _decode_records(section, elements, symbol["dimension"], 1, symbol_name)
    return {key: vals[0] for key, vals in records.items()}


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


def _section_at_offset(raw_data: bytes, data_offset: int) -> bytes:
    """Return the _DATA_ section bytes for the symbol at ``data_offset``.

    ``data_offset`` is always a byte offset into the file's RAW (on-disk, not
    zlib-expanded) bytes — this holds for both compressed and uncompressed
    GDX. Two layouts:

    - Uncompressed: ``data_offset`` points directly at the 7-byte
      pascal-prefixed marker ``b"\\x06_DATA_"``; the section payload starts
      immediately after it and runs to the next marker / EOF (over-sliced;
      the per-record 0xFF EOF byte bounds decoding, so a generous slice is
      safe).
    - Compressed (GDX above GAMS's per-section zlib threshold — see
      ``_decompress_gdx_sections``): each symbol's _DATA_ section is its OWN
      independent zlib stream, NOT part of one shared expanded buffer (a
      naive whole-file splice desyncs every offset after the first
      compressed span). ``data_offset + 3`` (a 3-byte tag we don't decode)
      is the zlib stream start; decompressing it directly yields the same
      pascal-prefixed ``b"\\x06_DATA_"`` marker, self-terminated by the
      stream's own EOF (no next-marker search needed or possible — the next
      bytes in the raw file belong to a DIFFERENT compressed stream).
    """
    if raw_data[data_offset : data_offset + 7] == b"\x06_DATA_":
        start = data_offset + 7
        nxt = raw_data.find(b"_DATA_", start)
        end = (
            (nxt - 1) if nxt != -1 else len(raw_data)
        )  # -1 drops the pascal-prefix byte
        return raw_data[start:end]

    # Una seccion comprimida NO es un unico stream zlib: GAMS la parte en BLOQUES
    # de 32768 bytes descomprimidos (medido en el GDX 10x7: pa 1264 de 2940 records
    # si se lee solo el primero). Cada bloque lleva 3 bytes de cabecera: un flag
    # (1 = zlib, 0 = guardado SIN comprimir) y su largo en disco, u16 big-endian.
    # El flag 0 aparece para secciones chicas dentro de un GDX comprimido (31 en el
    # 3x3); tratarlas como zlib las devolvia VACIAS (rorgeq: 0 de 2 records).
    # Verificado: en los 771 bloques zlib del 3x3 y 10x7 el largo es exactamente el
    # del stream.
    chunks: list[bytes] = []
    pos = data_offset
    while True:
        flag = raw_data[pos]
        n = int.from_bytes(raw_data[pos + 1 : pos + 3], "big")
        block = raw_data[pos + 3 : pos + 3 + n]
        if flag == 1:
            out = zlib.decompress(block)
        elif flag == 0:
            out = block
        else:
            raise ValueError(
                f"GDX _DATA_ block at offset {pos}: unknown flag {flag} (expected 0 or 1)"
            )
        chunks.append(out)
        pos += 3 + n
        # Un bloque corto es el ultimo de la seccion.
        if len(out) < _GDX_ZLIB_BLOCK:
            break
    expanded = b"".join(chunks)
    if expanded[:7] != b"\x06_DATA_":
        raise ValueError(
            f"GDX _DATA_ section at offset {data_offset}: no _DATA_ marker"
        )
    return expanded[7:]


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

    # Read raw bytes (transparently decompressed if zlib-per-section)
    raw_data: bytes = _read_gdx_bytes(filepath)

    # Locate this symbol's _DATA_ section by its own data_offset (positional
    # indexing is wrong: there are fewer _DATA_ sections than symbols).
    section = _section_at_offset(raw_data, symbol["data_offset"])
    elements: list[str] = gdx_data.get("elements", [])
    dimension: int = symbol["dimension"]

    # Calculate domain offsets
    domain_offsets: list[int] = _calculate_domain_offsets(gdx_data, dimension)

    return _decode_variable_section(
        section, elements, dimension, domain_offsets, symbol_name
    )


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

    # Read raw bytes (transparently decompressed if zlib-per-section)
    raw_data: bytes = _read_gdx_bytes(filepath)

    # Locate this symbol's _DATA_ section by its own data_offset (positional
    # indexing is wrong: there are fewer _DATA_ sections than symbols).
    section = _section_at_offset(raw_data, symbol["data_offset"])
    elements: list[str] = gdx_data.get("elements", [])
    dimension: int = symbol["dimension"]

    # Calculate domain offsets
    domain_offsets: list[int] = _calculate_domain_offsets(gdx_data, dimension)

    return _decode_variable_section(
        section, elements, dimension, domain_offsets, symbol_name
    )


def _decode_variable_section(
    section: bytes,
    elements: list[str],
    dimension: int,
    domain_offsets: list[int],
    symbol_name: str = "",
) -> dict[tuple[str, ...], dict[str, float]]:
    """Decode a variable/equation _DATA_ section: 5 fields per record
    [level, marginal, lower, upper, scale]. See ``_decode_records``."""
    records = _decode_records(section, elements, dimension, 5, symbol_name)
    return {
        key: dict(zip(_VALUE_FIELDS, vals, strict=True))
        for key, vals in records.items()
    }


def _decode_records(
    section: bytes,
    elements: list[str],
    dimension: int,
    nfields: int,
    symbol_name: str = "",
) -> dict[tuple[str, ...], tuple[float, ...]]:
    """Decode the records of a _DATA_ section (``nfields`` values per record:
    1 for a parameter, 5 for a variable or equation).

    Format (checked against gdxdump on every Par/Var/Equ of the gtap7 3x3 and 10x7
    GDX, tests/babel/gdx/test_reader_vs_gdxdump_snapshot.py): after the 7-byte
    ``\\x06_DATA_`` marker (stripped by ``_section_at_offset``), a header
    `dim u8 | -1 i32 | dim x (minElem i32, maxElem i32)` (no min/max for scalars),
    then per record a key-delta byte (same scheme as ``decode_parameter_delta``)
    and ``nfields`` value tokens (``_VALUE_SPECIAL``; 0x0A -> 8-byte double).
    0xFF ends the section. An unknown value token is an error.
    """
    values: dict[tuple[str, ...], tuple[float, ...]] = {}
    if not section:
        return values

    pos = 0
    dim = section[pos]
    pos += 1
    pos += 4  # skip the -1 lead sentinel (i32)

    mins: list[int] = []
    sizes: list[int] = []
    for _ in range(dim):
        mn = struct.unpack_from("<i", section, pos)[0]
        pos += 4
        mx = struct.unpack_from("<i", section, pos)[0]
        pos += 4
        mins.append(mn)
        sizes.append(get_integer_size(mx - mn + 1))

    last = [0] * max(dim, 1)
    while pos < len(section):
        b = section[pos]
        pos += 1
        if b == 0xFF:  # EOF
            break

        # ---- key (delta scheme identical to decode_parameter_delta) ----
        if dim == 0:
            pass  # scalar: delta byte present, no index bytes
        elif b > dim:
            last[dim - 1] += b - dim
        else:
            afdim = b
            for d in range(afdim - 1, dim):
                sz = sizes[d]
                idx = int.from_bytes(section[pos : pos + sz], "little")
                pos += sz
                last[d] = idx + mins[d]

        if dim == 0:
            key: tuple[str, ...] = ()
        else:
            try:
                key = tuple(elements[last[d] - 1] for d in range(dim))
            except IndexError:
                break  # corrupt / over-run; stop decoding this section

        vals: list[float] = []
        try:
            for _ in range(nfields):
                tok = section[pos]
                pos += 1
                if tok == 0x0A:
                    vals.append(struct.unpack_from("<d", section, pos)[0])
                    pos += 8
                else:
                    vals.append(_VALUE_SPECIAL[tok])
        except KeyError:
            raise ValueError(
                f"GDX symbol {symbol_name!r}, record {key}: unknown value code "
                f"0x{tok:02x} (known: 0x0a + {sorted(_VALUE_SPECIAL)})"
            ) from None
        except (IndexError, struct.error):
            break  # ran off the end of an over-sliced section; stop cleanly

        values[key] = tuple(vals)

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

    # Locate the section by data_offset (not positional read_data_sections
    # indexing — there are fewer _DATA_ sections than symbols, and on a
    # zlib-per-section GDX each symbol's section is its own independent
    # compressed stream; _section_at_offset handles both layouts).
    raw_data: bytes = _read_gdx_bytes(filepath)
    section = _section_at_offset(raw_data, symbol["data_offset"])
    elements: list[str] = gdx_data.get("elements", [])

    return _decode_set_section(section, elements, dimension, records)


def _decode_set_section(
    section: bytes,
    elements: list[str],
    dimension: int,
    expected_records: int,
) -> list[tuple[str, ...]]:
    """
    Decode a _DATA_ section for a set.

    Same key-delta scheme as ``_decode_variable_section`` (see its docstring
    for the byte-verified format), except a set record carries ONE trailing
    marker byte (the element's optional explanatory-text field — 0x05 =
    empty/no text, matching ``_VALUE_SPECIAL``) instead of the 5 numeric
    value fields a variable/equation record carries. Header:
    ``dim u8 | -1 i32 | dim×(minElem i32, maxElem i32)``, then per record a
    key-delta byte (identical scheme to ``decode_parameter_delta``) followed
    by the 1-byte text marker. 0xFF ends the section. Verified byte-exact
    against gdxdump-known set contents (reg/comm/t/endw/mapa0, both a plain
    and a zlib-per-section-compressed GDX).

    Args:
        section: Raw bytes of the _DATA_ section (marker already stripped).
        elements: UEL elements list.
        dimension: Set dimension (1, 2, 3, ...).
        expected_records: Number of records expected in the set (unused for
            decoding — the stream is self-terminating via 0xFF — kept for
            the empty-section short-circuit and API compatibility).

    Returns:
        List of index tuples representing set elements.
    """
    result: list[tuple[str, ...]] = []
    if not section or expected_records == 0:
        return result

    pos = 0
    dim = section[pos]
    pos += 1
    pos += 4  # skip the -1 lead sentinel (i32)

    mins: list[int] = []
    sizes: list[int] = []
    for _ in range(dim):
        mn = struct.unpack_from("<i", section, pos)[0]
        pos += 4
        mx = struct.unpack_from("<i", section, pos)[0]
        pos += 4
        mins.append(mn)
        sizes.append(get_integer_size(mx - mn + 1))

    last = [0] * max(dim, 1)
    while pos < len(section):
        b = section[pos]
        pos += 1
        if b == 0xFF:  # EOF
            break

        if dim == 0:
            pass  # scalar set element: delta byte present, no index bytes
        elif b > dim:
            last[dim - 1] += b - dim
        else:
            afdim = b
            for d in range(afdim - 1, dim):
                sz = sizes[d]
                idx = int.from_bytes(section[pos : pos + sz], "little")
                pos += sz
                last[d] = idx + mins[d]

        if dim == 0:
            key: tuple[str, ...] = ()
        else:
            try:
                key = tuple(elements[last[d] - 1] for d in range(dim))
            except IndexError:
                break  # corrupt / over-run; stop decoding this section

        if pos >= len(section):
            break
        pos += 1  # consume the 1-byte text marker (0x05 = no text)

        result.append(key)

    return result
