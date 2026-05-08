"""GDX V7 file writer — pure Python implementation.

Produces GDX files binary-compatible with the official GAMS reader (`gdxdump`,
`gams`, `gdxcc`). Implements the format reverse-engineered from the public
``GAMS-dev/gdx`` C++ source (``gxfile.cpp``, ``gmsstrm.cpp``):

* 16-byte byte-order preamble written by ``TMiBufferedStream``
  (size markers + ``PAT_WORD`` / ``PAT_INTEGER`` / ``PAT_DOUBLE`` constants).
* Magic header ``123 + "GAMSGDX"`` (Delphi short string).
* Version (int32 = 7), compression flag (int32 = 0), audit string,
  producer string, and 10 reserved Int64 slots (the *major-index* table).
* Data sections (one per symbol), each starting with ``"_DATA_"`` and a
  simple record encoding terminated by ``0xFF``.
* ``"_SYMB_"`` symbol table, ``"_UEL_"`` UEL list, plus empty
  ``"_SETT_"`` / ``"_ACRO_"`` / ``"_DOMS_"`` lists.
* Major-index block written back at the reserved offset with ``MARK_BOI =
  19510624`` followed by the six section offsets.
"""

from __future__ import annotations

import io
import platform
import struct
from pathlib import Path

from equilibria.babel.gdx.symbols import Equation, Parameter, Set, SymbolBase, Variable

# -- byte-order preamble ---------------------------------------------------
_PAT_WORD = 0x1234
_PAT_INTEGER = 0x12345678
_PAT_DOUBLE = 3.1415926535897932385

# -- file-format constants -------------------------------------------------
_GDX_HEADER_NR = 123
_GDX_HEADER_ID = "GAMSGDX"
_GDX_VERSION = 7
_MARK_BOI = 19510624

_MARK_SYMB = "_SYMB_"
_MARK_DATA = "_DATA_"
_MARK_SETT = "_SETT_"
_MARK_UEL = "_UEL_"
_MARK_ACRO = "_ACRO_"
_MARK_DOMS = "_DOMS_"

_DATA_TYPE: dict[str, int] = {
    "set": 0,
    "parameter": 1,
    "variable": 2,
    "equation": 3,
    "alias": 4,
}

_DEFAULT_USER_INFO: dict[str, int] = {
    "set": 0,
    "parameter": 0,
    "variable": 0,
    "equation": 53,
    "alias": 0,
}

_SZ_BYTE = 0
_SZ_WORD = 1
_SZ_INTEGER = 2

# vm_normal index in TgdxIntlValTyp (gxfile.hpp).
# Order: vm_valund, vm_valna, vm_valpin, vm_valmin, vm_valeps, vm_zero,
#        vm_one, vm_mone, vm_half, vm_two, vm_normal=10
_VM_NORMAL = 10
_END_OF_DATA = 0xFF


def write_gdx(
    filepath: str | Path,
    symbols: list[SymbolBase],
    *,
    version: int = 7,
    compress: bool = False,
    producer: str = "equilibria",
) -> None:
    """Write a list of symbols to a GAMS-compatible GDX V7 file."""
    if version != 7:
        raise ValueError(f"Only GDX version 7 is supported, got {version}")
    if compress:
        raise ValueError("Compressed GDX output is not supported")

    uel = _build_uel(symbols)
    uel_index: dict[str, int] = {name: idx + 1 for idx, name in enumerate(uel)}

    buf = io.BytesIO()
    _write_preamble(buf)
    _write_byte(buf, _GDX_HEADER_NR)
    _write_short_string(buf, _GDX_HEADER_ID)
    _write_int32(buf, _GDX_VERSION)
    _write_int32(buf, 0)
    _write_short_string(buf, _audit_line())
    _write_short_string(buf, producer)

    major_index_pos = buf.tell()
    for _ in range(10):
        _write_int64(buf, 0)

    sym_positions: list[int] = []
    for symbol in symbols:
        sym_positions.append(buf.tell())
        _write_data_section(buf, symbol, uel_index)
    next_write_position = buf.tell()

    symb_pos = buf.tell()
    _write_short_string(buf, _MARK_SYMB)
    _write_int32(buf, len(symbols))
    for symbol, sym_pos in zip(symbols, sym_positions):
        _write_symbol_entry(buf, symbol, sym_pos)
    _write_short_string(buf, _MARK_SYMB)

    set_text_pos = buf.tell()
    _write_short_string(buf, _MARK_SETT)
    # SetTextList is initialized OneBased=false with one empty entry "" at slot 0
    # (gxfile.cpp:451-453). Match that or sets won't read back.
    _write_int32(buf, 1)
    _write_short_string(buf, "")
    _write_short_string(buf, _MARK_SETT)

    uel_pos = buf.tell()
    _write_short_string(buf, _MARK_UEL)
    _write_int32(buf, len(uel))
    for name in uel:
        _write_short_string(buf, name)
    _write_short_string(buf, _MARK_UEL)

    acronym_pos = buf.tell()
    _write_short_string(buf, _MARK_ACRO)
    _write_int32(buf, 0)
    _write_short_string(buf, _MARK_ACRO)

    domain_str_pos = buf.tell()
    # _DOMS_ section (gxfile.cpp:602-620):
    #   MARK_DOMS + DomainStrList(empty=count 0) + MARK_DOMS
    #   + per-symbol-with-SDomStrings entries (none) + WriteInteger(-1)
    #   + MARK_DOMS
    _write_short_string(buf, _MARK_DOMS)
    _write_int32(buf, 0)
    _write_short_string(buf, _MARK_DOMS)
    _write_int32(buf, -1)
    _write_short_string(buf, _MARK_DOMS)

    buf.seek(major_index_pos)
    _write_int32(buf, _MARK_BOI)
    for offset in (
        symb_pos,
        uel_pos,
        set_text_pos,
        acronym_pos,
        next_write_position,
        domain_str_pos,
    ):
        _write_int64(buf, offset)

    Path(filepath).write_bytes(buf.getvalue())


# --------------------------------------------------------------------------
# Stream-level primitives
# --------------------------------------------------------------------------


def _write_byte(buf: io.BytesIO, value: int) -> None:
    buf.write(struct.pack("<B", value))


def _write_int32(buf: io.BytesIO, value: int) -> None:
    buf.write(struct.pack("<i", value))


def _write_int64(buf: io.BytesIO, value: int) -> None:
    buf.write(struct.pack("<q", value))


def _write_uint16(buf: io.BytesIO, value: int) -> None:
    buf.write(struct.pack("<H", value))


def _write_uint32(buf: io.BytesIO, value: int) -> None:
    buf.write(struct.pack("<I", value))


def _write_double(buf: io.BytesIO, value: float) -> None:
    buf.write(struct.pack("<d", value))


def _write_short_string(buf: io.BytesIO, s: str) -> None:
    """Delphi ShortString: 1 byte length + N ASCII chars (max 255)."""
    data = s.encode("ascii")
    if len(data) > 255:
        raise ValueError(f"Short string too long ({len(data)} > 255): {s!r}")
    buf.write(struct.pack("<B", len(data)))
    buf.write(data)


def _write_preamble(buf: io.BytesIO) -> None:
    """16-byte byte-order preamble produced by TMiBufferedStream."""
    _write_byte(buf, 2)
    _write_uint16(buf, _PAT_WORD)
    _write_byte(buf, 4)
    _write_uint32(buf, _PAT_INTEGER)
    _write_byte(buf, 8)
    _write_double(buf, _PAT_DOUBLE)


def _audit_line() -> str:
    arch = platform.machine() or "x86_64"
    osname = platform.system() or "unknown"
    return f"GDX Library (equilibria) V7 (AUDIT) {arch} {osname}"


# --------------------------------------------------------------------------
# Per-symbol data section
# --------------------------------------------------------------------------


def _records(symbol: SymbolBase) -> list[tuple[list[str], tuple[float, ...]]]:
    """Normalise records to ``(keys, value_tuple)``."""
    if isinstance(symbol, Set):
        return [(list(elem), (0.0,)) for elem in symbol.elements]
    if isinstance(symbol, Parameter):
        return [(list(keys), (float(val),)) for keys, val in symbol.records]
    if isinstance(symbol, (Variable, Equation)):
        return [
            (list(keys), tuple(float(v) for v in vals))
            for keys, vals in symbol.records
        ]
    return []


def _value_count(symbol: SymbolBase) -> int:
    if isinstance(symbol, Set):
        return 1
    if isinstance(symbol, Parameter):
        return 1
    if isinstance(symbol, (Variable, Equation)):
        return 5
    return 0


def _record_count(symbol: SymbolBase) -> int:
    if isinstance(symbol, Set):
        return len(symbol.elements)
    if isinstance(symbol, Parameter):
        return len(symbol.records)
    if isinstance(symbol, (Variable, Equation)):
        return len(symbol.records)
    return 0


def _get_integer_size(span: int) -> int:
    """Mirror gxfile.cpp::GetIntegerSize (boundary at 255 / 65535)."""
    if span <= 0:
        return _SZ_INTEGER
    if span <= 255:
        return _SZ_BYTE
    if span <= 65535:
        return _SZ_WORD
    return _SZ_INTEGER


def _write_data_section(
    buf: io.BytesIO, symbol: SymbolBase, uel_index: dict[str, int]
) -> None:
    """``MARK_DATA + dim + nrecs + min/max[] + records + 0xFF``.

    Records always emit ``FDim = 1`` (re-write all dimensions) and serialise
    every value as ``vm_normal + IEEE-754 double``. The reader accepts this
    as a valid uncompressed encoding.
    """
    _write_short_string(buf, _MARK_DATA)
    dim = symbol.dimensions
    _write_byte(buf, dim)

    records = _records(symbol)
    _write_int32(buf, len(records))

    if dim == 0:
        _write_byte(buf, _END_OF_DATA)
        return
    if not records:
        for _ in range(dim):
            _write_int32(buf, 0)
            _write_int32(buf, 0)
        _write_byte(buf, _END_OF_DATA)
        return

    indices_per_record = [
        [uel_index[k] for k in keys] for keys, _ in records
    ]
    min_elem: list[int] = []
    max_elem: list[int] = []
    for d in range(dim):
        col = [rec[d] for rec in indices_per_record]
        min_elem.append(min(col))
        max_elem.append(max(col))
        _write_int32(buf, min_elem[d])
        _write_int32(buf, max_elem[d])

    elem_type = [_get_integer_size(max_elem[d] - min_elem[d] + 1) for d in range(dim)]

    nval = _value_count(symbol)
    for indices, (_, values) in zip(indices_per_record, records):
        _write_byte(buf, 1)  # FDim = 1: write every dimension explicitly
        for d in range(dim):
            v = indices[d] - min_elem[d]
            if elem_type[d] == _SZ_BYTE:
                _write_byte(buf, v)
            elif elem_type[d] == _SZ_WORD:
                _write_uint16(buf, v)
            else:
                _write_int32(buf, v)
        for k in range(nval):
            _write_byte(buf, _VM_NORMAL)
            _write_double(buf, values[k])

    _write_byte(buf, _END_OF_DATA)


# --------------------------------------------------------------------------
# Symbol-table entry
# --------------------------------------------------------------------------


def _write_symbol_entry(
    buf: io.BytesIO, symbol: SymbolBase, sym_position: int
) -> None:
    _write_short_string(buf, symbol.name)
    _write_int64(buf, sym_position)
    _write_int32(buf, symbol.dimensions)
    _write_byte(buf, _DATA_TYPE[symbol.sym_type])
    _write_int32(buf, _DEFAULT_USER_INFO[symbol.sym_type])
    _write_int32(buf, _record_count(symbol))
    _write_int32(buf, 0)  # error count
    _write_byte(buf, 0)   # set-text flag
    _write_short_string(buf, symbol.description or "")
    _write_byte(buf, 0)   # compression flag
    _write_byte(buf, 0)   # SDomSymbols flag (relaxed domain)
    _write_int32(buf, 0)  # comment count


# --------------------------------------------------------------------------
# UEL builder
# --------------------------------------------------------------------------


def _build_uel(symbols: list[SymbolBase]) -> list[str]:
    """Declared sets first (each contiguous, in symbol order); then any extra
    labels referenced by parameter/variable records.
    """
    uel: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if name not in seen:
            seen.add(name)
            uel.append(name)

    for symbol in symbols:
        if isinstance(symbol, Set):
            for element_list in symbol.elements:
                for el in element_list:
                    _add(el)

    for symbol in symbols:
        if isinstance(symbol, Parameter):
            for keys, _ in symbol.records:
                for k in keys:
                    _add(k)
        elif isinstance(symbol, (Variable, Equation)):
            for keys, _ in symbol.records:
                for k in keys:
                    _add(k)

    return uel
