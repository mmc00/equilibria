"""GDX file writer — pure-Python implementation.

CLEAN-ROOM REIMPLEMENTATION of the GDX v7 wire format. Developed by
inspecting the binary layout of files produced by GAMS and reading public
documentation. Written WITHOUT consulting, copying, translating, or deriving
from the official GAMS GDX libraries (`gdxcclib`, `gdxapi`, the C/Java/.NET
SDKs), GAMS Transfer (Python/R/Matlab), or any other third-party GDX
implementation. The interoperability constants below (magic `GAMSGDX`, marker
strings, symbol-type codes) are the wire format itself, not copyrightable
expression. Distributed under MIT (see top-level NOTICE).

Layout (GDX v7, little-endian throughout):

  Header (0x00..0xC0, padded with zeros):
    [0x00:0x12] 18-byte magic prefix (constant)
    [0x12]      pascal_short(b'GAMSGDX')          → 8 bytes
    [0x1A]      uint32 version                    → 4 bytes
    [0x1E]      3 zero bytes
    [0x21]      pascal_short(audit_string)        → audit text or empty
    [0x??]      pascal_short(producer1)           → e.g. 'GDX Library...'
    [0x??]      pascal_short(producer2)           → e.g. 'GAMS Transfer'
    [0x??:+4]   4-byte hash (we emit zeros — gdxdump tolerates)
    [0x??:+48]  6 × uint64 LE offset table → [SYMB, UEL, SETT, ACRO, SYMB_dup, DOMS]
    [..0xC0]    zero padding to 0xC0

  Body (0xC0..EOF):
    Per-symbol _DATA_ sections, then _SYMB_, _SETT_, _UEL_, _ACRO_, _DOMS_.

  Each top-level section is wrapped with its pascal-prefixed marker on both
  ends (open + count_u32 + entries + close). _DATA_ sections do not have a
  closing marker but begin with their own marker.

Symbol entry (_SYMB_) layout — set/parameter, 36 bytes payload:
    pascal_short(name)   → 1+N bytes
    uint8 type_flag      → 0xC0 set, 0xDC parameter
    8 zero bytes
    uint32 dim
    4 zero bytes
    uint32 nrec
    14 zero bytes (includes desc_len=0; longer if description present)

Set _DATA_ payload:
    'pascal_short(_DATA_)'
    uint8  dim
    uint32 nrec
    For each dim: uint32 MinElem, uint32 MaxElem
    Records (delta-encoded): each record is
      uint8 af_dim_or_delta
        if af_dim_or_delta > dim: increment last dim by (val - dim)
        else: read indices for dims [af_dim-1..dim-1] sized by elem_range
      uint8 0x05 (set marker: empty associated string)
    Trailing 0xFF byte (EOF marker)
"""

from __future__ import annotations

import struct
from datetime import datetime
from pathlib import Path

from equilibria.babel.gdx.symbols import (
    Equation,
    Parameter,
    Set,
    SymbolBase,
    Variable,
)

# ── Constants ────────────────────────────────────────────────────────────────

MAGIC_PREFIX: bytes = bytes.fromhex(
    "023412047856341208182d4454fb2109407b"
)  # 18-byte fixed magic prefix observed in all GDX v7 files

MAGIC_PASCAL: bytes = b"GAMSGDX"
GDX_VERSION: int = 7

# Beginning-Of-Information marker at file offset 0x70. Constant across all
# observed GDX v7 files. gdxdump validates this exact 4-byte value.
BOI_MARKER: bytes = bytes.fromhex("60b52901")

MARKER_SYMB: bytes = b"_SYMB_"
MARKER_UEL: bytes = b"_UEL_"
MARKER_SETT: bytes = b"_SETT_"
MARKER_DOMS: bytes = b"_DOMS_"
MARKER_DATA: bytes = b"_DATA_"
MARKER_ACRO: bytes = b"_ACRO_"

TYPE_FLAG_SET: int = 0xC0
TYPE_FLAG_SET_DOMAIN: int = 0xDC       # set used as a non-first domain (subset semantics)
TYPE_FLAG_PARAMETER: int = 0xDC        # parameter without explicit domains
TYPE_FLAG_PARAMETER_DOM: int = 0xF6    # parameter with explicit subset domains

HEADER_END: int = 0xC0  # body always starts at 0xC0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pascal(s: bytes | str) -> bytes:
    """Pascal-prefixed short string (1 length byte + bytes)."""
    if isinstance(s, str):
        s = s.encode("ascii")
    if len(s) > 255:
        raise ValueError(f"pascal_short overflow ({len(s)} > 255)")
    return bytes([len(s)]) + s


def _u32(n: int) -> bytes:
    return struct.pack("<I", n)


def _u64(n: int) -> bytes:
    return struct.pack("<Q", n)


# ── Public API ───────────────────────────────────────────────────────────────

def write_gdx(
    filepath: str | Path,
    symbols: list[SymbolBase],
    *,
    version: int = 7,
    compress: bool = False,
) -> None:
    """Write a list of symbols to a GDX file.

    Currently supports sets and parameters. Variables and equations raise.
    """
    if version != 7:
        raise ValueError(f"Only GDX version 7 is supported, got {version}")
    if compress:
        raise NotImplementedError("Compression not yet supported")

    for sym in symbols:
        if not isinstance(sym, (Set, Parameter)):
            raise NotImplementedError(
                f"Writer supports Set and Parameter only; got {type(sym).__name__}"
            )

    blob = _build_gdx(symbols)
    Path(filepath).write_bytes(blob)


# ── Builder ──────────────────────────────────────────────────────────────────

def _build_gdx(symbols: list[SymbolBase]) -> bytes:
    """Two-pass build: emit body sections to learn offsets, then emit header."""
    uel = _build_uel(symbols)
    domains = _build_domains(symbols)

    # Pass 1: emit body, recording start offset of each section.
    body = bytearray()

    # _DATA_ sections (one per symbol, in symbol order)
    data_offsets: list[int] = []
    for sym in symbols:
        data_offsets.append(HEADER_END + len(body))
        body += _emit_data_section(sym, uel)

    symb_off = HEADER_END + len(body)
    body += _emit_symb_section(symbols, data_offsets, uel)

    sett_off = HEADER_END + len(body)
    body += _emit_sett_section()

    uel_off = HEADER_END + len(body)
    body += _emit_uel_section(uel)

    acro_off = HEADER_END + len(body)
    body += _emit_acro_section()

    doms_off = HEADER_END + len(body)
    body += _emit_doms_section(domains)

    # Offset table order: [SYMB, UEL, SETT, ACRO, SYMB(dup), DOMS]
    offset_table = [symb_off, uel_off, sett_off, acro_off, symb_off, doms_off]

    header = _emit_header(offset_table)
    assert len(header) == HEADER_END, f"header size {len(header)} != {HEADER_END}"

    return bytes(header) + bytes(body)


# ── Header ───────────────────────────────────────────────────────────────────

def _emit_header(offsets: list[int]) -> bytearray:
    """Emit 0xC0 bytes of header (magic + producers + hash + offset table)."""
    h = bytearray()
    h += MAGIC_PREFIX                              # 0x00..0x12
    h += _pascal(MAGIC_PASCAL)                     # 0x12: pascal 'GAMSGDX'
    h += _u32(GDX_VERSION)                         # 0x1A: version
    h += b"\x00\x00\x00"                           # 0x1E: 3 zero bytes
    # Three pascal strings (audit, producer1, producer2) must occupy exactly
    # 76 bytes (audit_len+1 + p1_len+1 + p2_len+1 = 76) so the 4-byte hash
    # lands at file offset 0x70 and the offset table at 0x74. gdxdump rejects
    # files whose offset table is at any other position.
    h += _pascal(b"")                              # 0x21: empty audit (1 byte)
    # Producer1: must be exactly 63 chars to match ref layout.
    # Producer2: 'GAMS Transfer' (13 chars). Total: 1+63+1+13+1=79 bytes including
    # the empty audit, putting hash at 0x70.
    producer1 = (
        f"GDX Library equilibria native V7 "
        f"{datetime.now().strftime('%a %b %d %H:%M:%S %Y')} arm64 macOS"
    ).encode("ascii")
    if len(producer1) >= 63:
        producer1 = producer1[:63]
    else:
        producer1 = producer1 + b" " * (63 - len(producer1))
    h += _pascal(producer1)                        # 1+63 = 64 bytes
    h += _pascal(b"GAMS Transfer")                 # 1+13 = 14 bytes (mimic ref)
    # Now at 0x70: 4-byte Beginning-Of-Information marker. Constant across all
    # GDX v7 files generated by GAMS — gdxdump validates it strictly.
    h += BOI_MARKER
    for off in offsets:
        h += _u64(off)                             # 6 × uint64 offset table
    if len(h) > HEADER_END:
        raise RuntimeError(
            f"header overflowed 0xC0 ({len(h)} bytes)"
        )
    h += b"\x00" * (HEADER_END - len(h))
    return h


# ── _SYMB_ section ───────────────────────────────────────────────────────────

def _emit_symb_section(
    symbols: list[SymbolBase],
    data_offsets: list[int],
    uel: list[str],
) -> bytes:
    # Domain set rule (from gams.transfer refs): a set gets 0xDA only when it
    # appears as a NON-FIRST domain position in some multidim param. The first
    # domain position keeps 0xC0. Single-dim params don't trigger 0xDA at all.
    domain_set_names: set[str] = set()
    for sym in symbols:
        if isinstance(sym, Parameter) and sym.dimensions >= 2 and sym.domain:
            for pos, d in enumerate(sym.domain):
                if d != "*" and pos > 0:
                    domain_set_names.add(d)
    # 1-based symbol-table index for every named symbol (used by param domain refs)
    sym_index: dict[str, int] = {s.name: i + 1 for i, s in enumerate(symbols)}

    out = bytearray()
    out += _pascal(MARKER_SYMB)
    out += _u32(len(symbols))
    for sym, off in zip(symbols, data_offsets):
        is_domain = isinstance(sym, Set) and sym.name in domain_set_names
        out += _emit_symbol_entry(sym, off, is_domain=is_domain, sym_index=sym_index)
    out += _pascal(MARKER_SYMB)
    return bytes(out)


def _emit_symbol_entry(
    sym: SymbolBase,
    data_offset: int,
    *,
    is_domain: bool = False,
    sym_index: dict[str, int] | None = None,
) -> bytes:
    """Symbol entry layout (decoded from gams.transfer reference files).

    The leading byte sequence that earlier was treated as a 1-byte "type_flag"
    is in fact the low byte of a u64 LE *data section offset*. Earlier
    references all had data_offset < 256, which made it look like a flag.

    Layout, post pascal_short(name), is:
        data_offset_u64 (8)
        dim_u32 (4)
        kind_u32 (4)         — 0 = set, 1 = parameter
        1 zero
        nrec_u32 (4)
        5 zeros
        desc_len_u8 + desc_bytes
        + set-specific or param-specific trailing
    """
    out = bytearray()
    out += _pascal(sym.name)
    desc = sym.description.encode("ascii") if sym.description else b""

    if isinstance(sym, Set):
        kind = 0
        out += _u64(data_offset)
        out += _u32(sym.dimensions or 1)
        out += _u32(kind)
        out += b"\x00"
        out += _u32(len(sym.elements))
        out += b"\x00" * 5
        out.append(len(desc))
        out += desc
        out += b"\x00" * 6
    elif isinstance(sym, Parameter):
        kind = 1
        out += _u64(data_offset)
        out += _u32(sym.dimensions)
        out += _u32(kind)
        out += b"\x00"
        out += _u32(len(sym.records))
        out += b"\x00" * 5
        out.append(len(desc))
        out += desc
        # Trailing — varies with domain presence:
        #   scalar / no-domain  : 6 zeros          (total 12 zeros incl. pre-desc 5z)
        #   1D-with-domain      : 6z + 01 02 + 6z  (16 bytes after desc)
        #   nD-with-domain (n≥2): 6z + 01 02 + 4z + u32(n+1) + 4z (20 bytes)
        has_domain = (
            sym.dimensions >= 1
            and bool(sym.domain)
            and any(d != "*" for d in sym.domain)
        )
        if not has_domain:
            # Total post-nrec = 5z + dlen + desc + 6z = 12 bytes (no desc).
            out += b"\x00" * 6
        elif sym.dimensions == 1:
            # Total post-nrec = 16 bytes: 5z + dlen + desc + 1z + 01 01 + 7z.
            out += b"\x00"
            out += b"\x01\x01"
            out += b"\x00" * 7
        else:
            # nD-with-domain (n≥2). Trailing pattern from ref:
            #   7z + 01 01 + 3z + u32(idx_of_dim2) + ... + u32(idx_of_dimN) + 4z
            # The u32 values are 1-based symbol-table indices of each domain
            # set (only positions 2..dim — position 1 is implicit).
            out += b"\x00"
            out += b"\x01\x01"
            out += b"\x00" * 3
            for d in range(1, sym.dimensions):  # positions 2..dim
                dname = sym.domain[d]
                idx = sym_index.get(dname, 0) if sym_index else 0
                out += _u32(idx)
            out += b"\x00" * 4
    else:
        raise NotImplementedError(type(sym).__name__)
    return bytes(out)


# ── _DATA_ section ───────────────────────────────────────────────────────────

def _emit_data_section(sym: SymbolBase, uel: list[str]) -> bytes:
    """Per-symbol data section (set or parameter)."""
    out = bytearray()
    out += _pascal(MARKER_DATA)

    if isinstance(sym, Set):
        out += _emit_set_data(sym, uel)
    elif isinstance(sym, Parameter):
        out += _emit_param_data(sym, uel)
    else:
        raise NotImplementedError(type(sym).__name__)

    return bytes(out)


def _emit_set_data(sym: Set, uel: list[str]) -> bytes:
    """Emit set _DATA_ payload using delta encoding."""
    elements = sym.elements  # list[list[str]] but for 1D it's [[a],[b],[c]]
    dim = sym.dimensions or 1
    nrec = len(elements)

    out = bytearray()
    out.append(dim)
    out += _u32(nrec)

    # Compute MinElem/MaxElem per dim (1-based UEL indices)
    if nrec == 0:
        for _ in range(dim):
            out += _u32(1) + _u32(1)
        out.append(0xFF)
        return bytes(out)

    # Convert each record's elements to UEL indices
    uel_idx = {name: i + 1 for i, name in enumerate(uel)}
    rec_indices: list[list[int]] = []
    for rec in elements:
        if isinstance(rec, str):
            rec = [rec]
        rec_indices.append([uel_idx[e] for e in rec])

    min_elem = [min(r[d] for r in rec_indices) for d in range(dim)]
    max_elem = [max(r[d] for r in rec_indices) for d in range(dim)]

    for d in range(dim):
        out += _u32(min_elem[d])
        out += _u32(max_elem[d])

    # Determine elem size per dim (byte/word/int)
    elem_size = [_get_elem_size(max_elem[d] - min_elem[d] + 1) for d in range(dim)]

    # Emit records with delta encoding. First record always uses af_dim path
    # (matches GAMS reference behavior); subsequent records prefer the
    # last-dim delta short-form when only the last dim moves forward.
    last = [0] * dim
    first = True
    for r in rec_indices:
        af_dim = dim
        for d in range(dim):
            if r[d] != last[d]:
                af_dim = d + 1
                break

        use_delta = (
            not first
            and af_dim == dim
            and dim >= 1
            and r[dim - 1] > last[dim - 1]
            and (r[dim - 1] - last[dim - 1]) + dim <= 254
        )

        if use_delta:
            out.append(dim + (r[dim - 1] - last[dim - 1]))
            last[dim - 1] = r[dim - 1]
        else:
            out.append(af_dim)
            for d in range(af_dim - 1, dim):
                idx_rel = r[d] - min_elem[d]
                sz = elem_size[d]
                if sz == 1:
                    out.append(idx_rel)
                elif sz == 2:
                    out += struct.pack("<H", idx_rel)
                else:
                    out += struct.pack("<I", idx_rel)
                last[d] = r[d]
        out.append(0x05)
        first = False

    out.append(0xFF)
    return bytes(out)


def _emit_param_data(sym: Parameter, uel: list[str]) -> bytes:
    """Emit parameter _DATA_ payload using delta encoding."""
    dim = sym.dimensions
    records = sym.records
    nrec = len(records)

    out = bytearray()
    out.append(dim)
    out += _u32(nrec)

    if nrec == 0:
        for _ in range(dim):
            out += _u32(1) + _u32(1)
        out.append(0xFF)
        return bytes(out)

    uel_idx = {name: i + 1 for i, name in enumerate(uel)}
    rec_indices: list[list[int]] = []
    rec_values: list[float] = []
    for keys, val in records:
        rec_indices.append([uel_idx[k] for k in keys])
        rec_values.append(float(val))

    if dim == 0:
        # Scalar — observed layout: 01 0A double FF
        out.append(0x01)
        out.append(0x0A)
        out += struct.pack("<d", rec_values[0])
        out.append(0xFF)
        return bytes(out)

    min_elem = [min(r[d] for r in rec_indices) for d in range(dim)]
    max_elem = [max(r[d] for r in rec_indices) for d in range(dim)]
    for d in range(dim):
        out += _u32(min_elem[d])
        out += _u32(max_elem[d])

    elem_size = [_get_elem_size(max_elem[d] - min_elem[d] + 1) for d in range(dim)]

    last = [0] * dim
    first = True
    for r, val in zip(rec_indices, rec_values):
        af_dim = dim
        for d in range(dim):
            if r[d] != last[d]:
                af_dim = d + 1
                break

        use_delta = (
            not first
            and af_dim == dim
            and dim >= 1
            and r[dim - 1] > last[dim - 1]
            and (r[dim - 1] - last[dim - 1]) + dim <= 254
        )

        if use_delta:
            out.append(dim + (r[dim - 1] - last[dim - 1]))
            last[dim - 1] = r[dim - 1]
        else:
            out.append(af_dim)
            for d in range(af_dim - 1, dim):
                idx_rel = r[d] - min_elem[d]
                sz = elem_size[d]
                if sz == 1:
                    out.append(idx_rel)
                elif sz == 2:
                    out += struct.pack("<H", idx_rel)
                else:
                    out += struct.pack("<I", idx_rel)
                last[d] = r[d]
        out.append(0x0A)  # vm_normal value marker
        out += struct.pack("<d", val)
        first = False

    out.append(0xFF)
    return bytes(out)


def _get_elem_size(range_size: int) -> int:
    if range_size <= 0:
        return 4
    if range_size <= 255:
        return 1
    if range_size <= 65535:
        return 2
    return 4


# ── Other sections ───────────────────────────────────────────────────────────

def _emit_sett_section() -> bytes:
    """_SETT_: count=1, single zero byte payload (observed)."""
    out = bytearray()
    out += _pascal(MARKER_SETT)
    out += _u32(1)
    out.append(0x00)
    out += _pascal(MARKER_SETT)
    return bytes(out)


def _emit_uel_section(uel: list[str]) -> bytes:
    out = bytearray()
    out += _pascal(MARKER_UEL)
    out += _u32(len(uel))
    for el in uel:
        out += _pascal(el)
    out += _pascal(MARKER_UEL)
    return bytes(out)


def _emit_acro_section() -> bytes:
    out = bytearray()
    out += _pascal(MARKER_ACRO)
    out += _u32(0)
    out += _pascal(MARKER_ACRO)
    return bytes(out)


def _emit_doms_section(domains: list[str]) -> bytes:
    """_DOMS_: marker + count + entries + marker + 0xFFFFFFFF + marker.

    Three pascal markers wrap the section; the trailing 0xFFFFFFFF marks
    end-of-domain-table (observed across all reference files including empty).
    """
    out = bytearray()
    out += _pascal(MARKER_DOMS)
    out += _u32(len(domains))
    for d in domains:
        out += _pascal(d)
    out += _pascal(MARKER_DOMS)
    out += b"\xff\xff\xff\xff"
    out += _pascal(MARKER_DOMS)
    return bytes(out)


# ── UEL/domains construction ─────────────────────────────────────────────────

def _build_uel(symbols: list[SymbolBase]) -> list[str]:
    """Build ordered Unique Element List preserving first-seen order."""
    seen: dict[str, None] = {}
    for sym in symbols:
        if isinstance(sym, Set):
            for rec in sym.elements:
                if isinstance(rec, str):
                    seen.setdefault(rec, None)
                else:
                    for el in rec:
                        seen.setdefault(el, None)
        elif isinstance(sym, Parameter):
            for keys, _ in sym.records:
                for k in keys:
                    seen.setdefault(k, None)
    return list(seen.keys())


def _build_domains(symbols: list[SymbolBase]) -> list[str]:
    """Always emit empty _DOMS_ — observed across all gams.transfer refs.
    Domain linkage is encoded inside symbol entries via type_flag, not here."""
    return []
