"""Shared Fortran-record + HAR-format primitives.

Single source of truth for the on-disk byte layout of GEMPACK HAR files.
Both reader.py and writer.py import from here so encode/decode pairs stay
in lockstep.

CLEAN-ROOM REIMPLEMENTATION — see reader.py and the babel/har/README.md
for the full provenance statement. Distributed under MIT.
"""
from __future__ import annotations

import struct
from typing import Iterator

# ── Wire constants ───────────────────────────────────────────────────────────

PAD = b"    "                       # 4-byte ASCII space padding used in
                                    # name and meta records
INT = struct.Struct("<i")           # 4-byte little-endian int

NAME_WIDTH = 4                      # header name field
LONG_NAME_WIDTH = 70                # long-name field inside the meta record
SET_NAME_WIDTH = 12                 # per-element + per-set-name width
META_RECORD_MIN_LEN = 80            # pad(4) + type(6) + long(70)

# Type tokens — exactly 6 ASCII bytes, fixed
TOKEN_1CFULL = "1CFULL"
TOKEN_REFULL = "REFULL"
TOKEN_RESPSE = "RESPSE"
TOKEN_2IFULL = "2IFULL"


# ── Fortran record framing ───────────────────────────────────────────────────

def iter_records(buf: bytes) -> Iterator[bytes]:
    """Yield raw record bytes from a Fortran unformatted stream.

    Each record is framed by a 4-byte little-endian length prefix and the
    same length suffix.
    """
    pos = 0
    n = len(buf)
    while pos < n:
        if pos + 4 > n:
            return
        rlen = INT.unpack_from(buf, pos)[0]
        pos += 4
        if rlen < 0 or pos + rlen + 4 > n:
            return
        yield buf[pos:pos + rlen]
        pos += rlen + 4  # skip the trailing length marker


def write_record(out: bytearray, payload: bytes) -> None:
    """Append a Fortran-framed record: length prefix + payload + length suffix."""
    out.extend(INT.pack(len(payload)))
    out.extend(payload)
    out.extend(INT.pack(len(payload)))


# ── Fixed-width string blocks ────────────────────────────────────────────────

def decode_str_block(blk: bytes, width: int = SET_NAME_WIDTH) -> list[str]:
    """Split a fixed-width padded string block into stripped strings."""
    out: list[str] = []
    p = 0
    while p + width <= len(blk):
        out.append(blk[p:p + width].decode("ascii", errors="replace").rstrip())
        p += width
    return out


def encode_str_block(strings: list[str], width: int = SET_NAME_WIDTH) -> bytes:
    """Pack strings into a fixed-width padded block.

    Each string is right-padded with ASCII spaces to `width`. Raises
    ValueError if any string exceeds `width`.
    """
    out = bytearray()
    for s in strings:
        b = s.encode("ascii")
        if len(b) > width:
            raise ValueError(
                f"string {s!r} exceeds fixed width {width}: got {len(b)} bytes"
            )
        out.extend(b.ljust(width, b" "))
    return bytes(out)


# ── Set-element record codec ─────────────────────────────────────────────────

def read_set_element_record(rec: bytes) -> list[str]:
    """Decode a set-elements record: pad(4) + flag(4) + n_total(4) + n_here(4) + names."""
    if len(rec) < 16:
        return []
    n = INT.unpack_from(rec, 8)[0]
    return decode_str_block(rec[16:16 + SET_NAME_WIDTH * n])


def write_set_element_record(
    elements: list[str],
    n_total: int,
    flag: int,
    width: int = SET_NAME_WIDTH,
) -> bytes:
    """Build one set-elements record.

    Layout: PAD(4) + flag(4) + n_total(4) + n_here(4) + names(n_here*width).
    `flag` is 2 when more records follow, 1 on the last record.
    """
    n_here = len(elements)
    payload = bytearray()
    payload.extend(PAD)
    payload.extend(INT.pack(flag))
    payload.extend(INT.pack(n_total))
    payload.extend(INT.pack(n_here))
    payload.extend(encode_str_block(elements, width=width))
    return bytes(payload)


# ── Set-descriptor codec ─────────────────────────────────────────────────────

def parse_set_descriptor(rec: bytes) -> tuple[str, list[str], int]:
    """Decode the per-coefficient set descriptor record.

    Layout:
      [0:4]    pad
      [4:8]    n_unique_sets (each distinct set counted once)
      [8:12]   flag (= 1 when set names follow)
      [12:16]  ndim (true number of dimensions; may exceed n_unique_sets
               when a set appears more than once, e.g. VMSB on REG×REG)
      [16:28]  coeff_name (12 chars padded)
      [28:32]  flag (= 1 when set names present)
      [32:32 + 12*ndim] set names

    Returns (coeff_name, set_names, ndim).
    """
    ndim = INT.unpack_from(rec, 12)[0]
    coeff_name = rec[16:28].decode("ascii", errors="replace").rstrip()
    set_names = decode_str_block(rec[32:32 + SET_NAME_WIDTH * ndim])
    return coeff_name, set_names, ndim


def build_set_descriptor(coeff_name: str, set_names: list[str]) -> bytes:
    """Build the set-descriptor record symmetric to parse_set_descriptor.

    The on-disk descriptor reports `n_unique_sets` (each distinct set name
    counted once) and `ndim` (the full count, with repeats). For an array
    with set_names=["COMM","REG","REG"], n_unique=2 and ndim=3.

    GEMPACK-canonical trailing layout (required by harpy3 0.3.1 and the
    Fortran HAR reader; see issue #12):

      ... + set_names(12*ndim)
          + set_status(1*n_unique)   # 'k' = known set, name resolved at load
          + dim_sizes(4*n_unique)    # int32; 0 means "use the element record"
          + Nexplicit(4)             # int32; 0 = no explicit subset overlay
    """
    unique: list[str] = []
    for sn in set_names:
        if sn not in unique:
            unique.append(sn)
    ndim = len(set_names)
    n_unique = len(unique)
    if len(coeff_name) > SET_NAME_WIDTH:
        raise ValueError(
            f"coeff_name {coeff_name!r} exceeds {SET_NAME_WIDTH} bytes"
        )
    payload = bytearray()
    payload.extend(PAD)
    payload.extend(INT.pack(n_unique))
    payload.extend(INT.pack(1))  # flag: set names follow
    payload.extend(INT.pack(ndim))
    payload.extend(coeff_name.encode("ascii").ljust(SET_NAME_WIDTH, b" "))
    payload.extend(INT.pack(1))  # second flag: set names present
    payload.extend(encode_str_block(set_names, width=SET_NAME_WIDTH))
    # Trailing fields required by GEMPACK / harpy3: status byte per dimension,
    # then a dim_size int32 per dimension, then Nexplicit. (harpy3 reads NSets
    # from the *ndim* slot, so the per-dim counts use ndim, not n_unique —
    # repeated sets such as REG×REG get one byte/int per occurrence.)
    payload.extend(b"k" * ndim)
    payload.extend(b"\x00\x00\x00\x00" * ndim)
    payload.extend(INT.pack(0))
    return bytes(payload)
