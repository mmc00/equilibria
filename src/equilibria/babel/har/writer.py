"""Native pure-Python writer for GEMPACK HAR files.

Symmetric to reader.py: emits the Fortran-framed records that the reader
parses. Uses wire.py as the single source of truth for the on-disk byte
layout so encode and decode stay in lockstep.

CLEAN-ROOM REIMPLEMENTATION — see README.md and the NOTICE file for the
full provenance statement. No harpy3, harpy, or GEMPACK source has been
consulted in writing this code. harpy3 is used only as a sandboxed
black-box oracle via scripts/har/oracle_check.py. Distributed under MIT.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Mapping

import numpy as np

from equilibria.babel.har.symbols import HeaderArray
from equilibria.babel.har import wire


# ── Top-level entry ──────────────────────────────────────────────────────────

def write_har(
    path: str | os.PathLike,
    headers: Mapping[str, HeaderArray],
) -> None:
    """Write a HAR file from a mapping of header name → HeaderArray.

    Emits all 1CFULL (set) headers first, then non-1CFULL headers, each
    group in mapping insertion order. Writes atomically via a temp file
    and os.replace; on any emission failure the target path is untouched.
    """
    if not headers:
        raise ValueError("no headers to write")

    _validate_headers(headers)

    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")

    out = bytearray()
    try:
        sets, arrays = _partition_sets_first(headers)
        for name, ha in [*sets, *arrays]:
            _emit_header(out, name, ha)
        tmp.write_bytes(bytes(out))
        os.replace(tmp, target)
    except BaseException:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


# ── Validation ───────────────────────────────────────────────────────────────

def _validate_headers(headers: Mapping[str, HeaderArray]) -> None:
    for name, ha in headers.items():
        if not (1 <= len(name) <= wire.NAME_WIDTH):
            raise ValueError(
                f"header name {name!r} must be 1..{wire.NAME_WIDTH} ASCII chars; got {len(name)}"
            )
        try:
            name.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(
                f"header name {name!r} must be ASCII"
            ) from exc
        if any(ch.isalpha() and not ch.isupper() for ch in name):
            warnings.warn(
                f"header name {name!r} is not uppercase ASCII; "
                f"written verbatim",
                UserWarning,
                stacklevel=3,
            )
        if len(ha.long_name) > wire.LONG_NAME_WIDTH:
            raise ValueError(
                f"long_name for {name!r} exceeds {wire.LONG_NAME_WIDTH} bytes"
            )


def _partition_sets_first(
    headers: Mapping[str, HeaderArray],
) -> tuple[list[tuple[str, HeaderArray]], list[tuple[str, HeaderArray]]]:
    """Split into (1CFULL sets, all other headers), preserving order within each group."""
    sets: list[tuple[str, HeaderArray]] = []
    arrays: list[tuple[str, HeaderArray]] = []
    for name, ha in headers.items():
        if _looks_like_1cfull(ha):
            sets.append((name, ha))
        else:
            arrays.append((name, ha))
    return sets, arrays


def _looks_like_1cfull(ha: HeaderArray) -> bool:
    return (
        ha.array.ndim == 1
        and ha.array.dtype == object
        and not ha.set_names
        and not ha.set_elements
    )


# ── Dispatch ─────────────────────────────────────────────────────────────────

def _emit_header(out: bytearray, name: str, ha: HeaderArray) -> None:
    if _looks_like_1cfull(ha):
        _write_1cfull(out, name, ha)
        return
    raise NotImplementedError(
        f"writer does not yet support emitting {name!r} "
        f"(dtype={ha.array.dtype}, ndim={ha.array.ndim}, "
        f"set_names={ha.set_names!r})"
    )


# ── Per-token emitters ───────────────────────────────────────────────────────

def _write_name_record(out: bytearray, name: str) -> None:
    """4-byte name record, right-padded with spaces to NAME_WIDTH."""
    payload = name.encode("ascii").ljust(wire.NAME_WIDTH, b" ")
    wire.write_record(out, payload)


def _write_meta_record(out: bytearray, token: str, long_name: str, tail: bytes) -> None:
    """Meta record: PAD(4) + token(6) + long_name(70) + tail-ints."""
    if len(token) != 6:
        raise ValueError(f"type token {token!r} must be 6 bytes")
    payload = bytearray()
    payload.extend(wire.PAD)
    payload.extend(token.encode("ascii"))
    payload.extend(long_name.encode("ascii").ljust(wire.LONG_NAME_WIDTH, b" "))
    payload.extend(tail)
    wire.write_record(out, bytes(payload))


def _write_1cfull(out: bytearray, name: str, ha: HeaderArray) -> None:
    """Emit a 1CFULL header (1-D character set).

    Layout:
      name record (4 bytes)
      meta record (PAD + "1CFULL" + long_name(70) + n_total(4) + width(4))
      one-or-more element records (PAD + flag + n_total + n_here + names)
    """
    elements = [str(e).rstrip() for e in ha.array.tolist()]
    n_total = len(elements)
    width = wire.SET_NAME_WIDTH

    _write_name_record(out, name)
    # Meta tail: filler flag (2 for 1CFULL) + n_total + width.
    # Reader unpacks n_total/width at offsets 84/88 of the meta payload,
    # so the tail must start with a 4-byte filler at offset 80.
    tail = wire.INT.pack(2) + wire.INT.pack(n_total) + wire.INT.pack(width)
    _write_meta_record(out, wire.TOKEN_1CFULL, ha.long_name, tail)

    # For now emit a single record (multi-record chunking added in Task 5
    # if a fixture requires it).
    rec = wire.write_set_element_record(elements, n_total=n_total, flag=1)
    wire.write_record(out, rec)
