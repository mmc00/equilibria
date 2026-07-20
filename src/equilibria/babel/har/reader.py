"""Native pure-Python reader for GEMPACK HAR files.

A HAR file is a sequence of Fortran unformatted records (each record framed
by a 4-byte little-endian length prefix and the same length suffix). Headers
are grouped: a 4-byte name record, a 92-byte metadata record (`"    "` +
6-char type token + 70-char long name + 3 trailing ints), then a number of
data records that depend on the type token.

Supported types: 1CFULL (1-D character set), REFULL (real dense), RESPSE
(real sparse), 2IFULL (2-D integer dense). These cover everything used by
the GTAP datasets shipped with equilibria.

CLEAN-ROOM REIMPLEMENTATION
---------------------------
This file is a clean-room reimplementation of the HAR format, developed by
inspecting the on-disk byte layout of HAR files produced by GEMPACK / RunGTAP
and by reading publicly available format descriptions. It was written WITHOUT
viewing, copying, translating, or deriving from any third-party HAR
implementation, including in particular `harpy3` / `harpy` (GPLv3) or any
GEMPACK source distribution. The format-level constants used here (record
framing, type tokens, set-descriptor layout, etc.) are the wire format itself
— not copyrightable expression. Distributed under MIT (see top-level NOTICE).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from equilibria.babel.har.symbols import HeaderArray
from equilibria.babel.har.wire import (
    INT as _INT,
)
from equilibria.babel.har.wire import (
    PAD as _PAD,
)
from equilibria.babel.har.wire import (
    decode_str_block as _decode_str_block,
)
from equilibria.babel.har.wire import (
    iter_records as _iter_records,
)
from equilibria.babel.har.wire import (
    parse_set_descriptor as _parse_set_descriptor,
)
from equilibria.babel.har.wire import (
    read_set_element_record as _read_set_element_record,
)


def _read_header(records: list[bytes], i: int) -> tuple[HeaderArray | None, int]:
    """Parse one header block starting at `records[i]`. Returns (header, next_i)."""
    name_rec = records[i]
    if len(name_rec) != 4:
        return None, i + 1
    if i + 1 >= len(records):
        return None, i + 1

    name = name_rec.decode("ascii", errors="replace").rstrip()
    meta = records[i + 1]
    if len(meta) < 80 or meta[:4] != _PAD:
        return None, i + 1

    type_token = meta[4:10].decode("ascii", errors="replace")
    long_name = meta[10:80].decode("ascii", errors="replace").rstrip()

    if type_token == "1CFULL":
        return _read_1cfull(name, long_name, records, i + 1)
    if type_token == "REFULL":
        return _read_refull(name, long_name, records, i + 1)
    if type_token == "RESPSE":
        return _read_respse(name, long_name, records, i + 1)
    if type_token == "2IFULL":
        return _read_2ifull(name, long_name, records, i + 1)

    raise NotImplementedError(
        f"HAR header {name!r} uses unsupported type {type_token!r}"
    )


def _read_1cfull(
    name: str, long_name: str, records: list[bytes], i: int
) -> tuple[HeaderArray, int]:
    """1CFULL: a 1-D character set.

    The meta record (length 92) ends with two trailing ints (n_elements, width).
    Element data follows in one or more records, each shaped:
      pad(4) + flag + n_total + n_in_this_record + names...
    where flag=2 means more records follow, flag=1 is the last.
    """
    meta = records[i]
    n_total, width = struct.unpack_from("<2i", meta, 84)
    elems: list[str] = []
    j = i + 1
    while len(elems) < n_total:
        rec = records[j]
        n_here = _INT.unpack_from(rec, 12)[0]
        elems.extend(_decode_str_block(rec[16 : 16 + width * n_here], width=width))
        j += 1
    arr = np.array(elems[:n_total], dtype=object)
    return (
        HeaderArray(
            name=name,
            coeff_name=name,
            long_name=long_name,
            array=arr,
            set_names=[],
            set_elements=[],
        ),
        j,
    )


def _read_refull(
    name: str, long_name: str, records: list[bytes], i: int
) -> tuple[HeaderArray, int]:
    """REFULL: a real dense N-dimensional array.

    The descriptor reports ndim (one slot per dimension, with repeats when
    the same set is reused, e.g. REG×REG) and ndim set names. The file
    only stores element records for *unique* sets, so VMSB on COMM×REG×REG
    has 3 set names but only 2 element records.

    Block (after meta at index i):
      i+1: set descriptor
      i+2 .. i+1+n_unique: one record per *unique* set
      i+2+n_unique: dim-summary record
      i+3+n_unique: dim-metadata record
      i+4+n_unique: data record (pad + int + n*float32)
    """
    desc = records[i + 1]
    coeff_name, set_names, ndim = _parse_set_descriptor(desc)
    unique_names: list[str] = []
    for sn in set_names:
        if sn not in unique_names:
            unique_names.append(sn)
    n_unique = len(unique_names)
    elems_by_name = {
        sn: _read_set_element_record(records[i + 2 + k])
        for k, sn in enumerate(unique_names)
    }
    set_elements = [elems_by_name[sn] for sn in set_names]
    shape = tuple(len(s) for s in set_elements)
    n = int(np.prod(shape)) if shape else 1
    # GEMPACK splits large arrays across multiple records when the data exceeds
    # one record's capacity (~32 KB). Concatenate until we have n float32 values.
    data_start = i + 4 + n_unique
    raw = b""
    extra = 0
    while len(raw) < 8 + n * 4:
        rec = records[data_start + extra]
        raw += rec[8:] if extra > 0 else rec
        extra += 1
    floats = struct.unpack_from(f"<{n}f", raw, 8)
    arr = (
        np.array(floats, dtype=np.float32).reshape(shape, order="F")
        if shape
        else np.array(floats, dtype=np.float32)
    )
    return (
        HeaderArray(
            name=name,
            coeff_name=coeff_name,
            long_name=long_name,
            array=arr,
            set_names=set_names,
            set_elements=set_elements,
        ),
        data_start + extra,
    )


def _read_respse(
    name: str, long_name: str, records: list[bytes], i: int
) -> tuple[HeaderArray, int]:
    """RESPSE: a real sparse N-dimensional array.

    Layout (after meta at index i):
      i+1: set descriptor
      i+2 .. i+1+ndim: set element records
      i+2+ndim: sparse setup (we only need it to skip ahead)
      i+3+ndim: data record (pad + 1 + nnz + nnz + idx[nnz] + val[nnz])
                with idx 1-based Fortran-order flat indices.

    Shape is taken from the set element records (no separate dim record).
    """
    desc = records[i + 1]
    coeff_name, set_names, ndim = _parse_set_descriptor(desc)
    unique_names: list[str] = []
    for sn in set_names:
        if sn not in unique_names:
            unique_names.append(sn)
    n_unique = len(unique_names)
    elems_by_name = {
        sn: _read_set_element_record(records[i + 2 + k])
        for k, sn in enumerate(unique_names)
    }
    set_elements = [elems_by_name[sn] for sn in set_names]
    shape = tuple(len(s) for s in set_elements)
    data_start = i + 3 + n_unique
    # Each data record: pad(4) + counter_down(4) + nnz_total(4) + nnz_chunk(4)
    #                   + idx[nnz_chunk] (4-byte ints) + val[nnz_chunk] (4-byte floats)
    # counter_down goes n_recs..1; nnz_total is repeated in every record.
    # Collect all (index, value) pairs across all chunk records.
    all_idx: list[int] = []
    all_val: list[float] = []
    rec_off = data_start
    while True:
        rec = records[rec_off]
        nnz_total = _INT.unpack_from(rec, 8)[0]
        nnz_chunk = _INT.unpack_from(rec, 12)[0]
        chunk_idx = struct.unpack_from(f"<{nnz_chunk}i", rec, 16)
        chunk_val = struct.unpack_from(f"<{nnz_chunk}f", rec, 16 + 4 * nnz_chunk)
        all_idx.extend(chunk_idx)
        all_val.extend(chunk_val)
        counter = _INT.unpack_from(rec, 4)[0]
        rec_off += 1
        if counter == 1:  # last chunk
            break
    flat = np.zeros(int(np.prod(shape)) if shape else 1, dtype=np.float32)
    for k, v in zip(all_idx, all_val):
        flat[k - 1] = v
    arr = flat.reshape(shape, order="F") if shape else flat
    return (
        HeaderArray(
            name=name,
            coeff_name=coeff_name,
            long_name=long_name,
            array=arr,
            set_names=set_names,
            set_elements=set_elements,
        ),
        rec_off,
    )


def _read_2ifull(
    name: str, long_name: str, records: list[bytes], i: int
) -> tuple[HeaderArray, int]:
    """2IFULL: a 2-D integer dense array. The meta record's trailing ints hold
    (rows, cols, width); the next record is the data record with pad(4) +
    1 int + rows*cols 4-byte ints."""
    meta = records[i]
    rows = _INT.unpack_from(meta, 84)[0]
    cols = _INT.unpack_from(meta, 88)[0]
    data_rec = records[i + 1]
    n = rows * cols
    ints = struct.unpack_from(f"<{n}i", data_rec, 8)
    arr = np.array(ints, dtype=np.int32).reshape((rows, cols), order="F")
    return (
        HeaderArray(
            name=name,
            coeff_name=name,
            long_name=long_name,
            array=arr,
            set_names=[],
            set_elements=[],
        ),
        i + 2,
    )


def read_har(
    filepath: str | Path,
    select_headers: list[str] | None = None,
) -> dict[str, HeaderArray]:
    """Read a GEMPACK HAR file and return its header arrays.

    Args:
        filepath: Path to the .har or .prm file.
        select_headers: If provided, only these headers are returned.

    Returns:
        Dict mapping header name → HeaderArray.

    Raises:
        FileNotFoundError: If filepath does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"HAR file not found: {filepath}")

    records = list(_iter_records(filepath.read_bytes()))
    selected = set(select_headers) if select_headers is not None else None

    result: dict[str, HeaderArray] = {}
    i = 0
    while i < len(records):
        if len(records[i]) != 4:
            i += 1
            continue
        try:
            ha, next_i = _read_header(records, i)
        except (struct.error, IndexError, NotImplementedError) as exc:
            raise ValueError(
                f"Failed to parse HAR header at record {i} of {filepath}"
            ) from exc
        if ha is not None and (selected is None or ha.name in selected):
            result[ha.name] = ha
        i = next_i if ha is not None else i + 1
    return result


def get_header_names(filepath: str | Path) -> list[str]:
    """Return all header names in a HAR file."""
    return list(read_har(filepath).keys())


def read_header_array(filepath: str | Path, name: str) -> HeaderArray:
    """Read a single named header array from a HAR file."""
    out = read_har(filepath, select_headers=[name])
    if name not in out:
        raise KeyError(f"Header {name!r} not found in {filepath}")
    return out[name]
