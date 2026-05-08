"""Native parser for GEMPACK HAR / PRM files.

HAR is a Fortran "unformatted sequential" file. Every record is:

    <int32 nbyte><payload (nbyte bytes)><int32 nbyte>

Records are grouped into headers. Each header has:

  1. A 4-byte name record (with possible leading empty records).
  2. A "second record" (metadata): 4 spaces, 2-byte data type, 4-byte storage
     type, 70-byte long_name, int32 ndim, then ndim int32 sizes.
  3. Data records depending on the type:
       - 1C: character vector (one string column).
       - 2R/2I: 2-D float32 / int32, FULL only.
       - RE: real-valued n-D (1..7) with set-element metadata, FULL or SPSE.
       - RL: like RE but with no set metadata.

Only the read path is implemented — equilibria does not write HAR files.
The reference implementation lives in `harpy3`; this module replaces it
so we don't pull a 5MB external dep just for a reader.
"""

from __future__ import annotations

import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from equilibria.babel.har.symbols import HeaderArray

_V1_DATA_TYPES = {"1C", "2R", "2I", "RE", "RL", "DE", "DL"}
_STORAGE_TYPES = {"FULL", "SPSE"}
_MAX_DIMS = 7


def _decode(b: bytes) -> str:
    return b.decode("latin-1") if b else ""


def _read_record_length(fp: BinaryIO) -> int | None:
    raw = fp.read(4)
    if not raw:
        return None
    return struct.unpack("=i", raw)[0]


def _read_record(fp: BinaryIO) -> bytes | None:
    """Read one Fortran-unformatted record (header, payload, trailer)."""
    nbyte = _read_record_length(fp)
    if nbyte is None:
        return None
    payload = fp.read(nbyte)
    trailer = _read_record_length(fp)
    if trailer != nbyte:
        raise IOError(
            f"HAR record corrupted: leading length {nbyte} != trailing {trailer} "
            f"at offset {fp.tell()}"
        )
    return payload


def _unpack(fmt: str, payload: bytes, offset: int = 0) -> tuple:
    return struct.unpack_from(fmt, payload, offset)


@dataclass
class _SetSpec:
    name: str | None
    status: str  # 'k' known, 'u' unknown, 'e' explicit (per-element), 'n' none
    elements: list[str] | None
    size: int


def _read_header_name(fp: BinaryIO) -> tuple[int, str | None]:
    """Skip empty records and return (record_start, header_name).

    Returns (-1, None) when EOF is reached.
    """
    while True:
        pos = fp.tell()
        nbyte = _read_record_length(fp)
        if nbyte is None:
            return -1, None
        # Read 4-byte name; if blank, skip the rest of the record and try again.
        name_bytes = fp.read(4).strip()
        if name_bytes:
            # Consume the rest of the record (most names are padded to 4 bytes
            # but we still have to advance past any extra payload + trailer).
            remaining = nbyte - 4
            if remaining > 0:
                fp.read(remaining)
            trailer = _read_record_length(fp)
            if trailer != nbyte:
                raise IOError("HAR header-name record corrupted")
            return pos, _decode(name_bytes)
        # Empty name → skip remainder of the record and the trailer, then loop.
        remaining = nbyte - 4
        if remaining > 0:
            fp.read(remaining)
        trailer = _read_record_length(fp)
        if trailer != nbyte:
            raise IOError("HAR padding record corrupted")


def _read_second_record(
    fp: BinaryIO, name: str
) -> tuple[str, str, str, list[int]]:
    """Return (data_type, storage_type, long_name, sizes)."""
    payload = _read_record(fp)
    if payload is None:
        raise IOError(f"Unexpected EOF reading metadata for header '{name}'")

    head_fmt = "=4s2s4s70si"
    head_size = struct.calcsize(head_fmt)
    if len(payload) < head_size:
        raise IOError(f"Header '{name}' second record too short ({len(payload)} bytes)")

    pad, dtype, storage, long_name, ndim = _unpack(head_fmt, payload)
    if pad != b"    ":
        raise IOError(f"Header '{name}' second record missing 4-space marker")

    dtype_s = _decode(dtype)
    storage_s = _decode(storage)
    long_s = _decode(long_name)

    if dtype_s not in _V1_DATA_TYPES:
        raise IOError(f"Header '{name}': unsupported HAR data type '{dtype_s}'")
    if storage_s not in _STORAGE_TYPES:
        raise IOError(f"Header '{name}': unknown storage type '{storage_s}'")
    if ndim > _MAX_DIMS:
        raise IOError(f"Header '{name}': ndim {ndim} exceeds max {_MAX_DIMS}")

    expected = head_size + 4 * ndim
    if expected != len(payload):
        raise IOError(
            f"Header '{name}': second record corrupted (expected {expected} bytes, got {len(payload)})"
        )

    sizes = list(_unpack("=" + "i" * ndim, payload, head_size))
    return dtype_s, storage_s, long_s, sizes


def _read_char_vec(fp: BinaryIO, n_entries: int, str_len: int) -> list[str]:
    """Read a 1C-style packed character vector."""
    out: list[str] = []
    nrec = 100
    while nrec > 1:
        payload = _read_record(fp)
        if payload is None:
            raise IOError("Unexpected EOF in character-vector record")
        # Header: 4 bytes pad, 3 ints
        pad, this_nrec, max_entries, n_on_rec = _unpack("=4siii", payload)
        if pad != b"    ":
            raise IOError("Char-vec record missing 4-space marker")
        if max_entries != n_entries:
            raise IOError(
                f"Char-vec size mismatch: header says {max_entries}, expected {n_entries}"
            )
        nrec = this_nrec
        body = payload[struct.calcsize("=4siii") :]
        # Each entry is `str_len` raw bytes, packed contiguously.
        for k in range(n_on_rec):
            chunk = body[k * str_len : (k + 1) * str_len]
            out.append(_decode(chunk))
        if len(out) > n_entries:
            raise IOError("Char-vec read more entries than declared")
    if len(out) != n_entries:
        raise IOError(
            f"Char-vec under-read: got {len(out)} of {n_entries} entries"
        )
    return out


def _read_set_info(
    fp: BinaryIO, sizes: list[int]
) -> tuple[str, list[_SetSpec]]:
    """Parse the set-element-info record that precedes RE arrays."""
    payload = _read_record(fp)
    if payload is None:
        raise IOError("Unexpected EOF reading set-info record")

    head_fmt = "=4siii12si"
    head_size = struct.calcsize(head_fmt)
    pad, _n_to_write, _flag, n_sets, coeff, sets_known = _unpack(head_fmt, payload)
    if pad != b"    ":
        raise IOError("Set-info record missing 4-space marker")
    coeff_name = _decode(coeff).strip()

    cursor = head_size
    set_names: list[str] = []
    statuses: list[str] = []
    if sets_known:
        names_block = payload[cursor : cursor + n_sets * 12]
        cursor += n_sets * 12
        status_block = payload[cursor : cursor + n_sets]
        cursor += n_sets
        # Discard the n_sets int32 reserved field (always zeros on write).
        cursor += 4 * n_sets
        for i in range(n_sets):
            set_names.append(_decode(names_block[i * 12 : (i + 1) * 12]).strip())
            statuses.append(_decode(status_block[i : i + 1]))
    else:
        set_names = [""] * n_sets
        statuses = ["u"] * n_sets

    n_explicit = _unpack("=i", payload, cursor)[0]
    cursor += 4
    explicit_elements: list[str] = []
    if n_explicit > 0:
        # In the harpy reference reader, the int n_explicit appears twice
        # (once for the real count, then re-read inside the same record).
        # We've already consumed the first; the elements follow immediately.
        block = payload[cursor : cursor + n_explicit * 12]
        for i in range(n_explicit):
            explicit_elements.append(_decode(block[i * 12 : (i + 1) * 12]).strip())

    # Resolve per-dimension set elements.
    specs: list[_SetSpec] = []
    seen_known: dict[str, list[str]] = {}
    explicit_idx = 0
    for idim, (name, status) in enumerate(zip(set_names, statuses, strict=False)):
        size = sizes[idim] if idim < len(sizes) else 1
        if status == "k":
            if name not in seen_known:
                # Each unique 'known' set has its own subsequent 1C-style
                # record listing the elements.
                seen_known[name] = _read_char_vec(fp, n_entries=size, str_len=12)
            elements = [s.strip() for s in seen_known[name]]
            specs.append(_SetSpec(name=name, status=status, elements=elements, size=size))
        elif status == "e":
            specs.append(
                _SetSpec(
                    name=name,
                    status=status,
                    elements=[explicit_elements[explicit_idx]] if explicit_elements else None,
                    size=size,
                )
            )
            explicit_idx += 1
        else:
            specs.append(_SetSpec(name=name, status=status, elements=None, size=size))

    return coeff_name, specs


def _read_re_full(fp: BinaryIO, sizes: list[int], ndim: int) -> np.ndarray:
    """Read a FULL-storage RE/RL array, returning a contiguous ndarray."""
    # First record of the data block: header with the (full-7D) shape.
    payload = _read_record(fp)
    if payload is None:
        raise IOError("Unexpected EOF reading RE-FULL header")
    pad, nrec, ndim_check = _unpack("=4sii", payload)
    if pad != b"    ":
        raise IOError("RE-FULL header missing 4-space marker")
    # Skip the 7 dimension sizes that follow — we already have `sizes`.

    target_shape = sizes[:ndim] if ndim > 0 else (1,)
    flat = np.zeros(int(np.prod(target_shape) if ndim > 0 else 1), dtype=np.float32)

    cursor = 0
    while nrec > 1:
        # Slice header: 4 spaces + 15 ints (7 (start,end) pairs + nrec).
        slice_payload = _read_record(fp)
        if slice_payload is None:
            raise IOError("Unexpected EOF in RE-FULL slice header")
        # The slice header tells us which slab of the 7D array follows; we
        # only need to advance `nrec`. The actual write target is sequential
        # because harpy slices in flatten('F') order.

        # Data record.
        data_payload = _read_record(fp)
        if data_payload is None:
            raise IOError("Unexpected EOF in RE-FULL data record")
        pad, this_nrec = _unpack("=4si", data_payload)
        if pad != b"    ":
            raise IOError("RE-FULL data record missing 4-space marker")
        nrec = this_nrec
        body = data_payload[struct.calcsize("=4si") :]
        ndata = len(body) // 4
        chunk = np.frombuffer(body[: ndata * 4], dtype="<f4")
        flat[cursor : cursor + ndata] = chunk
        cursor += ndata

    if cursor != flat.size:
        raise IOError(
            f"RE-FULL under-read: filled {cursor} of {flat.size} cells"
        )

    if ndim == 0:
        return flat.reshape(())
    arr = flat.reshape(target_shape, order="F")
    return np.ascontiguousarray(arr)


def _read_re_sparse(fp: BinaryIO, sizes: list[int], ndim: int) -> np.ndarray:
    """Read a SPSE-storage RE array."""
    payload = _read_record(fp)
    if payload is None:
        raise IOError("Unexpected EOF reading RE-SPSE header")
    head_fmt = "=4siii80s"
    pad, _nnz, isize, fsize, _comment = _unpack(head_fmt, payload)
    if pad != b"    ":
        raise IOError("RE-SPSE header missing 4-space marker")
    if isize != 4 or fsize != 4:
        raise IOError(f"RE-SPSE: unsupported int/float widths ({isize}/{fsize})")

    target_shape = sizes[:ndim] if ndim > 0 else (1,)
    flat = np.zeros(int(np.prod(target_shape) if ndim > 0 else 1), dtype=np.float32)

    nrec = 50
    while nrec > 1:
        chunk = _read_record(fp)
        if chunk is None:
            raise IOError("Unexpected EOF in RE-SPSE chunk")
        pad, this_nrec, _nnz_total, n_here = _unpack("=4siii", chunk)
        if pad != b"    ":
            raise IOError("RE-SPSE chunk missing 4-space marker")
        nrec = this_nrec
        body = chunk[struct.calcsize("=4siii") :]
        idx = np.frombuffer(body[: n_here * 4], dtype="<i4")
        vals = np.frombuffer(body[n_here * 4 : n_here * 8], dtype="<f4")
        # idx is 1-based into the Fortran-flattened array.
        flat[idx - 1] = vals

    if ndim == 0:
        return flat.reshape(())
    arr = flat.reshape(target_shape, order="F")
    return np.ascontiguousarray(arr)


def _read_2d(
    fp: BinaryIO, sizes: list[int], dtype_code: str
) -> np.ndarray:
    """Read 2R / 2I arrays (always FULL)."""
    rows, cols = sizes[0], sizes[1]
    np_dtype = np.float32 if dtype_code == "f" else np.int32
    arr = np.zeros((rows, cols), dtype=np_dtype, order="F")
    total = rows * cols
    nread = 0
    while nread < total:
        payload = _read_record(fp)
        if payload is None:
            raise IOError("Unexpected EOF reading 2D array")
        head_fmt = "=4siiiiiii"
        head_size = struct.calcsize(head_fmt)
        pad, _nrec, r_total, c_total, x0, x1, y0, y1 = _unpack(head_fmt, payload)
        if pad != b"    ":
            raise IOError("2D record missing 4-space marker")
        if r_total != rows or c_total != cols:
            raise IOError(
                f"2D record shape mismatch ({r_total},{c_total}) vs ({rows},{cols})"
            )
        xsize = x1 - x0 + 1
        ysize = y1 - y0 + 1
        ndata = xsize * ysize
        body = payload[head_size:]
        if dtype_code == "f":
            block = np.frombuffer(body[: ndata * 4], dtype="<f4")
        else:
            block = np.frombuffer(body[: ndata * 4], dtype="<i4")
        arr[x0 - 1 : x1, y0 - 1 : y1] = block.reshape(xsize, ysize, order="F")
        nread += ndata
    return np.ascontiguousarray(arr)


def _read_1c(fp: BinaryIO, sizes: list[int]) -> np.ndarray:
    n_entries = sizes[0]
    str_len = sizes[1]
    items = _read_char_vec(fp, n_entries=n_entries, str_len=str_len)
    return np.array([s.strip() for s in items], dtype=f"<U{str_len}")


def _read_one_header(fp: BinaryIO, name: str) -> HeaderArray:
    dtype, storage, long_name, sizes = _read_second_record(fp, name)

    if dtype == "1C":
        arr = _read_1c(fp, sizes)
        set_names: list[str] = [""]
        # If long_name starts with "Set ", treat as a known set whose elements
        # are the 1C values themselves (matches harpy convention).
        if long_name.startswith("Set "):
            set_token = long_name.split()[1] if len(long_name.split()) > 1 else ""
            set_names = [set_token]
            set_elements = [arr.tolist()]
        else:
            set_elements: list[list[str]] = [[]]
        return HeaderArray(
            name=name,
            coeff_name=name,
            long_name=long_name,
            array=arr,
            set_names=set_names,
            set_elements=set_elements,
        )

    if dtype in ("2R", "2I"):
        code = "f" if dtype == "2R" else "i"
        arr = _read_2d(fp, sizes, code)
        return HeaderArray(
            name=name,
            coeff_name=name,
            long_name=long_name,
            array=arr,
            set_names=["", ""],
            set_elements=[[], []],
        )

    if dtype in ("RE", "RL"):
        if dtype == "RE":
            coeff, specs = _read_set_info(fp, sizes)
            ndim = len(specs)
        else:
            coeff = name
            specs = [_SetSpec(name="", status="n", elements=None, size=s) for s in sizes]
            ndim = 7
        if storage == "FULL":
            arr = _read_re_full(fp, sizes, ndim)
        else:
            arr = _read_re_sparse(fp, sizes, ndim)
        return HeaderArray(
            name=name,
            coeff_name=coeff,
            long_name=long_name,
            array=arr,
            set_names=[s.name or "" for s in specs],
            set_elements=[(s.elements or []) for s in specs],
        )

    raise IOError(f"Header '{name}': unsupported data type '{dtype}'")


def _iter_headers(fp: BinaryIO):
    while True:
        pos, name = _read_header_name(fp)
        if name is None:
            return
        yield pos, name


def read_har(
    filepath: str | Path,
    select_headers: list[str] | None = None,
) -> dict[str, HeaderArray]:
    """Read a GEMPACK HAR file and return its header arrays.

    Args:
        filepath: Path to the .har or .prm file.
        select_headers: If provided, only these headers are loaded.

    Returns:
        Dict mapping header name → :class:`HeaderArray`.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"HAR file not found: {filepath}")

    wanted: set[str] | None = (
        {h.strip().upper() for h in select_headers}
        if select_headers is not None
        else None
    )

    out: dict[str, HeaderArray] = OrderedDict()
    with open(filepath, "rb") as fp:
        for _pos, raw_name in _iter_headers(fp):
            name = raw_name.strip()
            key = name.upper()
            if wanted is not None and key not in wanted:
                _skip_header_body(fp, name)
                continue
            arr = _read_one_header(fp, name)
            # Match GEMPACK/harpy convention: header keys are upper-cased.
            arr.name = key
            out[key] = arr
    return out


def _skip_header_body(fp: BinaryIO, name: str) -> None:
    """Read+discard a header body so the file pointer lands on the next header."""
    dtype, storage, _long_name, sizes = _read_second_record(fp, name)
    if dtype == "1C":
        _read_1c(fp, sizes)
        return
    if dtype in ("2R", "2I"):
        _read_2d(fp, sizes, "f" if dtype == "2R" else "i")
        return
    if dtype in ("RE", "RL"):
        if dtype == "RE":
            _coeff, specs = _read_set_info(fp, sizes)
            ndim = len(specs)
        else:
            ndim = 7
        if storage == "FULL":
            _read_re_full(fp, sizes, ndim)
        else:
            _read_re_sparse(fp, sizes, ndim)
        return
    raise IOError(f"Cannot skip header '{name}' of unknown type '{dtype}'")


def get_header_names(filepath: str | Path) -> list[str]:
    """Return all header names in a HAR file without loading array data."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"HAR file not found: {filepath}")
    names: list[str] = []
    with open(filepath, "rb") as fp:
        for _pos, raw_name in _iter_headers(fp):
            name = raw_name.strip()
            names.append(name.upper())
            _skip_header_body(fp, name)
    return names


def read_header_array(filepath: str | Path, name: str) -> HeaderArray:
    """Read a single named header array from a HAR file."""
    data = read_har(filepath, select_headers=[name])
    if name not in data:
        raise KeyError(f"Header '{name}' not found in {filepath}")
    return data[name]
