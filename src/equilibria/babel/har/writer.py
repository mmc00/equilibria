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
    *,
    prefer_sparse: list[str] | None = None,
) -> None:
    """Write a HAR file from a mapping of header name → HeaderArray.

    Args:
        path: target .har path.
        headers: mapping of header name → HeaderArray.
        prefer_sparse: names that should be emitted as RESPSE instead of
            REFULL. Default: empty (always emit dense REFULL for floats).

    Emits all 1CFULL (set) headers first, then non-1CFULL headers, each
    group in mapping insertion order. Writes atomically via a temp file
    and os.replace; on any emission failure the target path is untouched.
    """
    if not headers:
        raise ValueError("no headers to write")

    _validate_headers(headers)

    prefer_sparse_set = set(prefer_sparse or [])
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")

    out = bytearray()
    try:
        sets, arrays = _partition_sets_first(headers)
        for name, ha in [*sets, *arrays]:
            _emit_header(out, name, ha, sparse=name in prefer_sparse_set)
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

def _emit_header(out: bytearray, name: str, ha: HeaderArray, *, sparse: bool = False) -> None:
    if _looks_like_1cfull(ha):
        _write_1cfull(out, name, ha)
        return
    if ha.array.dtype == np.int32 and ha.array.ndim == 2 and not ha.set_names:
        _write_2ifull(out, name, ha)
        return
    # Float arrays without set names are scalar/no-set REFULL (e.g. DVER).
    # Skip the set-vs-ndim validator and let _write_refull handle it.
    if ha.set_names:
        _validate_array_header(name, ha)
    if ha.array.dtype in (np.float32, np.float64):
        if sparse:
            _write_respse(out, name, ha)
        else:
            _write_refull(out, name, ha)
        return
    raise NotImplementedError(
        f"writer does not yet support emitting {name!r} "
        f"(dtype={ha.array.dtype}, ndim={ha.array.ndim}, "
        f"set_names={ha.set_names!r})"
    )


def _validate_array_header(name: str, ha: HeaderArray) -> None:
    if ha.array.ndim != len(ha.set_names):
        raise ValueError(
            f"{name!r}: ndim {ha.array.ndim} != len(set_names) {len(ha.set_names)}"
        )
    if len(ha.set_elements) != len(ha.set_names):
        raise ValueError(
            f"{name!r}: len(set_elements) {len(ha.set_elements)} "
            f"!= len(set_names) {len(ha.set_names)}"
        )
    for k, (sn, elems) in enumerate(zip(ha.set_names, ha.set_elements)):
        if ha.array.shape[k] != len(elems):
            raise ValueError(
                f"{name!r}: shape[{k}]={ha.array.shape[k]} for set {sn!r} "
                f"but set_elements[{k}] has {len(elems)} entries"
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
    # Width defaults to SET_NAME_WIDTH (12) but is widened if any element
    # exceeds it (e.g. GEMPACK version strings stored in *_VER headers).
    max_len = max((len(e.encode("ascii")) for e in elements), default=0)
    width = max(wire.SET_NAME_WIDTH, max_len)

    _write_name_record(out, name)
    # Meta tail: filler flag (2 for 1CFULL) + n_total + width.
    # Reader unpacks n_total/width at offsets 84/88 of the meta payload,
    # so the tail must start with a 4-byte filler at offset 80.
    tail = wire.INT.pack(2) + wire.INT.pack(n_total) + wire.INT.pack(width)
    _write_meta_record(out, wire.TOKEN_1CFULL, ha.long_name, tail)

    # For now emit a single record (multi-record chunking added in Task 5
    # if a fixture requires it).
    rec = wire.write_set_element_record(elements, n_total=n_total, flag=1, width=width)
    wire.write_record(out, rec)


def _write_2ifull(out: bytearray, name: str, ha: HeaderArray) -> None:
    """Emit a 2IFULL header (2-D integer dense).

    Layout:
      name record (4 bytes)
      meta record (PAD + "2IFULL" + long_name(70) + filler + rows + cols)
      data record (PAD + 1 + rows*cols * int32, Fortran order)
    """
    if ha.array.dtype != np.int32:
        raise TypeError(
            f"{name!r}: 2IFULL requires int32 dtype; got {ha.array.dtype}. "
            f"Cast explicitly with .astype(np.int32) if truncation is intended."
        )
    if ha.array.ndim != 2:
        raise ValueError(
            f"{name!r}: 2IFULL requires 2-D array; got ndim={ha.array.ndim}"
        )
    rows, cols = ha.array.shape
    _write_name_record(out, name)
    # Filler at offset 80 so rows/cols land at 84/88 where the reader looks.
    tail = wire.INT.pack(0) + wire.INT.pack(rows) + wire.INT.pack(cols)
    _write_meta_record(out, wire.TOKEN_2IFULL, ha.long_name, tail)

    flat = ha.array.flatten(order="F").astype("<i4")
    data = bytearray()
    data.extend(wire.PAD)
    data.extend(wire.INT.pack(1))
    data.extend(flat.tobytes())
    wire.write_record(out, bytes(data))


def _write_respse(out: bytearray, name: str, ha: HeaderArray) -> None:
    """Emit a RESPSE header (real sparse N-D).

    Block layout (records, in order):
      0  name (4 bytes)
      1  meta (PAD + "RESPSE" + long_name(70) + filler + nnz + 0)
      2  set descriptor
      3 .. 3+n_unique-1  one element record per unique set
      3+n_unique         sparse-setup record (PAD + ndim + shape + nnz)
      4+n_unique         data record (PAD + 1 + nnz + nnz + idx[nnz] + val[nnz])
                         with idx 1-based Fortran-order flat indices.
    """
    arr = np.ascontiguousarray(ha.array, dtype=np.float32)
    flat_f = arr.flatten(order="F")
    nz = np.flatnonzero(flat_f)
    nnz = int(nz.size)

    _write_name_record(out, name)
    meta_tail = wire.INT.pack(0) + wire.INT.pack(nnz) + wire.INT.pack(0)
    _write_meta_record(out, wire.TOKEN_RESPSE, ha.long_name, meta_tail)

    wire.write_record(out, wire.build_set_descriptor(ha.coeff_name or name, ha.set_names))

    seen: set[str] = set()
    for sn, elems in zip(ha.set_names, ha.set_elements):
        if sn in seen:
            continue
        seen.add(sn)
        rec = wire.write_set_element_record(elems, n_total=len(elems), flag=1)
        wire.write_record(out, rec)

    setup = bytearray()
    setup.extend(wire.PAD)
    setup.extend(wire.INT.pack(arr.ndim))
    for dim in arr.shape:
        setup.extend(wire.INT.pack(dim))
    setup.extend(wire.INT.pack(nnz))
    wire.write_record(out, bytes(setup))

    data = bytearray()
    data.extend(wire.PAD)
    data.extend(wire.INT.pack(1))
    data.extend(wire.INT.pack(nnz))
    data.extend(wire.INT.pack(nnz))
    for k in nz:
        data.extend(wire.INT.pack(int(k) + 1))
    data.extend(flat_f[nz].astype("<f4").tobytes())
    wire.write_record(out, bytes(data))


def _write_refull(out: bytearray, name: str, ha: HeaderArray) -> None:
    """Emit a REFULL header (real dense N-D array).

    Block layout (records, in order):
      0  name (4 bytes)
      1  meta (PAD + "REFULL" + long_name(70) + 3 trailing ints)
      2  set descriptor
      3 .. 3+n_unique-1  one element record per unique set
      3+n_unique         dim-summary record
      4+n_unique         dim-metadata record
      5+n_unique         data record (PAD + 1 + n*float32, Fortran order)
    """
    arr = np.ascontiguousarray(ha.array, dtype=np.float32)

    _write_name_record(out, name)
    meta_tail = wire.INT.pack(arr.size) + wire.INT.pack(0) + wire.INT.pack(0)
    _write_meta_record(out, wire.TOKEN_REFULL, ha.long_name, meta_tail)

    wire.write_record(out, wire.build_set_descriptor(ha.coeff_name or name, ha.set_names))

    seen: set[str] = set()
    for sn, elems in zip(ha.set_names, ha.set_elements):
        if sn in seen:
            continue
        seen.add(sn)
        rec = wire.write_set_element_record(elems, n_total=len(elems), flag=1)
        wire.write_record(out, rec)

    summary = bytearray()
    summary.extend(wire.PAD)
    summary.extend(wire.INT.pack(arr.ndim))
    for dim in arr.shape:
        summary.extend(wire.INT.pack(dim))
    wire.write_record(out, bytes(summary))

    meta_dim = bytearray()
    meta_dim.extend(wire.PAD)
    meta_dim.extend(wire.INT.pack(arr.ndim))
    for _ in range(arr.ndim):
        meta_dim.extend(wire.INT.pack(0))
    wire.write_record(out, bytes(meta_dim))

    flat = arr.flatten(order="F")
    data = bytearray()
    data.extend(wire.PAD)
    data.extend(wire.INT.pack(1))
    data.extend(flat.astype("<f4").tobytes())
    wire.write_record(out, bytes(data))


# ── HarWriter builder ────────────────────────────────────────────────────────

class HarWriter:
    """High-level builder for HAR files.

    Maintains a set registry: the first call to `add_set` (or the first
    `add_array` that references a set) records its elements; subsequent
    references must match exactly or a ValueError is raised. On `close()`,
    sets are emitted as 1CFULL headers in first-registration order, then
    arrays in add-call order.
    """

    def __init__(self, path: str | os.PathLike) -> None:
        self._path = Path(path)
        self._sets: dict[str, list[str]] = {}
        self._arrays: list[tuple[str, HeaderArray, bool]] = []
        self._closed = False

    def __enter__(self) -> "HarWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None and not self._closed:
            self.close()

    def add_set(self, name: str, elements: list[str]) -> None:
        elements = list(elements)
        if name in self._sets:
            if self._sets[name] != elements:
                raise ValueError(
                    f"set {name!r} already registered with different elements: "
                    f"existing={self._sets[name]!r}, new={elements!r}"
                )
            return
        self._sets[name] = elements

    def add_array(
        self,
        name: str,
        array: np.ndarray,
        set_names: list[str],
        set_elements: list[list[str]] | None = None,
        long_name: str = "",
        *,
        sparse: bool = False,
    ) -> None:
        array = np.asarray(array)
        if set_elements is None:
            set_elements = []
            for sn in set_names:
                if sn not in self._sets:
                    raise ValueError(
                        f"set {sn!r} not registered; "
                        f"call add_set({sn!r}, [...]) first or "
                        f"pass set_elements= explicitly"
                    )
                set_elements.append(list(self._sets[sn]))
        else:
            for sn, elems in zip(set_names, set_elements):
                elems = list(elems)
                if sn in self._sets:
                    if self._sets[sn] != elems:
                        raise ValueError(
                            f"set {sn!r} elements conflict between array "
                            f"{name!r} and previously registered set: "
                            f"existing={self._sets[sn]!r}, new={elems!r}"
                        )
                else:
                    self._sets[sn] = elems

        ha = HeaderArray(
            name=name,
            coeff_name=name,
            long_name=long_name,
            array=array,
            set_names=list(set_names),
            set_elements=[list(e) for e in set_elements],
        )
        self._arrays.append((name, ha, sparse))

    def add_dataframe(
        self,
        name: str,
        df,
        set_names: list[str],
        long_name: str = "",
    ) -> None:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("add_dataframe requires pandas") from exc

        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                f"add_dataframe expects a 2-D pandas.DataFrame; "
                f"got {type(df).__name__}"
            )
        if len(set_names) != 2:
            raise ValueError(
                f"add_dataframe expects 2 set_names; got {len(set_names)}. "
                f"For higher ranks, call add_array with a numpy array."
            )
        index_elems = [str(x) for x in df.index.tolist()]
        col_elems = [str(x) for x in df.columns.tolist()]
        arr = df.values.astype(np.float32, copy=False)
        self.add_array(
            name,
            arr,
            set_names=set_names,
            set_elements=[index_elems, col_elems],
            long_name=long_name,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        headers: dict[str, HeaderArray] = {}
        for sn, elems in self._sets.items():
            headers[sn] = HeaderArray(
                name=sn,
                coeff_name=sn,
                long_name=sn,
                array=np.array(elems, dtype=object),
                set_names=[],
                set_elements=[],
            )
        sparse_names: list[str] = []
        for name, ha, sparse in self._arrays:
            headers[name] = ha
            if sparse:
                sparse_names.append(name)

        write_har(self._path, headers, prefer_sparse=sparse_names)
