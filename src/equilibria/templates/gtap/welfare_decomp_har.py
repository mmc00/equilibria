"""WELVIEW.har writer for GTAP welfare decomposition.

Writes a minimal RunGTAP-compatible HAR with the standard welfare headers:
    EVAL : EV by region (1D, REG)
    ALET : Allocative efficiency total by region (1D, REG)
    ALEF : Allocative efficiency by source × region (2D, BUCKET × REG)
    TOTE : Terms-of-trade by region (1D, REG)
    ISE  : Investment-Savings effect by region (1D, REG)
    ENDW : Endowment effect by region (1D, REG)
    TECH : Technical change effect by region (1D, REG)
    TOT  : Total decomposition (= sum of above) by region (1D, REG)

    HAR header names are limited to 4 characters (GEMPACK convention).

Format: GEMPACK HAR (Fortran unformatted records, little-endian, REFULL/1CFULL).
Round-tripped against `equilibria.babel.har.reader.read_har` for validation.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from equilibria.templates.gtap.welfare_decomp import ALLOC_BUCKETS, WelfareComponents

_PAD = b"    "
_INT = struct.Struct("<i")


def _record(payload: bytes) -> bytes:
    """Frame a Fortran unformatted record: <len><payload><len>."""
    n = len(payload)
    return _INT.pack(n) + payload + _INT.pack(n)


def _pad12(s: str) -> bytes:
    return s.upper().ljust(12)[:12].encode("ascii")


def _name_record(name: str) -> bytes:
    """4-char header name, padded."""
    return _record(name.upper().ljust(4)[:4].encode("ascii"))


def _meta_record(
    type_token: str, long_name: str, *, n_total: int = 0, width: int = 12
) -> bytes:
    """92-byte metadata: pad(4) + type(6) + long_name(70) + 3 trailing ints (12).

    Layout matches GEMPACK convention observed in real HAR files:
    trailing ints at offset 80 = (2, n_total, width). The reader reads
    (n_total, width) from offset 84 — i.e. the 2nd and 3rd of these.
    """
    payload = (
        _PAD
        + type_token.ljust(6)[:6].encode("ascii")
        + long_name.ljust(70)[:70].encode("ascii")
        + struct.pack("<3i", 2, n_total, width)
    )
    return _record(payload)


def _set_descriptor(coeff_name: str, set_names: Iterable[str]) -> bytes:
    """Set descriptor record (REFULL): pad + n_unique + flag + ndim + coeff + flag + names."""
    sn = list(set_names)
    unique = []
    for s in sn:
        if s not in unique:
            unique.append(s)
    payload = (
        _PAD
        + _INT.pack(len(unique))
        + _INT.pack(1)
        + _INT.pack(len(sn))
        + _pad12(coeff_name)
        + _INT.pack(1)
        + b"".join(_pad12(s) for s in sn)
    )
    return _record(payload)


def _set_element_record(elements: list[str], width: int = 12) -> bytes:
    """Element record: pad + flag(1=last) + n_total + n_here + names."""
    n = len(elements)
    payload = (
        _PAD
        + _INT.pack(1)
        + _INT.pack(n)
        + _INT.pack(n)
        + b"".join(_pad12(e) for e in elements)
    )
    return _record(payload)


def _dim_summary_record(shape: tuple[int, ...]) -> bytes:
    """Dim summary: pad + ndim + total_elems."""
    n = int(np.prod(shape)) if shape else 1
    payload = _PAD + _INT.pack(len(shape)) + _INT.pack(n)
    return _record(payload)


def _dim_metadata_record(shape: tuple[int, ...]) -> bytes:
    """Per-dim sizes."""
    payload = _PAD + b"".join(_INT.pack(s) for s in shape)
    return _record(payload)


def _data_record(arr: np.ndarray) -> bytes:
    """REFULL data: pad + int(=n) + n float32 (Fortran order)."""
    flat = np.asarray(arr, dtype=np.float32).flatten(order="F")
    n = flat.size
    payload = _PAD + _INT.pack(n) + flat.tobytes()
    return _record(payload)


def _write_1cfull(name: str, long_name: str, elements: list[str]) -> bytes:
    """Write a 1CFULL set header."""
    return (
        _name_record(name)
        + _meta_record("1CFULL", long_name, n_total=len(elements), width=12)
        + _set_element_record(elements)
    )


def _write_refull(
    name: str,
    long_name: str,
    coeff_name: str,
    set_names: list[str],
    set_elements: list[list[str]],
    array: np.ndarray,
) -> bytes:
    """Write a REFULL real-dense N-D header."""
    unique_names: list[str] = []
    for sn in set_names:
        if sn not in unique_names:
            unique_names.append(sn)
    name_to_elems = dict(zip(set_names, set_elements, strict=False))
    shape = tuple(len(set_elements[i]) for i in range(len(set_names)))

    blob = _name_record(name) + _meta_record("REFULL", long_name)
    blob += _set_descriptor(coeff_name, set_names)
    for sn in unique_names:
        blob += _set_element_record(name_to_elems[sn])
    blob += _dim_summary_record(shape)
    blob += _dim_metadata_record(shape)
    blob += _data_record(array)
    return blob


def write_welview_har(
    output_path: Path,
    welfare: Mapping[str, WelfareComponents],
) -> None:
    """Write WELVIEW.har with welfare decomposition components.

    Args:
        output_path: destination .har file
        welfare:     {region: WelfareComponents} from compute_welfare_decomposition
    """
    regions = sorted(welfare.keys())
    if not regions:
        raise ValueError("welfare dict is empty — nothing to write")

    bucket_codes = [b.upper()[:12] for b in ALLOC_BUCKETS]

    # Component vectors over regions
    ev_vec = np.array([welfare[r].EV for r in regions], dtype=np.float32)
    a_vec = np.array([welfare[r].A_total for r in regions], dtype=np.float32)
    t_vec = np.array([welfare[r].T for r in regions], dtype=np.float32)
    is_vec = np.array([welfare[r].IS for r in regions], dtype=np.float32)
    endw_vec = np.array([welfare[r].ENDW for r in regions], dtype=np.float32)
    tech_vec = np.array([welfare[r].TECH for r in regions], dtype=np.float32)
    total_vec = np.array([welfare[r].total for r in regions], dtype=np.float32)

    # Allocative breakdown: bucket × region
    alef_mat = np.zeros((len(ALLOC_BUCKETS), len(regions)), dtype=np.float32)
    for j, r in enumerate(regions):
        comp = welfare[r]
        for i, bucket in enumerate(ALLOC_BUCKETS):
            alef_mat[i, j] = comp.A.get(bucket, 0.0)

    blob = b""
    # Sets first (referenced by REFULL headers)
    blob += _write_1cfull("REG", "Regions in welfare report", regions)
    blob += _write_1cfull("ALSR", "Allocative-efficiency sources", bucket_codes)

    # 1D headers (REG) — write each as REFULL with single dim
    for header, long_name, vec in [
        ("EVAL", "Equivalent variation by region (USD M)", ev_vec),
        ("ALET", "Allocative efficiency total by region (USD M)", a_vec),
        ("TOTE", "Terms-of-trade effect by region (USD M)", t_vec),
        ("ISE", "Investment-Savings imbalance by region (USD M)", is_vec),
        ("ENDW", "Endowment effect by region (USD M)", endw_vec),
        ("TECH", "Technical change effect by region (USD M)", tech_vec),
        ("TOT", "Total decomposition (sum of components, USD M)", total_vec),
    ]:
        blob += _write_refull(
            header,
            long_name,
            header,
            ["REG"],
            [regions],
            vec,
        )

    # 2D header: ALSR × REG
    blob += _write_refull(
        "ALEF",
        "Allocative efficiency by source and region (USD M)",
        "ALEF",
        ["ALSR", "REG"],
        [bucket_codes, regions],
        alef_mat,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(blob)
