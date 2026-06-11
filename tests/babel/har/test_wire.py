"""L1: round-trip tests for wire.py codec primitives.

Each encoder must produce bytes that the matching decoder parses back to
the original input. These tests do not touch any HAR fixture — they only
exercise the wire-format primitives in isolation.
"""
from __future__ import annotations

import pytest

from equilibria.babel.har import wire


# ── Fortran record framing ───────────────────────────────────────────────────

@pytest.mark.parametrize("payload", [
    b"",
    b"x",
    b"abcd",
    b"\x00\x01\x02\x03\x04",
    b"hello world",
    b"A" * 1024,
])
def test_write_record_roundtrip(payload):
    out = bytearray()
    wire.write_record(out, payload)
    records = list(wire.iter_records(bytes(out)))
    assert records == [payload]


def test_write_multiple_records_roundtrip():
    out = bytearray()
    payloads = [b"first", b"second-longer", b"", b"\xff\xfe\x00\x01"]
    for p in payloads:
        wire.write_record(out, p)
    assert list(wire.iter_records(bytes(out))) == payloads


# ── Fixed-width string blocks ────────────────────────────────────────────────

def test_encode_decode_str_block_default_width():
    strings = ["USA", "ROW", "AGR"]
    encoded = wire.encode_str_block(strings)
    assert len(encoded) == 12 * 3
    assert wire.decode_str_block(encoded) == strings


def test_encode_decode_str_block_custom_width():
    strings = ["ab", "cd", "ef"]
    encoded = wire.encode_str_block(strings, width=4)
    assert len(encoded) == 4 * 3
    assert wire.decode_str_block(encoded, width=4) == strings


def test_encode_str_block_pads_with_spaces():
    encoded = wire.encode_str_block(["AB"], width=4)
    assert encoded == b"AB  "


def test_encode_str_block_rejects_overlong():
    with pytest.raises(ValueError, match="exceeds fixed width"):
        wire.encode_str_block(["TOOLONG"], width=4)


# ── Set-element record codec ─────────────────────────────────────────────────

def test_set_element_record_roundtrip_single_record():
    elements = ["USA", "ROW", "EUR", "ASI"]
    rec = wire.write_set_element_record(elements, n_total=len(elements), flag=1)
    assert wire.read_set_element_record(rec) == elements


def test_set_element_record_roundtrip_empty():
    rec = wire.write_set_element_record([], n_total=0, flag=1)
    assert wire.read_set_element_record(rec) == []


# ── Set-descriptor codec ─────────────────────────────────────────────────────

def test_set_descriptor_roundtrip_simple():
    payload = wire.build_set_descriptor("VDPP", ["COMM", "REG"])
    coeff, names, ndim = wire.parse_set_descriptor(payload)
    assert coeff == "VDPP"
    assert names == ["COMM", "REG"]
    assert ndim == 2


def test_set_descriptor_roundtrip_repeated_set():
    """VMSB on COMM×REG×REG: ndim=3, n_unique=2."""
    payload = wire.build_set_descriptor("VMSB", ["COMM", "REG", "REG"])
    coeff, names, ndim = wire.parse_set_descriptor(payload)
    assert coeff == "VMSB"
    assert names == ["COMM", "REG", "REG"]
    assert ndim == 3


def test_set_descriptor_roundtrip_3d_distinct():
    payload = wire.build_set_descriptor("VDFB", ["COMM", "ACTS", "REG"])
    coeff, names, ndim = wire.parse_set_descriptor(payload)
    assert coeff == "VDFB"
    assert names == ["COMM", "ACTS", "REG"]
    assert ndim == 3


def test_set_descriptor_rejects_overlong_coeff_name():
    with pytest.raises(ValueError, match="exceeds"):
        wire.build_set_descriptor("THIS_NAME_IS_WAY_TOO_LONG", ["X"])
