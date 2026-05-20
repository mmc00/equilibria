"""L2: writer unit tests.

Per-type-token emission plus HarWriter builder validation. These tests
do not require any GEMPACK fixture — they exercise the writer's own
encoders by constructing HeaderArrays in memory and round-tripping
through the reader.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from equilibria.babel.har import HeaderArray, read_har
from equilibria.babel.har.writer import write_har


# ── 1CFULL: 1-D character set ────────────────────────────────────────────────

def test_write_1cfull_single_record_roundtrip(tmp_path: Path):
    """Set of 2 elements — fits in a single element record."""
    ha = HeaderArray(
        name="REG",
        coeff_name="REG",
        long_name="regions in the model",
        array=np.array(["USA", "ROW"], dtype=object),
        set_names=[],
        set_elements=[],
    )
    out = tmp_path / "out.har"
    write_har(out, {"REG": ha})

    d = read_har(out)
    assert set(d.keys()) == {"REG"}
    got = d["REG"]
    assert got.long_name == "regions in the model"
    assert [str(e).strip() for e in got.array] == ["USA", "ROW"]
    assert got.shape == (2,)


def test_write_har_rejects_empty_headers(tmp_path: Path):
    with pytest.raises(ValueError, match="no headers"):
        write_har(tmp_path / "empty.har", {})


def test_write_har_rejects_overlong_header_name(tmp_path: Path):
    ha = HeaderArray(
        name="TOOLONG",
        coeff_name="TOOLONG",
        long_name="x",
        array=np.array(["A"], dtype=object),
        set_names=[],
        set_elements=[],
    )
    with pytest.raises(ValueError, match="header name"):
        write_har(tmp_path / "bad.har", {"TOOLONG": ha})


def test_write_har_rejects_overlong_long_name(tmp_path: Path):
    ha = HeaderArray(
        name="REG",
        coeff_name="REG",
        long_name="x" * 71,
        array=np.array(["A"], dtype=object),
        set_names=[],
        set_elements=[],
    )
    with pytest.raises(ValueError, match="long_name"):
        write_har(tmp_path / "bad.har", {"REG": ha})


def test_write_har_warns_on_non_uppercase_name(tmp_path: Path):
    ha = HeaderArray(
        name="reg",
        coeff_name="reg",
        long_name="lowercase test",
        array=np.array(["A"], dtype=object),
        set_names=[],
        set_elements=[],
    )
    out = tmp_path / "warn.har"
    with pytest.warns(UserWarning, match="uppercase"):
        write_har(out, {"reg": ha})
    # File is still written verbatim.
    d = read_har(out)
    assert "reg" in d


def test_write_har_atomic_on_failure(tmp_path: Path):
    """If emission raises mid-write, the target path is untouched."""
    out = tmp_path / "target.har"
    out.write_bytes(b"PRE-EXISTING")

    # Inject a failure by passing a HeaderArray with an unsupported dtype.
    ha = HeaderArray(
        name="BAD!",
        coeff_name="BAD!",
        long_name="x",
        array=np.array([1.0+2j], dtype=np.complex64),
        set_names=[],
        set_elements=[],
    )
    with pytest.raises((TypeError, NotImplementedError, ValueError)):
        write_har(out, {"BAD!": ha})
    assert out.read_bytes() == b"PRE-EXISTING"
    assert not (tmp_path / "target.har.tmp").exists()
