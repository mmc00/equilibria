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


# ── REFULL: real dense N-D ───────────────────────────────────────────────────

def test_write_refull_2d_roundtrip(tmp_path: Path):
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    ha = HeaderArray(
        name="VDPP",
        coeff_name="VDPP",
        long_name="domestic purchases by households",
        array=arr,
        set_names=["COMM", "REG"],
        set_elements=[["AGR", "MFG", "SER"], ["USA", "ROW"]],
    )
    sets = {
        "COMM": HeaderArray("COMM", "COMM", "commodities",
                            np.array(["AGR","MFG","SER"], dtype=object), [], []),
        "REG":  HeaderArray("REG", "REG", "regions",
                            np.array(["USA","ROW"], dtype=object), [], []),
    }
    out = tmp_path / "vdpp.har"
    write_har(out, {**sets, "VDPP": ha})
    d = read_har(out)
    got = d["VDPP"]
    assert got.shape == (3, 2)
    assert got.set_names == ["COMM", "REG"]
    assert got.set_elements == [["AGR","MFG","SER"], ["USA","ROW"]]
    np.testing.assert_array_equal(got.array, arr)


def test_write_refull_repeated_set_roundtrip(tmp_path: Path):
    """VMSB-style: ndim=3, n_unique=2 (COMM x REG x REG)."""
    arr = np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2)
    ha = HeaderArray(
        name="VMSB",
        coeff_name="VMSB",
        long_name="bilateral trade flows",
        array=arr,
        set_names=["COMM", "REG", "REG"],
        set_elements=[["AGR","MFG","SER"], ["USA","ROW"], ["USA","ROW"]],
    )
    sets = {
        "COMM": HeaderArray("COMM","COMM","commodities",
                            np.array(["AGR","MFG","SER"],dtype=object),[],[]),
        "REG":  HeaderArray("REG","REG","regions",
                            np.array(["USA","ROW"],dtype=object),[],[]),
    }
    out = tmp_path / "vmsb.har"
    write_har(out, {**sets, "VMSB": ha})
    d = read_har(out)
    got = d["VMSB"]
    assert got.shape == (3, 2, 2)
    assert got.set_names == ["COMM", "REG", "REG"]
    np.testing.assert_array_equal(got.array, arr)


def test_write_refull_3d_distinct_sets_roundtrip(tmp_path: Path):
    arr = np.linspace(1.0, 18.0, num=18, dtype=np.float32).reshape(3, 3, 2)
    ha = HeaderArray(
        name="VDFB",
        coeff_name="VDFB",
        long_name="domestic intermediate purchases",
        array=arr,
        set_names=["COMM", "ACTS", "REG"],
        set_elements=[
            ["AGR","MFG","SER"], ["AGR","MFG","SER"], ["USA","ROW"],
        ],
    )
    sets = {
        "COMM": HeaderArray("COMM","COMM","commodities",
                            np.array(["AGR","MFG","SER"],dtype=object),[],[]),
        "ACTS": HeaderArray("ACTS","ACTS","activities",
                            np.array(["AGR","MFG","SER"],dtype=object),[],[]),
        "REG":  HeaderArray("REG","REG","regions",
                            np.array(["USA","ROW"],dtype=object),[],[]),
    }
    out = tmp_path / "vdfb.har"
    write_har(out, {**sets, "VDFB": ha})
    d = read_har(out)
    got = d["VDFB"]
    assert got.shape == (3, 3, 2)
    assert got.set_names == ["COMM", "ACTS", "REG"]
    np.testing.assert_array_equal(got.array, arr)


def test_write_refull_rejects_shape_set_mismatch(tmp_path: Path):
    ha = HeaderArray(
        name="VDPP",
        coeff_name="VDPP",
        long_name="x",
        array=np.zeros((3, 2), dtype=np.float32),
        set_names=["COMM", "REG"],
        set_elements=[["A","B"], ["X","Y"]],
    )
    sets = {
        "COMM": HeaderArray("COMM","COMM","c", np.array(["A","B"],dtype=object),[],[]),
        "REG":  HeaderArray("REG","REG","r", np.array(["X","Y"],dtype=object),[],[]),
    }
    with pytest.raises(ValueError, match="shape"):
        write_har(tmp_path / "bad.har", {**sets, "VDPP": ha})


def test_write_refull_rejects_ndim_set_names_mismatch(tmp_path: Path):
    ha = HeaderArray(
        name="VDPP",
        coeff_name="VDPP",
        long_name="x",
        array=np.zeros((3, 2), dtype=np.float32),
        set_names=["COMM"],
        set_elements=[["A","B","C"]],
    )
    with pytest.raises(ValueError, match="ndim"):
        write_har(tmp_path / "bad.har", {"VDPP": ha})
