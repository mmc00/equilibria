"""L7 harpy interop: prove the writer's output is readable by GEMPACKsoftware
harpy3 0.3.1 — not just by our own reader.

This file exists because of issue #12: in v0.4.0 the set descriptor record
emitted by ``build_set_descriptor`` was incomplete (missing the trailing
``set_status``, ``dim_sizes``, and ``Nexplicit`` fields), so REFULL headers
with named sets crashed real harpy3 with a ``struct.error`` even though our
own reader round-tripped them cleanly.

These tests run only when ``harpy`` is importable. CI does not install harpy;
install it via ``uv sync --group har-oracle`` to enable this layer.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from equilibria.babel.har import HarWriter, read_har, write_har
from equilibria.babel.har.symbols import HeaderArray

harpy = pytest.importorskip(
    "harpy",
    reason="harpy3 not installed (install via `uv sync --group har-oracle`)",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
NUS333 = REPO_ROOT / "src/equilibria/templates/reference/gtap/data/nus333"


def test_harpy_reads_writer_refull_with_named_sets(tmp_path: Path):
    """Issue #12 repro: a REFULL header with named sets must be loadable by
    real harpy3, not just by our own reader."""
    out = tmp_path / "for_harpy.har"
    with HarWriter(out) as w:
        w.add_set("REG", ["USA", "ROW"])
        w.add_set("COMM", ["AGR", "MFG"])
        w.add_array(
            "VDPP",
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            set_names=["COMM", "REG"],
            long_name="value of domestic private purchases",
        )

    obj = harpy.HarFileObj.loadFromDisk(str(out))
    names = list(obj.getHeaderArrayNames())
    assert "REG" in names
    assert "COMM" in names
    assert "VDPP" in names

    ha = obj.getHeaderArrayObj("VDPP")
    arr = np.asarray(ha["array"])
    assert arr.shape == (2, 2)
    np.testing.assert_allclose(arr, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_harpy_reads_writer_roundtrip_gtap_basedata(tmp_path: Path):
    """The full GTAP NUS333 basedata.har round-tripped through our writer
    must be readable by real harpy3."""
    src = NUS333 / "basedata.har"
    out = tmp_path / "basedata_rt.har"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        write_har(out, read_har(src))

    obj = harpy.HarFileObj.loadFromDisk(str(out))
    names = list(obj.getHeaderArrayNames())
    assert len(names) > 10
    assert "EVFP" in names  # 3-D REFULL on ENDW x ACTS x REG
    assert "VOSB" in names  # 2-D REFULL on COMM x REG

    ha = obj.getHeaderArrayObj("VOSB")
    arr = np.asarray(ha["array"])
    assert arr.ndim == 2
    assert arr.size > 0


def test_2ifull_is_readable_by_harpy(tmp_path: Path) -> None:
    """2IFULL headers we emit must load in harpy3, at any 2-D shape.

    Issue #86: the meta record's rank slot at byte 80 was written as 0 so that
    rows/cols would land where our own reader looked. GEMPACK requires
    84 + 4*rank == len(meta), so harpy rejected every 2IFULL we wrote with
    "corrupted at dimensions in second Record" -- while our reader round-tripped
    them happily. The data record's 7-int prefix was transposed for the same
    reason: reader and writer agreed with each other and with nothing else.

    Non-square shapes are the point. With cols == 1 a transposed prefix is
    indistinguishable from a correct one, and every 2IFULL in the GEMPACK files
    on hand is N x 1, so only harpy can tell the two apart.
    """
    for shape in [(3, 4), (1, 1), (5, 1), (2, 7)]:
        arr = np.arange(1, int(np.prod(shape)) + 1, dtype=np.int32).reshape(shape)
        ha = HeaderArray(
            name="TEST",
            coeff_name="TEST",
            long_name=f"2IFULL {shape}",
            array=arr,
            set_names=[],
            set_elements=[],
        )
        out = tmp_path / f"i{shape[0]}x{shape[1]}.har"
        write_har(out, {"TEST": ha})

        back = harpy.HarFileObj.loadFromDisk(str(out)).getHeaderArrayObj("TEST")
        got = np.asarray(back["array"])
        assert got.shape == arr.shape, f"{shape}: harpy leyo {got.shape}"
        np.testing.assert_array_equal(got.astype(np.int32), arr)
