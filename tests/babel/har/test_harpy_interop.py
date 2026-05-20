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
