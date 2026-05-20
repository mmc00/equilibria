"""L7: end-to-end integration tests for the HAR writer.

These tests exercise the writer through real equilibria consumers
(the GTAP loader and a DataFrame-based export workflow) plus the
alter-tax-style round-trip simulating a tariff shock.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from equilibria.babel.har import HarWriter, read_har, write_har

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "src/equilibria/templates/reference/gtap/data"
NUS333 = DATA / "nus333"


# ── E2E-1: GTAP loader round-trip ─────────────────────────────────────────────

def test_gtap_loader_consumes_writer_output(tmp_path: Path):
    """Read a HAR, write it back via our writer, and assert that the
    GTAP loader produces the same parameter values from the written copy
    as it does from the original."""
    from equilibria.templates.gtap.gtap_parameters import GTAPBenchmarkValues
    from equilibria.templates.gtap.gtap_sets import GTAPSets

    sets = GTAPSets()
    sets.load_from_har(NUS333 / "sets.har")

    bench_orig = GTAPBenchmarkValues()
    bench_orig.load_from_har(NUS333 / "basedata.har", sets)

    out = tmp_path / "basedata_rt.har"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        write_har(out, read_har(NUS333 / "basedata.har"))

    bench_rt = GTAPBenchmarkValues()
    bench_rt.load_from_har(out, sets)

    assert bench_rt.vdpp == bench_orig.vdpp
    assert bench_rt.vmsb == bench_orig.vmsb
    assert bench_rt.save == bench_orig.save


# ── E2E-2: Alter-tax tariff-shock round-trip ─────────────────────────────────

def test_altertax_tariff_shock_roundtrip(tmp_path: Path):
    """Read baserate.har, multiply the bilateral import-tariff stream rTMS
    by 1.10, write back, re-read, assert the mutation survived intact."""
    original = read_har(NUS333 / "baserate.har")
    assert "rTMS" in original, "baserate.har is expected to carry rTMS"
    tm_orig = original["rTMS"].array.copy()

    original["rTMS"].array[...] = tm_orig * 1.10

    out = tmp_path / "baserate_shocked.har"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        write_har(out, original)

    reread = read_har(out)
    np.testing.assert_allclose(
        reread["rTMS"].array, tm_orig * 1.10, rtol=1e-6, atol=0,
    )


# ── E2E-3: DataFrame → HAR → reader ──────────────────────────────────────────

def test_dataframe_export_end_to_end(tmp_path: Path):
    """Construct a HAR purely from pandas DataFrames and read it back."""
    import pandas as pd

    regy = pd.DataFrame(
        [[100.0, 200.0], [300.0, 400.0]],
        index=pd.Index(["USA", "ROW"], name="REG"),
        columns=pd.Index(["A", "B"], name="X"),
    )
    out = tmp_path / "synthetic.har"
    with HarWriter(out) as w:
        w.add_set("REG", ["USA", "ROW"])
        w.add_set("X", ["A", "B"])
        w.add_dataframe("REGY", regy, set_names=["REG", "X"], long_name="regional y")

    d = read_har(out)
    assert "REGY" in d
    np.testing.assert_array_equal(
        d["REGY"].array.astype(np.float32),
        regy.values.astype(np.float32),
    )
    assert d["REGY"].set_names == ["REG", "X"]
    assert d["REGY"].set_elements == [["USA", "ROW"], ["A", "B"]]


# ── E2E-4: Optional harpy3 interop check ─────────────────────────────────────

def _harpy_available() -> bool:
    try:
        import harpy3  # type: ignore  # noqa
        return True
    except ImportError:
        try:
            import harpy  # type: ignore  # noqa
            return True
        except ImportError:
            return False


@pytest.mark.skipif(not _harpy_available(),
                    reason="harpy3/harpy not installed in this env (expected in CI)")
def test_harpy_reads_writer_output(tmp_path: Path):
    """If harpy3/harpy happens to be installed (local dev sandbox), verify it
    can read our output. CI never has harpy3 installed; this test skips
    there.

    We construct the HAR from scratch via HarWriter rather than round-tripping
    a GTAP fixture: those fixtures contain scalar/no-set REFULL headers (e.g.
    DVER) whose descriptor layout differs between GEMPACK and what locally
    installed harpy builds expect. The L3 semantic round-trip already covers
    that path; this test only needs to verify harpy can read a normally-shaped
    HAR built by our writer.
    """
    try:
        import harpy3 as harpy  # type: ignore
    except ImportError:
        import harpy  # type: ignore

    out = tmp_path / "for_harpy.har"
    with HarWriter(out) as w:
        w.add_set("REG", ["USA", "ROW"])
        w.add_set("COMM", ["AGR", "MFG"])
        w.add_array(
            "VDPP",
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            set_names=["COMM", "REG"],
            long_name="domestic private purchases",
        )

    obj = harpy.HarFileObj(str(out))
    names = list(obj.getHeaderArrayNames())
    assert "REG" in names
    assert "COMM" in names
    assert "VDPP" in names
