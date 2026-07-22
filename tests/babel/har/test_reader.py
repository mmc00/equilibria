"""Tests for the native pure-Python HAR reader.

Uses GTAP HAR/`.prm` fixtures already committed under
`src/equilibria/templates/reference/gtap/data/`, so the suite has no
external-path dependency. Golden values were captured against `harpy3`
(see NOTICE) and cover the four supported types plus edge cases:

  - 1CFULL with multi-record element list (9x10 MCOM/MREG)
  - REFULL with repeated set names (NUS333 VMSB on COMM×REG×REG)
  - REFULL 3-D with three distinct sets (NUS333 VDFB)
  - 2IFULL with shape (1,1) (NUS333 default.prm RDLT)
  - RESPSE sparse arrays (NUS333 MAKS)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from equilibria.babel.har import HeaderArray, get_header_names, read_har
from equilibria.babel.har.reader import read_header_array

REPO_ROOT = Path(__file__).resolve().parents[3]
NUS333 = REPO_ROOT / "src/equilibria/templates/reference/gtap/data/nus333"
NINEX10 = REPO_ROOT / "src/equilibria/templates/reference/gtap/data/9x10"

NUS333_BASE = NUS333 / "basedata.har"
NUS333_SETS = NUS333 / "sets.har"
NUS333_RATE = NUS333 / "baserate.har"
NUS333_PRM = NUS333 / "default.prm"
NINEX10_SETS = NINEX10 / "sets.har"
NINEX10_BASE = NINEX10 / "basedata.har"


# ── Header counts (golden against harpy3) ────────────────────────────────────


@pytest.mark.parametrize(
    "path,expected",
    [
        (NUS333_SETS, 14),
        (NUS333_BASE, 37),
        (NUS333_RATE, 16),
        (NUS333_PRM, 21),
        (NINEX10_SETS, 24),
        (NINEX10_BASE, 50),
    ],
)
def test_header_count(path: Path, expected: int):
    assert len(read_har(path)) == expected


# ── HeaderArray dataclass ────────────────────────────────────────────────────


def test_header_array_creation():
    arr = HeaderArray(
        name="VDPP",
        coeff_name="VDPP",
        long_name="domestic purchases by households",
        array=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        set_names=["COMM", "REG"],
        set_elements=[["AGR", "MFG", "SER"], ["USA", "ROW"]],
    )
    assert arr.rank == 2
    assert arr.shape == (3, 2)


# ── 1CFULL: 1-D character set ────────────────────────────────────────────────


def test_1cfull_single_record_set():
    """NUS333 REG: 2 elements, fits in one element record."""
    d = read_har(NUS333_SETS)
    reg = d["REG"]
    assert reg.shape == (2,)
    assert [str(e).strip() for e in reg.array] == ["USA", "ROW"]


def test_1cfull_multi_record_set():
    """9x10 MCOM: 57 entries split across two element records.

    MCOM is a mapping set — 29 unique commodities expanded to 57 source
    positions. The element data spans two records (29 + 28).
    """
    d = read_har(NINEX10_SETS)
    mcom = d["MCOM"]
    assert mcom.shape == (57,)
    vals = [str(e).strip() for e in mcom.array]
    assert vals[0] == "c_Crops"
    assert vals[8] == "c_MeatLstk"
    assert vals[56] == "c_OthService"


def test_1cfull_large_multi_record():
    """9x10 MREG: 140 entries also multi-record."""
    d = read_har(NINEX10_SETS)
    mreg = d["MREG"]
    assert mreg.shape == (140,)
    assert str(mreg.array[0]).strip() == "Oceania"


# ── REFULL: real dense N-D ───────────────────────────────────────────────────


def test_refull_2d():
    d = read_har(NUS333_BASE)
    vdpp = d["VDPP"]
    assert vdpp.shape == (3, 2)
    assert vdpp.set_names == ["COMM", "REG"]
    assert vdpp.array.dtype == np.float32
    assert float(vdpp.array.sum()) > 0


def test_refull_3d_distinct_sets():
    """VDFB: COMM × ACTS × REG (3 distinct sets)."""
    d = read_har(NUS333_BASE)
    vdfb = d["VDFB"]
    assert vdfb.shape == (3, 3, 2)
    assert vdfb.set_names == ["COMM", "ACTS", "REG"]
    # Golden sum captured against harpy3.
    assert float(vdfb.array.sum()) == pytest.approx(45069360.0, rel=1e-6)


def test_refull_3d_repeated_set():
    """VMSB: COMM × REG × REG — REG appears twice but only stored once on disk.

    This was the edge case that broke the first cut of the descriptor parser
    (n_unique=2 vs ndim=3). The reader must read 2 element records but emit
    a 3-tuple shape with REG repeated.
    """
    d = read_har(NUS333_BASE)
    vmsb = d["VMSB"]
    assert vmsb.shape == (3, 2, 2)
    assert vmsb.set_names == ["COMM", "REG", "REG"]
    assert vmsb.set_elements == [
        ["AGR", "MFG", "SER"],
        ["USA", "ROW"],
        ["USA", "ROW"],
    ]
    assert float(vmsb.array.sum()) == pytest.approx(15662711.0, rel=1e-6)


# ── RESPSE: real sparse N-D ──────────────────────────────────────────────────


def test_respse_3d():
    """MAKS in basedata.har is a 3-D sparse REAL array."""
    d = read_har(NUS333_BASE)
    maks = d["MAKS"]
    assert maks.array.dtype == np.float32
    assert len(maks.shape) >= 2
    # Sparse arrays have a fair number of zeros but at least some nonzero.
    assert (maks.array != 0).any()


# ── 2IFULL: 2-D integer dense ────────────────────────────────────────────────


def test_2ifull_minimal_shape():
    """RDLT in default.prm: shape (1,1) — exposed off-by-4-bytes bug.

    Earlier the reader used meta offsets 80/84 (which read 2/1) and produced
    shape (2,1). The correct offsets are 84/88 → (1,1).
    """
    d = read_har(NUS333_PRM)
    rdlt = d["RDLT"]
    assert rdlt.shape == (1, 1)
    assert rdlt.array.dtype == np.int32


# ── select_headers filter ────────────────────────────────────────────────────


def test_select_headers_filters():
    d = read_har(NUS333_BASE, select_headers=["VDPP", "VKB"])
    assert set(d.keys()) == {"VDPP", "VKB"}


def test_read_header_array_named():
    ha = read_header_array(NUS333_BASE, "VDPP")
    assert ha.name == "VDPP"
    assert ha.shape == (3, 2)


def test_read_header_array_missing_raises():
    with pytest.raises(KeyError):
        read_header_array(NUS333_BASE, "NOPE")


# ── Convenience APIs ─────────────────────────────────────────────────────────


def test_get_header_names():
    names = get_header_names(NUS333_BASE)
    assert "VDPP" in names
    assert "VMSB" in names


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        read_har(Path("/does/not/exist.har"))


def test_public_api_import():
    from equilibria.babel.har import HeaderArray, get_header_names, read_har

    assert callable(read_har)
    assert callable(get_header_names)
    assert HeaderArray is not None


# ── GTAP loader integration (uses repo-shipped HAR) ──────────────────────────


def test_gtap_sets_load_from_har():
    from equilibria.templates.gtap.gtap_sets import GTAPSets

    sets = GTAPSets()
    sets.load_from_har(NUS333_SETS)
    assert sets.r == ["USA", "ROW"]
    assert sets.i == ["AGR", "MFG", "SER"]
    assert sets.a == ["AGR", "MFG", "SER"]
    assert {"LAND", "LABOR", "CAPITAL"}.issubset(set(sets.f))
    assert sets.marg == ["SER"]


def test_gtap_benchmark_load_from_har():
    from equilibria.templates.gtap.gtap_parameters import GTAPBenchmarkValues
    from equilibria.templates.gtap.gtap_sets import GTAPSets

    sets = GTAPSets()
    sets.load_from_har(NUS333_SETS)
    bench = GTAPBenchmarkValues()
    bench.load_from_har(NUS333_BASE, sets)
    assert bench.vdpp.get(("USA", "AGR"), 0.0) > 0
    assert len(bench.vmsb) > 0
    val = bench.vdpp.get(("USA", "AGR"), 0.0)
    assert 0.0 < val < 1.0
    assert isinstance(next(iter(bench.save.keys())), str)


def test_gtap_elasticities_load_from_har():
    from equilibria.templates.gtap.gtap_parameters import GTAPElasticities
    from equilibria.templates.gtap.gtap_sets import GTAPSets

    sets = GTAPSets()
    sets.load_from_har(NUS333_SETS)
    elast = GTAPElasticities()
    elast.load_from_har(NUS333_PRM, sets)
    assert elast.esubd[("USA", "AGR")] > 0
    assert ("USA", "AGR") in elast.esubm
    assert ("USA", "AGR") in elast.esubva
    assert "LAND" in elast.etrae


def test_gtap_taxes_load_from_har():
    from equilibria.templates.gtap.gtap_parameters import (
        GTAPBenchmarkValues,
        GTAPTaxRates,
    )
    from equilibria.templates.gtap.gtap_sets import GTAPSets

    sets = GTAPSets()
    sets.load_from_har(NUS333_SETS)
    bench = GTAPBenchmarkValues()
    bench.load_from_har(NUS333_BASE, sets)
    taxes = GTAPTaxRates()
    taxes.load_from_har(NUS333_RATE, sets, bench)
    assert len(taxes.imptx) > 0
    assert len(taxes.rtxs) > 0


def test_gtap_parameters_load_from_har_roundtrip():
    from equilibria.templates.gtap.gtap_parameters import GTAPParameters

    params = GTAPParameters()
    params.load_from_har(
        basedata_path=NUS333_BASE,
        sets_path=NUS333_SETS,
        default_path=NUS333_PRM,
        baserate_path=NUS333_RATE,
    )
    assert params.sets.r == ["USA", "ROW"]
    assert params.sets.i == ["AGR", "MFG", "SER"]
    assert params.benchmark.vdpp.get(("USA", "AGR"), 0.0) > 0
    assert params.elasticities.esubd.get(("USA", "AGR"), 0.0) > 0
    assert len(params.taxes.imptx) > 0
