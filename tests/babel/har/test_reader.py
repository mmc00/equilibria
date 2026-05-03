from pathlib import Path

import numpy as np
import pytest

from equilibria.babel.har.symbols import HeaderArray

NUS333_BASE = Path("/Users/marmol/Downloads/10284/basedata.har")
NUS333_SETS = Path("/Users/marmol/Downloads/10284/sets.har")


def test_header_array_creation():
    arr = HeaderArray(
        name="VDPP",
        coeff_name="VDPP",
        long_name="domestic purchases by households",
        array=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        set_names=["COMM", "REG"],
        set_elements=[["AGR", "MFG", "SER"], ["USA", "ROW"]],
    )
    assert arr.name == "VDPP"
    assert arr.rank == 2
    assert arr.shape == (3, 2)


def test_get_header_names():
    from equilibria.babel.har.reader import get_header_names
    names = get_header_names(NUS333_BASE)
    assert "VDPP" in names
    assert "VMSB" in names
    assert "VKB" in names


def test_read_har_returns_dict():
    from equilibria.babel.har.reader import read_har
    data = read_har(NUS333_BASE)
    assert "VDPP" in data
    arr = data["VDPP"]
    assert arr.shape == (3, 2)   # (COMM, REG)
    assert arr.set_names == ["COMM", "REG"]


def test_read_har_3d():
    from equilibria.babel.har.reader import read_har
    data = read_har(NUS333_BASE)
    vmsb = data["VMSB"]
    assert vmsb.shape == (3, 2, 2)   # (COMM, REG, REG)


def test_read_har_select():
    from equilibria.babel.har.reader import read_har
    data = read_har(NUS333_BASE, select_headers=["VDPP", "VKB"])
    assert set(data.keys()) == {"VDPP", "VKB"}


def test_read_header_array_elements():
    from equilibria.babel.har.reader import read_har
    data = read_har(NUS333_SETS)
    reg = data["REG"]
    assert list(reg.set_elements[0]) == ["USA", "ROW"]


def test_missing_file_raises():
    from equilibria.babel.har.reader import read_har
    with pytest.raises(FileNotFoundError):
        read_har(Path("/does/not/exist.har"))


def test_public_api_import():
    from equilibria.babel.har import read_har, get_header_names, HeaderArray
    assert callable(read_har)
    assert callable(get_header_names)


# ── Task 5: GTAPSets.load_from_har ──────────────────────────────────────────

def test_gtap_sets_load_from_har():
    from equilibria.templates.gtap.gtap_sets import GTAPSets
    sets = GTAPSets()
    sets.load_from_har(NUS333_SETS)
    assert sets.r == ["USA", "ROW"]
    assert sets.i == ["AGR", "MFG", "SER"]
    assert sets.a == ["AGR", "MFG", "SER"]
    assert "LAND" in sets.f
    assert "LABOR" in sets.f
    assert "CAPITAL" in sets.f
    assert sets.marg == ["SER"]


# ── Task 6: GTAPBenchmarkValues.load_from_har ────────────────────────────────

def test_gtap_benchmark_load_from_har():
    from equilibria.templates.gtap.gtap_sets import GTAPSets
    from equilibria.templates.gtap.gtap_parameters import GTAPBenchmarkValues
    sets = GTAPSets()
    sets.load_from_har(NUS333_SETS)
    bench = GTAPBenchmarkValues()
    bench.load_from_har(NUS333_BASE, sets)
    # VDPP[USA, AGR] should be non-zero; key order is (r, i)
    assert bench.vdpp.get(("USA", "AGR"), 0.0) > 0
    # VMSB has (r, i, rp) keys
    assert len(bench.vmsb) > 0
    # values scaled by 1e-6 (trillions), ~0.05 for USA AGR
    val = bench.vdpp.get(("USA", "AGR"), 0.0)
    assert 0.0 < val < 1.0
    # save is plain-string keyed
    assert isinstance(next(iter(bench.save.keys())), str)


# ── Task 7: GTAPElasticities.load_from_har ───────────────────────────────────

def test_gtap_elasticities_load_from_har():
    from equilibria.templates.gtap.gtap_sets import GTAPSets
    from equilibria.templates.gtap.gtap_parameters import GTAPElasticities
    sets = GTAPSets()
    sets.load_from_har(NUS333_SETS)
    elast = GTAPElasticities()
    elast.load_from_har(Path("/Users/marmol/Downloads/10284/default.prm"), sets)
    assert ("USA", "AGR") in elast.esubd
    assert elast.esubd[("USA", "AGR")] > 0
    assert ("USA", "AGR") in elast.esubm
    assert ("USA", "AGR") in elast.esubva
    assert "LAND" in elast.etrae   # factor-keyed plain string


# ── Task 8: GTAPTaxRates.load_from_har ───────────────────────────────────────

def test_gtap_taxes_load_from_har():
    from equilibria.templates.gtap.gtap_sets import GTAPSets
    from equilibria.templates.gtap.gtap_parameters import GTAPBenchmarkValues, GTAPTaxRates
    sets = GTAPSets()
    sets.load_from_har(NUS333_SETS)
    bench = GTAPBenchmarkValues()
    bench.load_from_har(NUS333_BASE, sets)
    taxes = GTAPTaxRates()
    taxes.load_from_har(Path("/Users/marmol/Downloads/10284/baserate.har"), sets, bench)
    # imptx should be derived from vcif/vmsb
    assert len(taxes.imptx) > 0
    # rtxs should have some entries
    assert len(taxes.rtxs) > 0
