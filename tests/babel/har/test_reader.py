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
