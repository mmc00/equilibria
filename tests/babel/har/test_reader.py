from equilibria.babel.har.symbols import HeaderArray
import numpy as np


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
