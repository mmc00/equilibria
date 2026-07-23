"""Build a tiny synthetic RunGTAP-style updated.har for testing the GEMPACK
cell-by-cell reference path end-to-end WITHOUT a Windows-produced ref.

Replace with the real updated.har when it lands. This is a TEST fixture, not a
coverage-matrix row.
"""

from __future__ import annotations

import numpy as np

from equilibria.babel.har.symbols import HeaderArray
from equilibria.babel.har.writer import write_har


def build_synthetic_updated_har(path: str) -> None:
    regs = ["USA", "ROW"]
    comm = ["Food", "Mnfcs"]
    # one firm-domestic-purchases value header, 2 regions x 2 sectors
    vdfb = HeaderArray(
        name="VDFB",
        coeff_name="VDFB",
        long_name="firm domestic purchases at agents prices (post-shock)",
        array=np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        set_names=["REG", "COMM"],
        set_elements=[regs, comm],
    )
    write_har(path, {"VDFB": vdfb})


if __name__ == "__main__":
    import sys

    build_synthetic_updated_har(
        sys.argv[1]
        if len(sys.argv) > 1
        else "tests/fixtures/gtap7_gempack/updated_synthetic.har"
    )
