"""Task 5: the calibrated point (Julia's calibrated_data) loads for the port.

Rather than re-derive Julia's ~60-line share back-out (error-prone, no added
value), the port loads Julia's own calibrated point — the faithful reference per
the spec. dump_calibrated.jl emits it; load_calibrated reads it into a dict of
np arrays keyed by param/quantity name.
"""

import numpy as np
import pytest

from equilibria.templates.gtap_julia.calibration import (
    dump_and_load_calibrated,
    load_calibrated,
)


@pytest.mark.slow
def test_calibrated_point_dumps_and_loads(tmp_path):
    csv = dump_and_load_calibrated(dataset="sample", out_dir=tmp_path)
    cal = load_calibrated(csv)
    # calibration produces the CES share params the equations consume
    for k in ("α_qintva", "α_qfe", "α_qxs", "γ_qfe"):
        assert k in cal, f"missing calibrated param {k}"
    # α_qfe is a dict {idx_tuple: value}; shares are finite and non-negative
    vals = [v for v in cal["α_qfe"].values() if np.isfinite(v)]
    assert len(vals) > 0 and all(v >= 0 for v in vals)


def test_load_calibrated_parses_indexed_and_scalar(tmp_path):
    # a tiny synthetic CSV: one indexed key, one scalar key
    p = tmp_path / "cal.csv"
    p.write_text("α_qfe,land,crops,usa,0.4\nα_qfe,capital,crops,usa,0.6\nσ_qinv,1.0\n")
    cal = load_calibrated(p)
    assert cal["σ_qinv"].shape == () or float(cal["σ_qinv"]) == 1.0
    assert cal["α_qfe"][("land", "crops", "usa")] == 0.4
