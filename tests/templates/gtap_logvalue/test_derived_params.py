"""Task 1 gate: native log-value calibration value-shares ≡ Julia dump (1e-9).

`derived_shares` computes the closed-form σ seeds the inverted-closure calibration
consumes. They must match the Julia calibrated point cell-by-cell — the σ are pure
value shares of the benchmark, so any mismatch is a data-translation bug (evfp
reconstruction or key reorder), caught here before it reaches the α/γ solve.
"""

from pathlib import Path

from equilibria.blocks.gtap_logvalue._derived_params import derived_shares
from equilibria.templates.gtap import GTAPParameters
from equilibria.templates.gtap_julia.calibration import load_calibrated

ROOT = Path(__file__).resolve().parents[3]
DS = ROOT / "datasets" / "gtap7_3x3"
FIX = ROOT / "tests" / "fixtures" / "gtap_logvalue" / "julia_3x3_calibrated.csv"


def _params():
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=DS / "basedata.har",
        sets_path=DS / "sets.har",
        default_path=DS / "default.prm",
    )
    return p


def _non_nan(d):
    return {k: v for k, v in d.items() if not (isinstance(v, float) and v != v)}


def test_sigma_qxs_matches_julia():
    p = _params()
    ours = derived_shares(p, p.sets, DS)["sigma_qxs"]
    ref = _non_nan(load_calibrated(FIX)["σ_qxs"])
    assert ref, "no non-nan σ_qxs cells in the reference"
    for key, val in ref.items():
        assert key in ours, f"σ_qxs[{key}] missing from derived_shares"
        assert abs(ours[key] - val) < 1e-9, f"σ_qxs[{key}] {ours[key]} != {val}"


def test_sigma_vff_matches_julia():
    p = _params()
    ours = derived_shares(p, p.sets, DS)["sigma_vff"]
    ref = _non_nan(load_calibrated(FIX)["σ_vff"])
    assert ref, "no non-nan σ_vff cells in the reference"
    for key, val in ref.items():
        assert key in ours, f"σ_vff[{key}] missing from derived_shares"
        assert abs(ours[key] - val) < 1e-9, f"σ_vff[{key}] {ours[key]} != {val}"
