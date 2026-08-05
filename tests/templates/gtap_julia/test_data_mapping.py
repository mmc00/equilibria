"""Task 2: GTAP HAR → Julia-named data mapping.

Our basedata.har carries the standard GTAP headers in UPPER case (VFOB, VDFP,
EVFP, ...); the Julia model consumes the same headers in lower case (vfob, vdfp,
evfp, ...). load_julia_data reads our HAR and returns lower-cased arrays keyed
the way Julia expects, so the same dataset can feed the Julia oracle AND the port.
"""

import numpy as np

from equilibria.templates.gtap_julia.data import load_julia_data
from equilibria.templates.gtap_julia.sets import build_sets

DATASET = "gtap7_3x3"


def test_sets_have_gtap_dimensions():
    s = build_sets(DATASET)
    assert set(s) >= {"reg", "comm", "acts", "endw", "marg"}
    assert len(s["reg"]) == 3
    # 3x3 has 3 sectors / commodities
    assert len(s["comm"]) == 3


def test_data_has_julia_headers_lowercase():
    d = load_julia_data(DATASET)
    # Julia's model reads these (lower case)
    for h in ("vfob", "vcif", "vmsb", "vdfp", "vmfp", "evfp", "vst", "vtwr"):
        assert h in d, f"missing Julia header {h}"
    # values are finite arrays
    assert np.isfinite(d["vfob"]).any()


def test_vfob_matches_raw_har():
    """The mapped lower-case vfob equals the raw UPPER-case VFOB from the HAR."""
    from equilibria.babel.har import read_har
    from equilibria.templates.gtap_julia.data import dataset_dir

    raw = read_har(str(dataset_dir(DATASET) / "basedata.har"))
    d = load_julia_data(DATASET)
    assert np.allclose(
        np.asarray(d["vfob"]), np.asarray(raw["VFOB"].array, dtype=float), rtol=1e-9
    )
