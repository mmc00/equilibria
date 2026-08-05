"""GTAP HAR → Julia-named data dict for the gtap_julia port.

Our basedata.har carries standard GTAP value headers in UPPER case; Julia's
model reads them in lower case. load_julia_data reads the HAR and returns the
value arrays keyed by Julia's lower-case header names, so the same dataset feeds
both the Julia oracle and the port.
"""

from __future__ import annotations

import numpy as np

from equilibria.babel.har import read_har

from .sets import dataset_dir

# The value headers the Julia GTAPv7 model consumes (build + starting values).
# Read from our basedata.har by their UPPER-case names, returned lower-cased.
_DATA_HEADERS = [
    "vfob",
    "vcif",
    "vmsb",
    "vxsb",
    "vtwr",
    "vst",
    "vdfp",
    "vmfp",
    "vdpp",
    "vmpp",
    "vdgp",
    "vmgp",
    "vdip",
    "vmip",
    "vdfb",
    "vmfb",
    "vdpb",
    "vmpb",
    "vdgb",
    "vmgb",
    "vdib",
    "vmib",
    "evfp",
    "evfb",
    "evos",
    "makb",
    "maks",
    "vkb",
    "vdep",
    "save",
    "pop",
]

__all__ = ["load_julia_data", "dataset_dir"]


def load_julia_data(dataset: str) -> dict[str, np.ndarray]:
    """Return {julia_header_lower: ndarray} for the dataset.

    Missing optional headers are simply omitted (the port asserts the required
    ones are present).
    """
    har = read_har(str(dataset_dir(dataset) / "basedata.har"))
    out: dict[str, np.ndarray] = {}
    for lower in _DATA_HEADERS:
        ha = har.get(lower.upper())
        if ha is None:
            continue
        out[lower] = np.asarray(ha.array, dtype=float)
    return out
