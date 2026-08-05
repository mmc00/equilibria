"""GTAP set members for the gtap_julia port, read from our sets.har.

Julia's model indexes over reg/comm/acts/endw/marg plus the endowment
sub-partitions (endwc capital, endws sluggish, endwm mobile, endwf fixed). We
read the same members from our HAR so a dataset feeds both the Julia oracle and
the port.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from equilibria.babel.har import read_har

# repo root: src/equilibria/templates/gtap_julia/sets.py -> up 4 to worktree root
_ROOT = Path(__file__).resolve().parents[4]
_DATASETS = _ROOT / "datasets"

# GTAP set header (UPPER) -> Julia set name (lower). Endowment partitions:
#   ENDW all, ENDS sluggish, ENDM mobile, ENDC capital, ENDF fixed.
_SET_MAP = {
    "REG": "reg",
    "COMM": "comm",
    "ACTS": "acts",
    "ENDW": "endw",
    "MARG": "marg",
    "ENDS": "endws",
    "ENDM": "endwm",
    "ENDC": "endwc",
    "ENDF": "endwf",
}


def dataset_dir(dataset: str) -> Path:
    return _DATASETS / dataset


def build_sets(dataset: str) -> dict[str, list[str]]:
    """Return {julia_set_name: [members]} for the dataset."""
    har = read_har(str(dataset_dir(dataset) / "sets.har"))
    out: dict[str, list[str]] = {}
    for upper, lower in _SET_MAP.items():
        ha = har.get(upper)
        if ha is None:
            continue
        out[lower] = [str(x) for x in np.asarray(ha.array).ravel().tolist()]
    return out
