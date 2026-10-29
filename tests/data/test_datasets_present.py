"""Every input dataset lives in data/<model>/<dataset>/ with its HAR files."""

from __future__ import annotations

import pytest

from equilibria.sources import dataset_dir

GTAP = [
    f"gtap{v}_{s}"
    for v in (6, 7)
    for s in ("3x3", "3x4", "5x5", "10x7", "15x10", "20x41")
]


@pytest.mark.parametrize("name", [*GTAP, "nus333", "9x10"])
def test_dataset_has_its_har_files(name: str) -> None:
    d = dataset_dir(name)
    for f in ("basedata.har", "sets.har", "default.prm"):
        assert (d / f).is_file(), f"{name}: {f} missing in {d}"
