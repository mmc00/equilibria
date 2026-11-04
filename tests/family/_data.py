"""Dataset loading shared by the family-backend tests."""

from __future__ import annotations

from functools import cache

import pytest

DATASETS = ("gtap7_3x3", "gtap7_10x7", "gtap7_15x10")


@cache
def load_params(dataset: str):
    """GTAPParameters for a bundled gtap7 dataset; SKIP if it is missing."""
    from equilibria.sources import dataset_dir, has_dataset
    from equilibria.templates.gtap import GTAPParameters

    if not has_dataset(dataset):
        pytest.skip(f"{dataset} not available")
    d = dataset_dir(dataset)
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    return p
