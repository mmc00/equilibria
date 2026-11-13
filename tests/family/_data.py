"""Dataset loading shared by the family-backend tests."""

from __future__ import annotations

from functools import cache

import pytest

DATASETS = ("gtap7_3x3", "gtap7_10x7", "gtap7_15x10")

# The bitwise goldens (np.array_equal on arrays, sha256 of float bytes) were recorded on
# macOS with Python 3.13 and the locked jax/numpy/scipy. Other interpreters and
# platforms round differently in the last bits: measured 2026-11, py3.11 (jax 0.10.2)
# differs on macOS and Linux, and py3.13 on Linux aarch64 differs on the hashed rows
# only. Like test_newton_golden, a golden for another interpreter is a skip, not a
# failure; the tolerance-based parity tests cover those environments.
GOLDEN_PY = (3, 13)


def skip_unless_golden_env(*, platform: str | None = None) -> None:
    """Skip a bitwise-golden test outside the environment its golden was recorded in."""
    import sys

    if sys.version_info[:2] != GOLDEN_PY:
        pytest.skip(
            f"bitwise golden recorded on Python {GOLDEN_PY[0]}.{GOLDEN_PY[1]}, "
            f"this is {sys.version_info[0]}.{sys.version_info[1]}"
        )
    if platform is not None and sys.platform != platform:
        pytest.skip(f"bitwise golden recorded on {platform}, this is {sys.platform}")


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
