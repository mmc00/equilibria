"""Slow tests are skipped by default in tests/family (opt in with ``-m slow`` or
``EQUILIBRIA_RUN_SLOW=1``). Only the items under this directory are touched, and a
single slow node id run without ``-m slow`` is skipped too."""

import os
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def _wants_slow(config: pytest.Config) -> bool:
    return (
        "slow" in (config.getoption("-m") or "")
        or os.environ.get("EQUILIBRIA_RUN_SLOW") == "1"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    if _wants_slow(config):
        return
    skip = pytest.mark.skip(reason="slow: run with -m slow or EQUILIBRIA_RUN_SLOW=1")
    for item in items:
        if (
            item.path.is_relative_to(HERE)
            and item.get_closest_marker("slow") is not None
        ):
            item.add_marker(skip)
