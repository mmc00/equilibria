"""Fixtures for MANAGE-WB Ghana: the recorded GAMS run, asked for by its variant."""

from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.sources import reference

# The floor of every residual test: GAMS's own rows at its 2015 solution are at
# most this (measured 2.79e-9, test_convert.test_solution_floor).
SOLUTION_FLOOR = 3e-9


def gams_run() -> Path:
    """The nlpec-ipopt run: GHA_BaU.gdx (both years) and the CONVERT files."""
    return reference(
        "gha_gtap9", "gams", form="mcp", task="baseline", year=2015,
        mcp_solver="nlpec-ipopt",
    )  # fmt: skip


@pytest.fixture(scope="session")
def run_dir() -> Path:
    return gams_run()


@pytest.fixture(scope="session")
def bau_path(run_dir: Path) -> Path:
    return run_dir / "GHA_BaU.gdx"


@pytest.fixture(scope="session")
def dict_path(run_dir: Path) -> Path:
    return run_dir / "gha2015_dict.txt"
