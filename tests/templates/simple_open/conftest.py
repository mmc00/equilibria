"""Fixtures for simple_open (Hosoe's stdcge): the SAM, its calibration, GAMS's GDX."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from equilibria.babel.gdx.reader import read_gdx
from equilibria.sources import dataset_dir, reference


def gams_points() -> Path:
    return reference("hosoe_2x2", "gams", form="nlp", experiment="tariff0")


@pytest.fixture(scope="session")
def hosoe_sam_path() -> Path:
    return dataset_dir("hosoe_2x2") / "sam.csv"


@pytest.fixture(scope="session")
def hosoe_sam(hosoe_sam_path: Path) -> Any:
    from equilibria.templates.simple_open import load_sam

    return load_sam(hosoe_sam_path)


@pytest.fixture(scope="session")
def calibration(hosoe_sam: Any) -> Any:
    from equilibria.templates.simple_open import calibrate

    return calibrate(hosoe_sam)


@pytest.fixture(scope="session")
def base_gdx() -> dict[str, Any]:
    return read_gdx(gams_points() / "base.gdx")
