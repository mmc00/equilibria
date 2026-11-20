"""The saved Pyomo photos: where they live and how the regeneration command writes them."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from equilibria.family import photos
from equilibria.sources import DATA_ROOT, load_source, sha256_file

ROOT = Path(__file__).resolve().parents[2]


def test_photo_dir_name():
    got = photos.photo_dir("gtap7_3x3", 1, "capFix", "gtap")
    assert got == DATA_ROOT / "gtap7" / "gtap7_3x3" / "equilibria_pyomo-photo_tol_today"
    got = photos.photo_dir("gtap7_3x3", 0, "capFix", "gtap")
    assert got.name == "equilibria_pyomo-photo_gtap_ifsub0_capFix"


def test_variants_cover_the_per_step_consumers():
    keep = {(ds, ifsub, savf): k for ds, ifsub, savf, _, k in photos.VARIANTS}
    for v in [
        ("gtap7_3x3", 1, "capFix"),
        ("gtap7_3x3", 0, "capFix"),
        ("gtap7_3x3", 1, "capFlex"),
        ("gtap7_10x7", 1, "capFix"),
    ]:
        assert keep[v] == "all"
    assert keep[("gtap7_15x10", 1, "capFix")] == "gate"
