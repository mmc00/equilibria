"""CI must have what the photo comparisons need, or they fail instead of skipping."""

from __future__ import annotations

import importlib.util
import os

import pytest

from equilibria.family import photos


@pytest.mark.skipif(os.environ.get("CI") != "true", reason="checks the CI image")
def test_ci_has_the_photos_linear_solver():
    # the photos are MUMPS runs; a family run matches one at 1e-9 only with MUMPS
    assert photos.LINSOLVE == "mumps"
    assert importlib.util.find_spec("mumps") is not None, "pymumps missing in CI"
