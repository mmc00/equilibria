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


@pytest.mark.skipif(os.environ.get("CI") != "true", reason="checks the CI image")
def test_ci_has_jax():
    # without jax the conftest skips every family test, and CI would go green having
    # compared nothing; this file is exempt from that skip so the check can fail
    assert importlib.util.find_spec("jax") is not None, (
        "jax (family extra) missing in CI"
    )
