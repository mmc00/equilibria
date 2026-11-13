"""The Pyomo Newton-TR path on 3x3 is bitwise what it was before the extraction."""

import json
import sys

import pytest

pytest.importorskip("pyomo")

from tests.family._newton_golden import golden_path, run_in_child  # noqa: E402

# The interpreters we run: a missing golden there is a failure, not a skip.
SUPPORTED = {("mumps", (3, 12)), ("direct", (3, 13))}


@pytest.mark.slow
@pytest.mark.parametrize("linsolve", ["direct", "mumps"])
def test_pyomo_newton_matches_golden(linsolve):
    if linsolve == "mumps":
        pytest.importorskip("mumps")
    path = golden_path(linsolve)
    if not path.exists():
        if (linsolve, sys.version_info[:2]) in SUPPORTED:
            pytest.fail(f"missing golden for a supported interpreter: {path.name}")
        pytest.skip(f"no golden for this interpreter: {path.name}")
    want = json.loads(path.read_text())
    got = run_in_child(linsolve)
    assert got["code"] == want["code"] == 1
    assert len(got["calls"]) == len(want["calls"])
    for g, w in zip(got["calls"], want["calls"], strict=True):
        assert g == w, (w["n"], w["period"])
    assert got["levels"].keys() == want["levels"].keys()
    bad = [k for k in want["levels"] if got["levels"][k] != want["levels"][k]]
    assert bad == []
