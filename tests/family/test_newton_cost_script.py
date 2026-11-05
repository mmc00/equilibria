"""The Newton-cost script, exercised on the 3x3 capture (needs pymumps)."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("jax")
pytest.importorskip("mumps")

from tests.family._prep import capture_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TIMINGS = ("build_s", "FJ_s", "symbolic_s", "factor_s", "solve_s")


def _script():
    path = ROOT / "scripts" / "family" / "newton_cost_20x41.py"
    spec = importlib.util.spec_from_file_location("newton_cost_20x41", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
def test_newton_cost_on_3x3():
    s = _script()
    out = s.measure("gtap7_3x3", capture_dir())
    calls = out["calls"]
    assert [c["period"] for c in calls] == ["check", "check", "shock"]
    for c in calls:
        for t in TIMINGS:
            assert isinstance(c[t], float) and c[t] > 0, t
        assert c["n"] > 0 and isinstance(c["proxy_rows"], int)
        assert isinstance(c["same_pattern_as_prev"], bool)
    assert calls[0]["same_pattern_as_prev"] is False
    nit, nfact = out["NIT"], out["NFACT"]
    assert nit == [3, 3, 22] and nfact == [2, 2, 21]
    want = sum(
        (0.0 if c["same_pattern_as_prev"] else c["symbolic_s"])
        + nit[k] * (c["FJ_s"] + c["solve_s"])
        + nfact[k] * c["factor_s"]
        for k, c in enumerate(calls)
    )
    assert out["newton_projected_s"] == pytest.approx(want)
