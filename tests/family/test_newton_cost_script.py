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


@pytest.mark.slow
@pytest.mark.parametrize(
    "variant", ["pairing", "pairing_explicit_zeros", "matching", "mumps_auto"]
)
def test_newton_cost_colperm_variants_on_3x3(variant):
    s = _script()
    out = s.measure("gtap7_3x3", capture_dir(), colperm=variant)
    assert out["colperm"] == variant
    # The ICNTL recorded is the one handed to MUMPS (mumps_auto: ICNTL(6)=7).
    want6 = 7 if variant == "mumps_auto" else 0
    assert out["ICNTL"]["6"] == want6
    for c in out["calls"]:
        assert c["variant"] == variant
        assert c["icntl"] == out["ICNTL"]
        assert c["mumps_ok"] and c["residual"] < 1e-8
        for k in ("infog1", "infog28", "infog9", "infog29", "infog12", "infog13"):
            assert isinstance(c[k], int), k
        assert "hard_pairs_dropped" not in c
        if variant == "matching":
            assert c["pairs_moved"] >= c["in_pattern_pairs_dropped"] >= 0
    assert out["newton_projected_s"] > 0


@pytest.mark.slow
def test_newton_cost_orderings_on_3x3():
    """Pyomo's order and pattern: the bijection holds, each variant reports its nnz and
    finite stats, and the family variants take the same Newton step."""
    import json
    import math

    s = _script()
    out = s.measure("gtap7_3x3", capture_dir(), orderings=True)
    json.dumps(out)  # the structure the script writes
    stats = ("symbolic_s", "factor_s", "solve_s")
    infog = ("infog9", "infog29", "infog13")
    for c in out["calls"]:
        o = c["orderings"]
        bij = o["bijection"]
        assert bij["bijection"], bij  # every name maps, none dropped
        assert bij["rows"]["unmapped"] == bij["cols"]["unmapped"] == 0
        assert isinstance(o["pairs_not_pyomo"], int)
        for name in s.ORDERINGS:
            v = o[name]
            assert "skipped" not in v, (name, v)
            assert v["variant"] == name and v["icntl"]["6"] == 0 and v["mumps_ok"]
            assert v["icntl"] == o["pairing"]["icntl"]  # no ICNTL change between orders
            assert isinstance(v["nnz"], int) and v["nnz"] > 0, name
            for k in stats:
                assert math.isfinite(v[k]) and v[k] > 0, (name, k)
            for k in infog:
                assert isinstance(v[k], int), (name, k)
            assert v["residual"] < 1e-8, name
        # Same linear system, only reordered / padded with explicit zeros: same step
        # up to the rounding of a different pivot sequence (the J is ill-conditioned).
        for name in ("pyomo_order", "pyomo_order_pyomo_pattern"):
            assert o[name]["step_rel_diff"] < 1e-6, (name, o[name]["step_rel_diff"])
        assert o["pyomo_order_pyomo_pattern"]["nnz"] >= o["pairing"]["nnz"]
        assert o["pyomo_order"]["nnz"] == o["pairing"]["nnz"]
        # Pyomo's J carries real cross-period rows where the family has unit proxies:
        # a different system, so its step is only reported.
        assert math.isfinite(o["pyomo_J"]["step_rel_diff"])
        for name in s.ORDERINGS:
            v = o[name]
            print(
                f"{c['period']:6s} {name:26s} nnz {v['nnz']:6d} sym {v['symbolic_s']:.4f} "
                f"fac {v['factor_s']:.4f} sol {v['solve_s']:.4f} delayed {v['infog13']} "
                f"infog9 {v['infog9']} infog29 {v['infog29']} "
                f"step {v.get('step_rel_diff', 0.0):.2e}"
            )
        print(
            "pairs not Pyomo's:",
            o["pairs_not_pyomo"],
            "bijection:",
            o["bijection"]["bijection"],
        )
