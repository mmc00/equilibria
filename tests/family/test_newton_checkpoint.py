"""Task 6 checkpoint on 3x3: the family Newton-TR reaches Pyomo's solution.

The reference is Pyomo's own run with the same ``linsolve`` (the loop stops at
``TR_RELTOL``, so two trajectories end at points ~1e-5 apart on the shock call: against
a ``direct`` reference a ``mumps`` run misses 1e-9 although both are converged).

Gate: every solved cell within 1e-9 of Pyomo's photo P (``|a - b| / max(1, |b|)``) and
``code == 1`` on the three solve calls. ``nit`` is reported, never asserted.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tests.family._prep import _newton_mumps, capture_dir

pytest.importorskip("pyomo")

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/family/newton_checkpoint.py"


def _script():
    spec = importlib.util.spec_from_file_location("newton_checkpoint", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compare_to_p_is_relative_with_unit_floor_and_flags_missing_cells():
    from equilibria.family.prep_capture import Photo, VarCells

    class Sq:
        col_names = ["x[a,t]", "x[b,t]", "y[t]", "z[t]"]

    def cells(keys, val):
        n = len(keys)
        return VarCells(
            keys, np.array(val), np.zeros(n, bool), np.zeros(n), np.full(n, np.inf)
        )

    p = Photo(
        "P",
        1,
        "t",
        {
            "x": cells([("a", "t"), ("b", "t")], [1000.0, 0.5]),
            "y": cells([("t",)], [np.nan]),
        },
        {},
    )
    x = np.array([1000.0 * (1 + 1e-10), 0.5 + 1e-8, np.nan, 3.0])
    out = _script().compare_to_p(Sq, x, p)
    assert out["n_missing_in_P"] == 1  # z is not in P: infinite difference
    assert out["max_rel_diff"] == np.inf
    # x[a]: 1e-7 / 1000 -> 1e-10 relative; x[b]: 1e-8 / max(1, 0.5) = 1e-8 (floor 1)
    assert out["n_over_tol"] == 2  # x[b] over 1e-9 and the missing z
    assert out["worst20"][0][0] == "z[t]"
    assert {w[0]: w[3] for w in out["worst20"]}["y[t]"] == 0.0  # NaN == NaN


def _run(monkeypatch, linsolve):
    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": linsolve}.items():
        monkeypatch.setenv(k, v)
    out = _script().measure("gtap7_3x3", capture_dir("gtap7_3x3", linsolve), linsolve)
    for c in out["calls"]:  # reported, not asserted
        print(
            f"{linsolve} call {c['n']:03d} {c['period']}: nit ours={c['ours']['nit']} "
            f"pyomo={c['pyomo']['nit']} max_rel={c['max_rel_diff']:.2e} code={c['code']}"
        )
    return out


def _assert_gate(out):
    assert [c["period"] for c in out["calls"]] == ["check", "check", "shock"]
    for c in out["calls"]:
        assert c["n_missing_in_P"] == 0, c["worst20"][:3]
        assert c["success"] and c["code"] == 1, c["residual"]
        assert c["max_rel_diff"] <= 1e-9, (c["n"], c["worst20"][:5])
        assert c["ours"]["nit"] >= 1


@pytest.mark.slow
def test_newton_checkpoint_3x3_direct(monkeypatch):
    _assert_gate(_run(monkeypatch, "direct"))


@pytest.mark.slow
def test_newton_checkpoint_3x3_mumps(monkeypatch):
    pytest.importorskip("mumps")
    out = _run(monkeypatch, "mumps")
    _assert_gate(out)
    for c in out["calls"]:
        assert set(c["ours"]["infog1"]) <= {0, 1}, c["ours"]["infog1"]
        assert c["timing"]["factor_s"] > 0
