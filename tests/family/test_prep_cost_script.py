"""The period-prep cost script, exercised on the 3x3 capture."""

import importlib.util
from pathlib import Path

import pytest

from tests.family._prep import capture_dir

ROOT = Path(__file__).resolve().parents[2]
SQUARE_STEPS = (
    "initialize_benchmark",
    "apply_closure",
    "conditional_fixing",
    "fix_sluggish_pft",
    "deactivate_fixed_quantity_eqs",
    "relative_floors",
    "names",
    "edges",
    "gams_pairs",
    "match",
    "cells",
)


def _script():
    path = ROOT / "scripts" / "family" / "prep_cost_20x41.py"
    spec = importlib.util.spec_from_file_location("prep_cost_20x41", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
def test_prep_cost_on_3x3(tmp_path):
    out = _script().measure("gtap7_3x3", capture_dir(), tmp_path)
    assert out["parity"] is True
    assert [c["period"] for c in out["calls"]] == ["check", "check", "shock"]
    for c in out["calls"]:
        steps = c["square_steps"]
        assert set(SQUARE_STEPS) <= set(steps)
        assert all(isinstance(s, float) and s >= 0 for s in steps.values())
        # Sub-steps are inside square(): they cannot add up to more than it.
        assert sum(steps.values()) <= c["square_s"]
    # Built once per run: apart from the per-call cost.
    assert out["incidence_once_s"] > 0
    assert out["masks_once_s"] > 0
    assert out["prep_total_s"] == pytest.approx(
        out["sp_reference_s"]
        + out["incidence_once_s"]
        + out["masks_once_s"]
        + out["prep_per_call_total_s"]
    )
    # The merge with photo B's cross-period rows is a harness artefact: not in totals.
    assert all(c["incidence_merge_is_harness"] is True for c in out["calls"])
    assert all(c["incidence_merge_s"] >= 0 for c in out["calls"])
    assert out["prep_per_call_total_s"] == pytest.approx(
        sum(c["run_recipe_s"] + c["square_s"] for c in out["calls"])
    )
    assert out["prep_per_call_total_s"] != pytest.approx(
        sum(
            c["run_recipe_s"] + c["square_s"] + c["incidence_merge_s"]
            for c in out["calls"]
        )
    )
    # The prep cost on top of the budget's compose_state: recipe + square per call.
    assert out["prep_per_call_total_s"] == pytest.approx(
        sum(c["run_recipe_s"] + c["square_s"] for c in out["calls"])
    )
