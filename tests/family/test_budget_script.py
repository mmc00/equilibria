"""The 20x41 budget script, exercised on 3x3."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("jax")

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = (
    "load_params",
    "compose_state",
    "cache_key",
    "cache_load",
    "layout_system",
    "first_FJ",
    "warm_FJ",
    "names",
    "matching",
)


def _script():
    path = ROOT / "scripts" / "family" / "budget_20x41.py"
    spec = importlib.util.spec_from_file_location("budget_20x41", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
def test_budget_runs_on_3x3(tmp_path):
    b = _script()
    b.run("gtap7_3x3", tmp_path, warm=False)
    out = b.run("gtap7_3x3", tmp_path, warm=True)
    pieces = out["pieces"]
    for name in REQUIRED:
        assert isinstance(pieces[name], float) and pieces[name] > 0, name
    want = (
        pieces["load_params"]
        + pieces["compose_state"]
        + pieces["cache_key"]
        + pieces["cache_load"]
        + 3
        * (
            pieces["layout_system"]
            + pieces["first_FJ"]
            + pieces["names"]
            + pieces["matching"]
        )
        + out["newton_family_s"]
        + out["period_prep_on_arrays_s"]
        + out["prep_once_s"]
    )
    assert out["newton_family_s"] == 23.2
    assert out["period_prep_on_arrays_s"] == 9.3
    assert out["prep_once_s"] == 11.3
    # Pyomo's Newton stays as a reported reference, outside projected_s.
    assert out["newton_today_s"] == 48.0
    assert out["projected_s"] == pytest.approx(want)
    assert out["verdict"] in {"go", "tight", "no"}
    # The formerly unmeasured pieces are measured end to end (timing_20x41.py).
    assert set(out["measured_rest_s"]) == {
        "cross_period_rows",
        "calibration_rest",
        "outside_phases",
    }
    assert out["with_measured_rest_s"] == pytest.approx(
        out["projected_s"] + sum(out["measured_rest_s"].values())
    )
    assert isinstance(out["e2e_process_wall_s"], float)
    assert out["e2e_process_wall_s"] > 0
    assert out["verdict_conditional"] == bool(out["not_measured"])
    assert not (
        {"cross_period_rows", "calibration_rest", "outside_phases"}
        & set(out["not_measured"])
    )
    assert out["if_sub"] == 1
    # Measured now (family_prep_20x41_30d400b.json: incidence_once_s + masks_once_s).
    assert "incidence_from_pattern" not in out["not_measured"]
    # Measured now (prep_cost_20x41.py): neither unmeasured nor in the upper bound.
    assert "period_prep_on_arrays" not in out["not_measured"]
    assert "period_prep_on_arrays" not in out["measured_rest_s"]


def test_a_failing_piece_fails_the_run(tmp_path, monkeypatch):
    b = _script()
    import equilibria.templates.gtap.gtap_block_model as gbm

    def boom(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(gbm, "compose_block_state", boom)
    with pytest.raises(RuntimeError, match="compose_state"):
        b.measure("gtap7_3x3", tmp_path)
