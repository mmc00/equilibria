import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from cascade_run import sweep_period


def _fake_runner(script):
    """script: dict layer_name -> (payload, exit_code). Returns a runner.
    Layers not in `script` default to clean (so adding cascade layers doesn't
    break tests that only care about a subset)."""
    _CLEAN = ({"status": "clean", "meta": {}, "headline": ""}, 0)

    def runner(argv, timeout):
        # argv[1] is the tool script path; map by basename stem to a layer name.
        name = next((k for k in script if k in argv[1]), None)
        return script[name] if name is not None else _CLEAN
    return runner


def test_stops_at_first_dirty():
    gdx = Path("/ref/out.gdx")
    script = {
        "diff_mcp_pairing.py": ({"status": "clean", "meta": {}, "headline": ""}, 0),
        "nl_compare.py": ({"status": "dirty", "meta": {}, "headline": "gap here"}, 1),
        # later layers must NOT run
    }
    results = sweep_period("gtap7_3x3", "shock", gdx,
                           runner=_fake_runner(script))
    names = [r.name for r in results]
    # holdfixed/tautology default to clean here, so the sweep passes them and stops
    # at the first dirty (nl_compare). Later layers must NOT run.
    assert names == ["mcp_pairing", "holdfixed", "tautology", "nl_compare"]
    assert results[-1].name == "nl_compare"
    assert results[-1].action == "explain_stop"
    assert "calibration" not in names


def test_no_convergence_stops_before_seed_layers():
    gdx = Path("/ref/out.gdx")
    script = {
        "diff_mcp_pairing.py": ({"status": "clean", "meta": {}, "headline": ""}, 0),
        "nl_compare.py": ({"status": "clean", "meta": {}, "headline": ""}, 0),
        "diff_calibration.py": ({"status": "clean", "meta": {}, "headline": ""}, 0),
        "validate_reference.py": ({"status": "clean", "meta": {}, "headline": ""}, 0),
        "probe.py": ({"status": "error",
                      "meta": {"error_kind": "no_convergence"},
                      "headline": "did not converge"}, 2),
    }
    results = sweep_period("gtap7_3x3", "shock", gdx,
                           runner=_fake_runner(script))
    assert results[-1].name == "probe_seed"
    assert results[-1].action == "upstream_stop"
    assert "drift_test" not in [r.name for r in results]


def test_vacuity_and_exception_do_not_stop():
    gdx = Path("/ref/out.gdx")
    script = {
        "diff_mcp_pairing.py": ({"status": "error",
                                 "meta": {"error_kind": "exception"},
                                 "headline": "boom"}, 2),
        "nl_compare.py": ({"status": "error",
                           "meta": {"error_kind": "no_common_constraints"},
                           "headline": "no common constraints"}, 2),
        "diff_calibration.py": ({"status": "clean", "meta": {}, "headline": ""}, 0),
        "validate_reference.py": ({"status": "clean", "meta": {}, "headline": ""}, 0),
        "probe.py": ({"status": "clean", "meta": {}, "headline": ""}, 0),
        "drift_test.py": ({"status": "clean", "meta": {}, "headline": ""}, 0),
    }
    results = sweep_period("gtap7_3x3", "shock", gdx,
                           runner=_fake_runner(script))
    actions = {r.name: r.action for r in results}
    assert actions["mcp_pairing"] == "tool_broken_continue"
    assert actions["nl_compare"] == "vacuous_continue"
    assert "drift_test" in actions  # sweep continued to the end


def test_drift_skipped_for_base():
    gdx = Path("/ref/out.gdx")
    script = {
        "diff_mcp_pairing.py": ({"status": "clean", "meta": {}, "headline": ""}, 0),
        "nl_compare.py": ({"status": "clean", "meta": {}, "headline": ""}, 0),
        "diff_calibration.py": ({"status": "clean", "meta": {}, "headline": ""}, 0),
        "validate_reference.py": ({"status": "clean", "meta": {}, "headline": ""}, 0),
        "probe.py": ({"status": "clean", "meta": {}, "headline": ""}, 0),
    }
    results = sweep_period("9x10", "base", gdx, runner=_fake_runner(script))
    drift = [r for r in results if r.name == "drift_test"][0]
    assert drift.status == "skipped"
