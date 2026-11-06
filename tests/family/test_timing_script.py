"""The 20x41 timing script, exercised on 3x3 (needs pymumps: the child runs NEWTON_MUMPS)."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("jax")
pytest.importorskip("mumps")

ROOT = Path(__file__).resolve().parents[2]


def _script():
    path = ROOT / "scripts" / "family" / "timing_20x41.py"
    spec = importlib.util.spec_from_file_location("timing_20x41", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
def test_timing_runs_on_3x3(tmp_path):
    s = _script()
    out = tmp_path / "t.json"
    rc = s.main(
        [
            "gtap7_3x3",
            "--runs",
            "2",
            "--cache-root",
            str(tmp_path / "cache"),
            "--out",
            str(out),
            "--force",
        ]
    )
    assert rc == 0
    d = json.loads(out.read_text())
    assert d["runs_measured"] == 2 and len(d["machine"]) == 3
    assert d["process_wall_s"]["median"] >= d["marks"]["solve_shock_s"]["median"] > 0
    for name in ("shock.newton_J", "settle_check.square", "check.post_solve"):
        assert d["phases"][name]["median"] >= 0
    assert d["detail"]["mumps_factor_calls"] > 0
    assert len(d["detail"]["first_J_s_per_system"]) == 3
    assert d["import_time"]["total_s"] > 0 and d["pyomo_imported"] in (True, False)
    # the additive phases explain the family call
    got = sum(v["median"] for v in d["phases"].values())
    assert got <= d["marks"]["family_call_s"]["median"] * 1.001


def test_gc_option_defaults_to_the_route_mode():
    s = _script()
    assert s.run_child.__defaults__[-1] is False  # gc_off: the route's own GC
    src = (ROOT / "scripts" / "family" / "timing_20x41.py").read_text()
    assert 'choices=("route", "off")' in src and 'default="route"' in src
