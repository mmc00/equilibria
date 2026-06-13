import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "parity"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from _probe_cache import compute_cache_key, store_solution, load_solution, KEY_FILES


def test_cache_key_changes_when_key_file_changes(tmp_path):
    f = tmp_path / "eq.py"
    f.write_text("x = 1")
    k1 = compute_cache_key("gtap7_3x3", "altertax_check", "altertax", [f])
    f.write_text("x = 2")
    k2 = compute_cache_key("gtap7_3x3", "altertax_check", "altertax", [f])
    assert k1 != k2
    assert len(k1) == 16


def test_cache_key_stable_for_same_inputs(tmp_path):
    f = tmp_path / "eq.py"
    f.write_text("x = 1")
    k1 = compute_cache_key("gtap7_3x3", "altertax_check", "altertax", [f])
    k2 = compute_cache_key("gtap7_3x3", "altertax_check", "altertax", [f])
    assert k1 == k2


def test_store_and_load_roundtrip(tmp_path):
    sol = {"pf": {("EU_28", "UnSkLab", "Mnfcs"): 1.45213}, "pi": {("ROW",): 1.0}}
    cache_dir = tmp_path / "cache"
    store_solution("abc123", sol, cache_dir=cache_dir)
    loaded = load_solution("abc123", cache_dir=cache_dir)
    assert loaded == sol


def test_load_missing_returns_none(tmp_path):
    assert load_solution("nope", cache_dir=tmp_path) is None


def test_key_files_are_real_paths():
    for p in KEY_FILES:
        assert p.exists(), f"KEY_FILES entry missing: {p}"


from _probe_queries import extract_solution, inject_solution


def test_extract_inject_roundtrip_with_real_model():
    import pytest
    pytest.importorskip("pyomo")
    from _adapter_protocol import AdapterRegistry
    adapter = AdapterRegistry.get("gtap")()
    if ("gtap7_3x3", "altertax_check") not in adapter.enumerate_combinations():
        pytest.skip("gtap7_3x3 not available")
    m = adapter.build_warmstarted_model("gtap7_3x3", "altertax_check")
    sol = extract_solution(m)
    assert "pf" in sol and len(sol["pf"]) > 0
    from pyomo.environ import value
    key = next(iter(sol["pf"]))
    original = sol["pf"][key]
    m.pf[key].set_value(0.0)
    inject_solution(m, sol)
    assert abs(value(m.pf[key]) - original) < 1e-12


from _probe_queries import query_show, query_residuals


def test_query_show_filters_by_region():
    import pytest
    pytest.importorskip("pyomo")
    from _adapter_protocol import AdapterRegistry
    adapter = AdapterRegistry.get("gtap")()
    if ("gtap7_3x3", "altertax_check") not in adapter.enumerate_combinations():
        pytest.skip("gtap7_3x3 not available")
    m = adapter.build_warmstarted_model("gtap7_3x3", "altertax_check")
    rows = query_show(m, ["pi"], region="ROW")
    assert all(r["var"] == "pi" for r in rows)
    assert all("ROW" in str(r["idx"]) for r in rows)
    assert len(rows) >= 1


def test_query_residuals_sorted_desc():
    import pytest
    pytest.importorskip("pyomo")
    from _adapter_protocol import AdapterRegistry
    adapter = AdapterRegistry.get("gtap")()
    if ("gtap7_3x3", "altertax_check") not in adapter.enumerate_combinations():
        pytest.skip("gtap7_3x3 not available")
    m = adapter.build_warmstarted_model("gtap7_3x3", "altertax_check")
    rows = query_residuals(m, top_n=5)
    assert len(rows) <= 5
    vals = [r["resid"] for r in rows]
    assert vals == sorted(vals, reverse=True)


from _probe_queries import seed_gams_point


def test_seed_gams_reports_coverage():
    import pytest
    pytest.importorskip("pyomo")
    from _adapter_protocol import AdapterRegistry
    ref = ROOT / "output" / "gtap7_3x3_altertax_neos_bundle" / "out_local.gdx"
    if not ref.exists():
        pytest.skip("reference GDX absent")
    adapter = AdapterRegistry.get("gtap")()
    if ("gtap7_3x3", "altertax_check") not in adapter.enumerate_combinations():
        pytest.skip("gtap7_3x3 not available")
    m = adapter.build_warmstarted_model("gtap7_3x3", "altertax_check")
    result = seed_gams_point(m, ref, "base")
    assert result["cells_set"] > 100
    assert 0.0 <= result["coverage"] <= 1.0
    assert isinstance(result["below_threshold"], bool)


import subprocess


def test_cli_show_runs_and_caches(tmp_path):
    import pytest
    env_cache = tmp_path / "cache"
    cmd = [
        "uv", "run", "python", "scripts/parity/probe.py",
        "--template", "gtap", "--dataset", "gtap7_3x3",
        "--scenario", "altertax_check", "--show", "pi", "--region", "ROW",
        "--cache-dir", str(env_cache),
    ]
    r1 = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    if "not available" in (r1.stdout + r1.stderr) or "Skip" in r1.stdout:
        pytest.skip("gtap7_3x3 not available")
    assert r1.returncode == 0, r1.stderr
    assert "pi" in r1.stdout
    r2 = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r2.returncode == 0, r2.stderr
    assert "cache hit" in r2.stdout.lower()


def test_compare_ref_runs_against_head_itself(tmp_path):
    import pytest
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    head = r.stdout.strip()
    cmd = [
        "uv", "run", "python", "scripts/parity/probe.py",
        "--template", "gtap", "--dataset", "gtap7_3x3",
        "--scenario", "altertax_check", "--show", "pi", "--region", "ROW",
        "--compare-ref", head, "--cache-dir", str(tmp_path / "c"),
    ]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    if "not available" in (out.stdout + out.stderr):
        pytest.skip("gtap7_3x3 not available")
    assert out.returncode == 0, out.stderr
    assert "HEAD" in out.stdout and "Δ" in out.stdout
    wl = subprocess.run(["git", "worktree", "list"], cwd=ROOT,
                        capture_output=True, text=True)
    assert "probe_compare_" not in wl.stdout
