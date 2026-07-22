import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "parity"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from _probe_cache import KEY_FILES, compute_cache_key, load_solution, store_solution


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


from _probe_queries import query_residuals, query_show


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
    assert result["exportable_cells"] >= result["cells_set"]
    assert 0.0 <= result["coverage"] <= 1.0
    assert isinstance(result["below_threshold"], bool)


import subprocess


def test_cli_show_runs_and_caches(tmp_path):
    import pytest

    env_cache = tmp_path / "cache"
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/parity/probe.py",
        "--template",
        "gtap",
        "--dataset",
        "gtap7_3x3",
        "--scenario",
        "altertax_check",
        "--show",
        "pi",
        "--region",
        "ROW",
        "--cache-dir",
        str(env_cache),
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
        "uv",
        "run",
        "python",
        "scripts/parity/probe.py",
        "--template",
        "gtap",
        "--dataset",
        "gtap7_3x3",
        "--scenario",
        "altertax_check",
        "--show",
        "pi",
        "--region",
        "ROW",
        "--compare-ref",
        head,
        "--cache-dir",
        str(tmp_path / "c"),
    ]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    if "not available" in (out.stdout + out.stderr):
        pytest.skip("gtap7_3x3 not available")
    assert out.returncode == 0, out.stderr
    assert "HEAD" in out.stdout and "Δ" in out.stdout
    wl = subprocess.run(
        ["git", "worktree", "list"], cwd=ROOT, capture_output=True, text=True
    )
    assert "probe_compare_" not in wl.stdout


from _probe_cache import compute_cache_key as _cck


def test_cache_key_invalidates_on_real_key_file_touch(tmp_path):
    f = tmp_path / "gtap_model_equations.py"
    f.write_text("# v1")
    k1 = _cck("gtap7_3x3", "altertax_check", "altertax", [f])
    f.write_text("# v2  (an equation changed)")
    k2 = _cck("gtap7_3x3", "altertax_check", "altertax", [f])
    assert k1 != k2, "editing a key equation file must invalidate the cache"


def test_seed_gams_coverage_is_vs_exported_not_all_free_vars(tmp_path):
    """Coverage is cells_set / GAMS-exported-cells, NOT / all free vars. So when
    the name mapping is healthy it should be high (~>0.85), unlike the old
    free-var denominator that read ~0.68 and made the 95% gate unhittable."""
    import pytest

    pytest.importorskip("pyomo")
    from _adapter_protocol import AdapterRegistry

    ref = ROOT / "output" / "gtap7_3x3_altertax_neos_bundle" / "out_local.gdx"
    if not ref.exists():
        pytest.skip("reference GDX absent")
    adapter = AdapterRegistry.get("gtap")()
    if ("gtap7_3x3", "altertax_check") not in adapter.enumerate_combinations():
        pytest.skip("gtap7_3x3 not available")
    fresh = adapter.build_warmstarted_model("gtap7_3x3", "altertax_check")
    res = seed_gams_point(fresh, ref, "base")
    # exportable denominator is smaller than total free vars
    assert res["exportable_cells"] <= res["total_free_cells"]
    # healthy mapping → high coverage vs the exported universe
    assert res["coverage"] >= 0.85, (
        f"coverage vs exported too low: {res['coverage']:.2%} "
        f"({res['cells_set']}/{res['exportable_cells']})"
    )


from _probe_params import ALIAS_MAP, extract_params, resolve_gams_symbol


def test_extract_params_with_real_model():
    import pytest

    pytest.importorskip("pyomo")
    from _adapter_protocol import AdapterRegistry

    adapter = AdapterRegistry.get("gtap")()
    if ("gtap7_3x3", "altertax_check") not in adapter.enumerate_combinations():
        pytest.skip("gtap7_3x3 not available")
    m = adapter.build_warmstarted_model("gtap7_3x3", "altertax_check")
    params = extract_params(m)
    assert "pf0" in params and len(params["pf0"]) > 0
    assert all(isinstance(v, float) for v in params["pf0"].values())


def test_alias_map_and_resolve():
    assert ALIAS_MAP["pf0"] == "pf@base"
    name, period = resolve_gams_symbol("pf0")
    assert name == "pf" and period == "base"
    name, period = resolve_gams_symbol("kappaf")
    assert name == "kappaf" and period is None


from _probe_params import diff_params_vs_gams


def test_diff_params_vs_gams_three_groups():
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
    result = diff_params_vs_gams(m, ref, "check", tol_rel=1e-3)
    assert "diverge" in result and "ok" in result and "no_match" in result
    assert isinstance(result["diverge"], list)
    assert isinstance(result["ok"], list)
    assert isinstance(result["no_match"], list)
    assert "kappaf" not in {r["param"] for r in result["diverge"]}


from _probe_params import diff_param_builds


def test_diff_param_builds_runs_and_is_well_formed():
    """diff_param_builds returns a sorted list of {param,cells,changed,max_rel,worst}.

    NOTE: it returns [] on the current (faithful) code — the t0-dependence that
    caused the shock bug was the base solving to MIS-calibrated pf0; now the base
    solves to the calibrated values, so with/without t0 agree. The tool is the
    regression guard: if a future change re-introduces a build-dependent Param,
    this surfaces it. So we assert structure + sortedness, not a specific param.
    """
    import pytest

    pytest.importorskip("pyomo")
    from _adapter_protocol import AdapterRegistry

    if ("gtap7_3x3", "altertax_check") not in AdapterRegistry.get(
        "gtap"
    )().enumerate_combinations():
        pytest.skip("gtap7_3x3 not available")
    changed = diff_param_builds("gtap7_3x3", tol_rel=1e-3)
    assert isinstance(changed, list)
    for r in changed:
        assert {"param", "cells", "changed", "max_rel", "worst"} <= set(r)
        assert r["changed"] >= 1
    rels = [r["max_rel"] for r in changed]
    assert rels == sorted(rels, reverse=True)


def test_cli_params_runs(tmp_path):
    import pytest

    ref = ROOT / "output" / "gtap7_3x3_altertax_neos_bundle" / "out_local.gdx"
    if not ref.exists():
        pytest.skip("reference GDX absent")
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/parity/probe.py",
        "--dataset",
        "gtap7_3x3",
        "--scenario",
        "altertax_check",
        "--params",
        "--gdx-ref",
        str(ref),
        "--cache-dir",
        str(tmp_path / "c"),
    ]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    if "not available" in (out.stdout + out.stderr):
        pytest.skip("gtap7_3x3 not available")
    assert out.returncode == 0, out.stderr
    assert "coverage:" in out.stdout
    assert "param" in out.stdout.lower()
