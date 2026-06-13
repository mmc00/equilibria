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
