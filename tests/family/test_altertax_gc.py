"""The GC setup of an ALTERTAX run: one freeze and one raised gen-0 threshold for the
whole run (solve, decomposition path and outputs), restored when the run ends."""

import gc
import types
from pathlib import Path
from typing import Any, cast

import pytest

from equilibria.solver import gc_tuning


@pytest.fixture
def clean_gc(monkeypatch):
    monkeypatch.delenv("EQUILIBRIA_GTAP_GC_FREEZE", raising=False)
    monkeypatch.delenv("EQUILIBRIA_GTAP_GC_THRESHOLD", raising=False)
    gc.unfreeze()
    before = gc.get_threshold()
    yield before
    gc.unfreeze()
    gc.set_threshold(*before)


def _state():
    return gc.get_freeze_count(), gc.get_threshold()[0]


def test_family_scope_freezes_raises_and_restores(clean_gc):
    keep = [[i] for i in range(3000)]
    with gc_tuning._gc_family_scope():
        frozen, t0 = _state()
        assert frozen >= len(keep)
        assert t0 == gc_tuning._GC_THRESHOLD0
    assert _state() == (0, clean_gc[0])
    with pytest.raises(RuntimeError), gc_tuning._gc_family_scope():
        raise RuntimeError
    assert _state() == (0, clean_gc[0])


def test_family_scope_nested_keeps_the_outer_setup(clean_gc):
    """An inner scope (``solve_gtap_family`` inside ``solve_altertax``) neither unfreezes
    nor restores the threshold: what runs after it is still inside the outer scope."""
    with gc_tuning._gc_family_scope():
        with gc_tuning._gc_family_scope():
            inner = _state()
        keep = [[i] for i in range(3000)]  # noqa: F841 - alive, young
        after_inner = _state()
        assert after_inner[1] == gc_tuning._GC_THRESHOLD0
        assert after_inner[0] == inner[0] > 0  # the inner exit did not unfreeze
    assert _state() == (0, clean_gc[0])


def test_family_scope_env_off(clean_gc, monkeypatch):
    monkeypatch.setenv("EQUILIBRIA_GTAP_GC_FREEZE", "0")
    monkeypatch.setenv("EQUILIBRIA_GTAP_GC_THRESHOLD", "0")
    with gc_tuning._gc_family_scope():
        assert _state() == (0, clean_gc[0])


def test_freeze_now_does_not_collect(clean_gc, monkeypatch):
    """``gc.freeze()`` moves every generation to the permanent one; the full collection
    before it walked the ~10M freshly loaded records of 20x41 for 7.4 s."""
    calls = []
    monkeypatch.setattr(gc, "collect", lambda *a: calls.append(a) or 0)
    keep = [[i] for i in range(3000)]
    gc_tuning._gc_freeze_now(True)
    assert calls == []
    assert gc.get_freeze_count() >= len(keep)


def test_solve_altertax_keeps_the_gc_setup_through_the_decomp_path(
    clean_gc, monkeypatch
):
    """The decomposition path runs after ``solve_gtap_family`` returns; it is still
    frozen and under the raised threshold. Everything is restored when the run ends."""
    from equilibria.family import driver as drv
    from equilibria.family.altertax import path as altpath
    from equilibria.family.altertax import run as altrun

    seen = {}
    monkeypatch.setattr(drv, "check_supported", lambda **k: None)

    def fake_solve_family(*a, **k):
        seen["solve"] = _state()
        return types.SimpleNamespace(
            results={"check": {"code": 1}, "shock": {"code": 1}}
        )

    monkeypatch.setattr(drv, "_solve_family", fake_solve_family)

    def fake_path(run, shock, steps):
        seen["path"] = _state()
        return "path"

    monkeypatch.setattr(altpath, "solve_decomp_path", fake_path)
    params = types.SimpleNamespace(sets=types.SimpleNamespace(r=["ROW"]))
    monkeypatch.setattr(altrun, "_check_cells", lambda p, s: None)
    run = altrun.solve_altertax(
        Path("nowhere"),
        params=cast(Any, params),
        decomp_steps=2,
        residual_region="ROW",
    )
    assert run.decomp_path == "path"
    assert seen["solve"][0] > 0 and seen["solve"][1] == gc_tuning._GC_THRESHOLD0
    assert seen["path"][0] > 0 and seen["path"][1] == gc_tuning._GC_THRESHOLD0
    assert _state() == (0, clean_gc[0])


def test_entry_points_run_under_the_family_scope(clean_gc, monkeypatch):
    """``run_altertax`` and the RunGTAP measure script run their outputs inside the same
    scope as the solve."""
    import importlib.util

    root = Path(__file__).resolve().parents[2]
    for name in ("run_altertax", "altertax_rungtap_measure"):
        spec = importlib.util.spec_from_file_location(
            name, root / "scripts" / "family" / f"{name}.py"
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        seen = []

        def body(*a, **k):
            seen.append(_state())
            return 0

        monkeypatch.setattr(mod, "_main", body)
        rc = mod.main([]) if name == "run_altertax" else mod.main()
        assert rc == 0
        assert seen[0][0] > 0 and seen[0][1] == gc_tuning._GC_THRESHOLD0
        assert _state() == (0, clean_gc[0])


def test_backend_load_freezes_its_records_only_inside_a_frozen_scope(
    clean_gc, tmp_path, monkeypatch
):
    """The loaded records are frozen before the collector is back on: otherwise the
    next gen-0 pass walks all of them (1.8 s and 1.2 s on 20x41, one per load). Outside
    a family scope, or with EQUILIBRIA_GTAP_GC_FREEZE=0, nothing is frozen (no one
    would unfreeze it)."""
    from equilibria.backends.family_backend import FamilyBackend

    b = FamilyBackend()
    for k in FamilyBackend._SAVED:
        setattr(b, k, {"rows": [[i] for i in range(3000)]})
    b.save(tmp_path / "r.pkl")

    FamilyBackend.load(tmp_path / "r.pkl")
    assert gc.get_freeze_count() == 0
    with gc_tuning._gc_family_scope():
        before = gc.get_freeze_count()
        loaded = FamilyBackend.load(tmp_path / "r.pkl")
        assert gc.get_freeze_count() >= before + 3000
        assert gc.isenabled()
    assert gc.get_freeze_count() == 0
    del loaded
    monkeypatch.setenv("EQUILIBRIA_GTAP_GC_FREEZE", "0")
    with gc_tuning._gc_family_scope():
        FamilyBackend.load(tmp_path / "r.pkl")
        assert gc.get_freeze_count() == 0
