"""Records cached on disk: a loaded backend gives the same F and J as a fresh build."""

import numpy as np
import pytest

pytest.importorskip("jax")

from tests.family._data import load_params  # noqa: E402
from tests.family._oracle import comp_stat_closure  # noqa: E402

DATASET = "gtap7_3x3"


def _system(backend, state):
    layout = backend.layout(state)
    return layout, backend.system(layout)


def _point(layout):
    u = np.random.default_rng(0).uniform(-1.0, 1.0, layout.n_cols)
    return layout.free_values * (1.0 + 0.05 * u)


@pytest.mark.parametrize("if_sub", [False, True])
def test_roundtrip_same_F_J(tmp_path, if_sub):
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.templates.gtap.gtap_block_model import compose_block_state

    p = load_params(DATASET)
    closure = comp_stat_closure(if_sub)
    model, state = compose_block_state(p, p.sets, closure, list(p.sets.r)[-1])
    fresh = FamilyBackend()
    fresh.build(model, skip=state.ifsub_skipped)
    fresh.save(tmp_path / "records.pkl")
    loaded = FamilyBackend.load(tmp_path / "records.pkl")

    lay_a, sys_a = _system(fresh, state)
    lay_b, sys_b = _system(loaded, state)
    z = _point(lay_a)
    assert np.array_equal(sys_a.F(z), sys_b.F(z))
    ja, jb = sys_a.jac(z), sys_b.jac(z)
    for attr in ("data", "row", "col"):
        assert np.array_equal(getattr(ja, attr), getattr(jb, attr)), attr


def test_key_changes_with_if_sub_and_closure():
    from equilibria.family.disk_cache import cache_key

    p = load_params(DATASET)
    base = comp_stat_closure()
    keys = {
        cache_key(p, base, "ROW"),
        cache_key(p, comp_stat_closure(True), "ROW"),
        cache_key(p, base.model_copy(update={"savf_flag": "capFlex"}), "ROW"),
        cache_key(p, base, "USA"),
        cache_key(load_params("gtap7_10x7"), base, "ROW"),
    }
    assert None not in keys and len(keys) == 5
    assert cache_key(p, base, "ROW") == cache_key(p, comp_stat_closure(), "ROW")


def test_key_changes_with_params_content():
    import copy

    from equilibria.family.disk_cache import cache_key

    p = load_params(DATASET)
    q = copy.deepcopy(p)
    taxes = q.taxes
    name = next(k for k, v in vars(taxes).items() if isinstance(v, dict) and v)
    cell = next(iter(getattr(taxes, name)))
    getattr(taxes, name)[cell] = float(getattr(taxes, name)[cell]) + 0.01
    assert cache_key(p, comp_stat_closure(), "ROW") != cache_key(
        q, comp_stat_closure(), "ROW"
    )


def test_key_changes_with_family_code(tmp_path, monkeypatch):
    import shutil

    from equilibria.family import disk_cache

    p = load_params(DATASET)
    src = tmp_path / "family"
    shutil.copytree(disk_cache._FAMILY_DIR, src)
    monkeypatch.setattr(disk_cache, "_FAMILY_DIR", src)
    k0 = disk_cache.cache_key(p, comp_stat_closure(), "ROW")
    (src / "recorder.py").write_text((src / "recorder.py").read_text() + "\n# edit\n")
    assert disk_cache.cache_key(p, comp_stat_closure(), "ROW") != k0


def test_no_key_under_overwrite_hooks(monkeypatch):
    from equilibria.blocks.gtap.overwrite import overwrite
    from equilibria.family.disk_cache import cache_key

    monkeypatch.setattr(type(overwrite), "active", lambda self: True)
    assert cache_key(load_params(DATASET), comp_stat_closure(), "ROW") is None


def test_interleaved_saves_leave_a_whole_file(tmp_path, monkeypatch):
    """A second save to the same path while the first is writing must not corrupt
    it (a fixed temp name made both write into one inode)."""
    import pickle

    from equilibria.backends.family_backend import FamilyBackend

    b = FamilyBackend()
    b.records = {"eq": [((), 1.0)]}
    path = tmp_path / "records.pkl"
    real_dump = pickle.dump
    nested = []

    def dump(obj, fh, *a, **k):
        if not nested:
            nested.append(1)
            b.save(path)  # the other writer, mid-way through ours
        return real_dump(obj, fh, *a, **k)

    monkeypatch.setattr(pickle, "dump", dump)
    b.save(path)
    assert FamilyBackend.load(path).records == b.records
    assert [f.name for f in tmp_path.iterdir()] == ["records.pkl"]


def test_load_or_build_writes_then_reads(tmp_path, monkeypatch):
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.family.disk_cache import load_or_build

    p = load_params(DATASET)
    closure = comp_stat_closure(True)
    rr = list(p.sets.r)[-1]
    b1, s1 = load_or_build(p, closure, rr, tmp_path)
    assert list(tmp_path.glob("*/records.pkl"))

    def boom(self, *a, **k):
        raise AssertionError("build called on a cache hit")

    monkeypatch.setattr(FamilyBackend, "build", boom)
    b2, s2 = load_or_build(p, closure, rr, tmp_path)
    assert set(b2.records) == set(b1.records)
    assert s2 is not s1
