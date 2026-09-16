"""Tests for the built-model disk cache (src/equilibria/blocks/gtap/model_cache.py).

The cache stores a whole ConcreteModel, so a stale entry would solve silently against
outdated data or outdated equations — no exception, just a wrong answer. These tests
pin the invalidation contract, which is the only thing making the cache safe to enable.
"""

import os

import pytest

from equilibria.blocks.gtap import model_cache


class _Closure:
    """Minimal stand-in carrying the fields the key reads."""

    def __init__(self, **kw):
        self.name = kw.get("name", "base")
        self.closure_type = kw.get("closure_type", "MCP")
        self.savf_flag = kw.get("savf_flag", "capFix")
        self.if_sub = kw.get("if_sub", True)
        self.capital_mobility = kw.get("capital_mobility", "sluggish")
        self.numeraire = kw.get("numeraire", "pnum")
        self.fix_endowments = kw.get("fix_endowments", False)
        self.fix_taxes = kw.get("fix_taxes", False)
        self.fix_technology = kw.get("fix_technology", False)


@pytest.fixture
def dataset(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    for n in ("basedata.har", "sets.har", "default.prm", "baserate.har"):
        (d / n).write_bytes(b"x" * 16)
    return d


def _key(dataset, closure=None, **kw):
    return model_cache.cache_key(
        kw.pop("dataset_id", "gtap7_20x41"),
        dataset,
        closure or _Closure(),
        kw.pop("residual_region", "ROW"),
        kw.pop("base_calibrated", True),
        **kw,
    )


# ── The cache is opt-in ─────────────────────────────────────────────────────────────


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("EQUILIBRIA_GTAP_MODEL_CACHE", raising=False)
    assert model_cache.enabled() is False


def test_enabled_only_by_explicit_flag(monkeypatch):
    monkeypatch.setenv("EQUILIBRIA_GTAP_MODEL_CACHE", "1")
    assert model_cache.enabled() is True
    monkeypatch.setenv("EQUILIBRIA_GTAP_MODEL_CACHE", "0")
    assert model_cache.enabled() is False


def test_load_returns_none_when_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("EQUILIBRIA_GTAP_MODEL_CACHE", raising=False)
    monkeypatch.setenv("EQUILIBRIA_GTAP_MODEL_CACHE_DIR", str(tmp_path))
    assert model_cache.load("model-whatever") is None


# ── Key stability: the same inputs must reuse the same entry ────────────────────────


def test_key_is_stable_across_calls(dataset):
    assert _key(dataset) == _key(dataset)


def test_key_has_readable_prefix(dataset):
    assert _key(dataset).startswith("model-")


# ── Key invalidation: every input that changes the model must change the key ────────


@pytest.mark.parametrize(
    "field,value",
    [
        ("closure_type", "NLP"),
        ("savf_flag", "capFlex"),
        ("if_sub", False),
        ("capital_mobility", "mobile"),
        ("numeraire", "pfact"),
        ("fix_endowments", True),
        ("fix_taxes", True),
        ("fix_technology", True),
        ("name", "shock"),
    ],
)
def test_closure_field_change_invalidates(dataset, field, value):
    assert _key(dataset) != _key(dataset, _Closure(**{field: value}))


def test_dataset_id_change_invalidates(dataset):
    assert _key(dataset) != _key(dataset, dataset_id="gtap7_3x3")


def test_residual_region_change_invalidates(dataset):
    assert _key(dataset) != _key(dataset, residual_region="USA")


def test_base_calibrated_change_invalidates(dataset):
    assert _key(dataset) != _key(dataset, base_calibrated=False)


def test_ref_gdx_presence_invalidates(dataset):
    assert _key(dataset) != _key(dataset, ref_gdx="/some/out.gdx")


def test_data_file_change_invalidates(dataset):
    """A regenerated .har must not reuse a model built from the old one."""
    before = _key(dataset)
    f = dataset / "basedata.har"
    st = f.stat()
    f.write_bytes(b"y" * 32)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert _key(dataset) != before


def test_missing_data_file_invalidates(dataset):
    before = _key(dataset)
    (dataset / "sets.har").unlink()
    assert _key(dataset) != before


def test_code_change_invalidates(dataset, monkeypatch, tmp_path):
    """Editing an equation module must invalidate — this is what makes it safe to leave on.

    Points the digest at a throwaway tree so the real sources are never touched.
    """
    fake_root = tmp_path / "equilibria"
    for rel in model_cache._CODE_MODULES:
        p = fake_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("original\n")
    monkeypatch.setattr(model_cache, "_package_root", lambda: fake_root)

    before = _key(dataset)
    edited = fake_root / "templates/gtap/gtap_model_equations.py"
    edited.write_text("original\n# one changed equation\n")
    assert _key(dataset) != before, "a changed equation module must invalidate the key"


def test_every_listed_code_module_is_covered(dataset, monkeypatch, tmp_path):
    """Each module in _CODE_MODULES must actually move the key when edited."""
    fake_root = tmp_path / "equilibria"
    for rel in model_cache._CODE_MODULES:
        p = fake_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("original\n")
    monkeypatch.setattr(model_cache, "_package_root", lambda: fake_root)

    for rel in model_cache._CODE_MODULES:
        base = _key(dataset)
        f = fake_root / rel
        f.write_text("edited\n")
        assert _key(dataset) != base, f"{rel} does not affect the key"
        f.write_text("original\n")


def test_listed_code_modules_exist_in_the_real_tree():
    """Guard against a typo or a renamed module silently dropping out of the key."""
    root = model_cache._package_root()
    missing = [rel for rel in model_cache._CODE_MODULES if not (root / rel).exists()]
    assert not missing, f"listed in _CODE_MODULES but absent: {missing}"


# ── Round-trip and failure modes ────────────────────────────────────────────────────


def test_save_and_load_round_trip(monkeypatch, tmp_path):
    pytest.importorskip("cloudpickle")
    monkeypatch.setenv("EQUILIBRIA_GTAP_MODEL_CACHE", "1")
    monkeypatch.setenv("EQUILIBRIA_GTAP_MODEL_CACHE_DIR", str(tmp_path))

    import pyomo.environ as pyo

    def mk_init(scale):
        def _init(m, i):  # the nested closure stdlib pickle chokes on
            return scale * i

        return _init

    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=range(20))
    m.x = pyo.Var(m.I, initialize=mk_init(3.0))
    m._settled_seed = {"x": {0: 1.0}}

    model_cache.save("model-rt", m)
    back = model_cache.load("model-rt")
    assert back is not None
    assert len(back.x) == 20
    assert pyo.value(back.x[4]) == pyo.value(m.x[4])
    assert back._settled_seed == {"x": {0: 1.0}}


def test_load_of_corrupt_file_returns_none_not_raises(monkeypatch, tmp_path):
    """A truncated cache file degrades to a rebuild; it must never fail the run."""
    monkeypatch.setenv("EQUILIBRIA_GTAP_MODEL_CACHE", "1")
    monkeypatch.setenv("EQUILIBRIA_GTAP_MODEL_CACHE_DIR", str(tmp_path))
    (tmp_path / "model-bad.pkl").write_bytes(b"not a pickle at all")
    assert model_cache.load("model-bad") is None


def test_save_leaves_no_temp_file_behind(monkeypatch, tmp_path):
    pytest.importorskip("cloudpickle")
    monkeypatch.setenv("EQUILIBRIA_GTAP_MODEL_CACHE", "1")
    monkeypatch.setenv("EQUILIBRIA_GTAP_MODEL_CACHE_DIR", str(tmp_path))

    import pyomo.environ as pyo

    m = pyo.ConcreteModel()
    m.y = pyo.Var(initialize=1.0)
    model_cache.save("model-tmp", m)
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name]
    assert not leftovers, f"temp files left behind: {leftovers}"


def test_save_is_noop_when_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("EQUILIBRIA_GTAP_MODEL_CACHE", raising=False)
    monkeypatch.setenv("EQUILIBRIA_GTAP_MODEL_CACHE_DIR", str(tmp_path))

    import pyomo.environ as pyo

    m = pyo.ConcreteModel()
    m.y = pyo.Var(initialize=1.0)
    model_cache.save("model-off", m)
    assert not list(tmp_path.glob("*.pkl"))
