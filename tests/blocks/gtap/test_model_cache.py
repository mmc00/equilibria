"""Tests for the built-model disk cache (src/equilibria/blocks/gtap/model_cache.py).

The cache stores a whole ConcreteModel, so a stale entry would solve silently against
outdated data or outdated equations — no exception, just a wrong answer. These tests
pin the invalidation contract, which is the only thing making the cache safe to enable.
"""

import pathlib
from types import SimpleNamespace

import pytest

from equilibria.blocks.gtap import model_cache


class _Closure:
    """Stand-in exposing ``model_dump`` the way GTAPClosureConfig does."""

    def __init__(self, **kw):
        self._d = {
            "name": "base",
            "closure_type": "MCP",
            "savf_flag": "capFix",
            "if_sub": True,
            "capital_mobility": "sluggish",
            "numeraire": "pnum",
            "fix_endowments": False,
            "fix_taxes": False,
            "fix_technology": False,
            "va_subsidy_basis": "gtap",
            **kw,
        }

    def model_dump(self):
        return dict(self._d)


class _Params:
    """Stand-in carrying the benchmark arrays the key digests."""

    def __init__(self, evfb=None, rtf=None):
        self.benchmark = SimpleNamespace(
            evfb=evfb if evfb is not None else {("USA", "Land", "Food"): 1.0},
            vfm={("USA", "Land", "Food"): 2.0},
            vkb={"USA": 3.0},
            vdfb={("USA", "Food"): 4.0},
            vmfb={("USA", "Food"): 5.0},
            vst={("USA", "Food"): 6.0},
        )
        self.taxes = SimpleNamespace(
            rtf=rtf if rtf is not None else {("USA", "Land", "Food"): 0.1},
            kappaf_activity={("USA", "Land", "Food"): 0.2},
        )


@pytest.fixture
def params():
    return _Params()


def _key(params, closure=None, **kw):
    return model_cache.cache_key(
        params,
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


# ── Key stability: the same inputs must reuse the same entry ────────────────────


def test_key_is_stable_across_calls(params):
    assert _key(params) == _key(params)


def test_key_has_readable_prefix(params):
    assert _key(params).startswith("model-")


# ── The key REFUSES to exist when it cannot cover the inputs ────────────────────


def test_key_is_none_without_benchmark_content():
    """No reachable content -> no key -> no cache.

    The bug this pins: the earlier key digested ``params._source_paths`` and
    defaulted to ``{}``, so a loader that never set it (load_from_gdx) produced
    the sha256 of the empty string -- a valid, STABLE key covering no data at all.
    Two datasets collided and the second silently solved against the first's model.
    """
    empty = SimpleNamespace(benchmark=None, taxes=None)
    assert model_cache.cache_key(empty, _Closure(), "ROW", True) is None


def test_key_is_none_when_arrays_are_unreadable():
    bad = SimpleNamespace(
        benchmark=SimpleNamespace(evfb={("USA",): "not-a-number"}),
        taxes=None,
    )
    assert model_cache.cache_key(bad, _Closure(), "ROW", True) is None


def test_key_is_none_for_a_closure_without_model_dump(params):
    assert model_cache.cache_key(params, object(), "ROW", True) is None


# ── Key invalidation: every input that changes the model must change the key ────


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
        # Was MISSING from the old 9-field allowlist.  It picks ftrv+fbep vs
        # ftrv-fbep in _derived_params._va_wedge, moving the VA-vs-intermediates
        # weighting (0.571 vs 0.679 on subsidised agriculture) -- two closures
        # differing only here shared a key.
        ("va_subsidy_basis", "gempack"),
    ],
)
def test_closure_field_change_invalidates(params, field, value):
    assert _key(params) != _key(params, _Closure(**{field: value}))


def test_any_new_closure_field_invalidates(params):
    """The key digests model_dump(), so a field added later is covered on day one."""
    assert _key(params) != _key(params, _Closure(some_future_switch=True))


def test_residual_region_change_invalidates(params):
    assert _key(params) != _key(params, residual_region="USA")


def test_base_calibrated_change_invalidates(params):
    assert _key(params) != _key(params, base_calibrated=False)


def test_ref_gdx_presence_invalidates(params):
    assert _key(params) != _key(params, ref_gdx="/some/out.gdx")


# ── Data invalidation is by CONTENT, which is what path+mtime could not do ──────


def test_benchmark_content_change_invalidates(params):
    other = _Params(evfb={("USA", "Land", "Food"): 1.0000001})
    assert _key(params) != _key(other), "a changed benchmark array must invalidate"


def test_tax_content_change_invalidates(params):
    other = _Params(rtf={("USA", "Land", "Food"): 0.9})
    assert _key(params) != _key(other)


def test_same_content_from_a_different_file_shares_a_key(params):
    """Content-addressed: identical numbers reuse the entry no matter the path.

    The flip side of the scenario path+mtime got wrong -- a dataset restored by
    rsync/cp -p keeps its mtime while its content changes, and the old key could
    not tell.  Here the content IS the key.
    """
    assert _key(params) == _key(_Params())


def test_two_datasets_do_not_collide():
    """The GDX-loader collision, pinned.

    Neither params carries a file path at all; under the old path-based digest
    both produced the sha256 of the empty string and shared one cache entry.
    """
    a = _Params(evfb={("USA", "Land", "Food"): 1.0})
    b = _Params(evfb={("EU", "Land", "Food"): 7.5})
    assert _key(a) != _key(b), "two different datasets must not share a key"


def test_code_change_invalidates(params, monkeypatch, tmp_path):
    """Editing an equation module must invalidate — this is what makes it safe to leave on."""
    fake_root = tmp_path / "equilibria"
    for rel in model_cache._CODE_MODULES:
        p = fake_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("original\n")
    monkeypatch.setattr(model_cache, "_package_root", lambda: fake_root)

    before = _key(params)
    edited = fake_root / "templates/gtap/gtap_model_equations.py"
    edited.write_text("original\n# one changed equation\n")
    assert _key(params) != before, "a changed equation module must invalidate the key"


def test_every_listed_code_module_is_covered(params, monkeypatch, tmp_path):
    """Each module in _CODE_MODULES must actually move the key when edited."""
    fake_root = tmp_path / "equilibria"
    for rel in model_cache._CODE_MODULES:
        p = fake_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("original\n")
    monkeypatch.setattr(model_cache, "_package_root", lambda: fake_root)

    for rel in model_cache._CODE_MODULES:
        base = _key(params)
        f = fake_root / rel
        f.write_text("edited\n")
        assert _key(params) != base, f"{rel} does not affect the key"
        f.write_text("original\n")


def test_listed_code_modules_exist_in_the_real_tree():
    """Guard against a typo or a renamed module silently dropping out of the key."""
    root = model_cache._package_root()
    missing = [rel for rel in model_cache._CODE_MODULES if not (root / rel).exists()]
    assert not missing, f"listed in _CODE_MODULES but absent: {missing}"


def test_the_pyomo_translation_layer_is_in_the_key():
    """These decide the built model as directly as the blocks do.

    pyomo_backend.py sets every Var's bounds and domain; a policy change there
    changes the whole model without touching a block.  It was absent from the
    original list, so such a change served stale models in silence.
    """
    for rel in (
        "backends/pyomo_backend.py",
        "backends/pyomo_equations.py",
        "blocks/base.py",
        "core/symbolic_equations.py",
        "core/variables.py",
    ):
        assert rel in model_cache._CODE_MODULES, f"{rel} missing from _CODE_MODULES"


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


# ── Integration: the production path, which unit tests cannot reach ─────────────

_DS = pathlib.Path(__file__).resolve().parents[3] / "datasets" / "gtap7_3x3"


def _real_params():
    from equilibria.templates.gtap.gtap_parameters import GTAPParameters

    p = GTAPParameters()
    p.load_from_har(
        basedata_path=_DS / "basedata.har",
        sets_path=_DS / "sets.har",
        default_path=_DS / "default.prm",
        baserate_path=_DS / "baserate.har",
    )
    return p


def _real_closure():
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        savf_flag="capFix",
        numeraire="pnum",
    )


@pytest.mark.skipif(not _DS.exists(), reason="gtap7_3x3 dataset not present")
def test_real_params_and_closure_produce_a_usable_key():
    """The objects build_block_model actually passes must yield a key.

    Every unit test above uses stand-ins.  This is the one that would have caught
    the original bug: the key was built from ``params._source_paths``, an attribute
    no loader on main defines, so the real production call digested ``{}``.
    """
    key = model_cache.cache_key(_real_params(), _real_closure(), "ROW", True)
    assert key is not None, "real GTAP params/closure must produce a key"
    assert key.startswith("model-")


@pytest.mark.skipif(not _DS.exists(), reason="gtap7_3x3 dataset not present")
def test_real_datasets_do_not_share_a_key():
    """Two real datasets must never collide.

    Under the path-based key this failed for every loader that does not populate
    _source_paths: both sides digested the empty dict and shared one entry.
    """
    other = _DS.parent / "gtap7_5x5"
    if not other.exists():
        pytest.skip("gtap7_5x5 not present")
    from equilibria.templates.gtap.gtap_parameters import GTAPParameters

    p2 = GTAPParameters()
    p2.load_from_har(
        basedata_path=other / "basedata.har",
        sets_path=other / "sets.har",
        default_path=other / "default.prm",
        baserate_path=other / "baserate.har",
    )
    k1 = model_cache.cache_key(_real_params(), _real_closure(), "ROW", True)
    k2 = model_cache.cache_key(p2, _real_closure(), "ROW", True)
    assert k1 != k2, "two real datasets shared a cache key"


@pytest.mark.skipif(not _DS.exists(), reason="gtap7_3x3 dataset not present")
def test_build_block_model_serves_a_hit_and_never_a_stale_model(monkeypatch, tmp_path):
    """End-to-end through build_block_model: a hit is reused, changed data is not.

    This is the test whose absence let the whole suite pass green while the
    production key was degenerate -- the unit tests exercised a different branch
    from the one build_block_model calls.
    """
    pytest.importorskip("cloudpickle")
    monkeypatch.setenv("EQUILIBRIA_GTAP_MODEL_CACHE", "1")
    monkeypatch.setenv("EQUILIBRIA_GTAP_MODEL_CACHE_DIR", str(tmp_path))
    from equilibria.templates.gtap.gtap_block_model import build_block_model

    params, closure = _real_params(), _real_closure()
    from equilibria.templates.gtap.gtap_sets import GTAPSets  # noqa: F401

    m1, _ = build_block_model(params, params.sets, closure, "ROW")
    n1 = len(list(m1.component_objects()))
    assert list(tmp_path.glob("*.pkl")), "miss did not write a cache entry"

    m2, _ = build_block_model(params, params.sets, closure, "ROW")
    assert len(list(m2.component_objects())) == n1, "hit returned a different model"

    # Now change the DATA.  The key is content-addressed, so this must miss.
    key_before = model_cache.cache_key(params, closure, "ROW", True)
    k = next(iter(params.benchmark.evfb))
    params.benchmark.evfb[k] = float(params.benchmark.evfb[k]) * 1.5
    key_after = model_cache.cache_key(params, closure, "ROW", True)
    assert key_after != key_before, "changed benchmark data still hit the same entry"
