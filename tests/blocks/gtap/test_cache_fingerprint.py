"""Tests for the one cache key shared by the model cache and the seed cache.

Both caches used to build their own key by hand, so a stale-cache fix landed in
one and not the other (b15f9b9 / b5b191e).  The contract pinned here: ANY change
to the benchmark data, the closure, the flags or the code folders changes the
key, and a key that cannot cover all of that is ``None`` (= skip the cache).
"""

from types import SimpleNamespace

import pytest

from equilibria.blocks.gtap import fingerprint


class _Closure:
    def __init__(self, **kw):
        self._d = {"name": "base", "if_sub": True, "va_subsidy_basis": "gtap", **kw}

    def model_dump(self):
        return dict(self._d)


def _params(**over):
    bm = {
        "evfb": {("USA", "Land", "Food"): 1.0},
        "vfm": {("USA", "Land", "Food"): 2.0},
        "vkb": {"USA": 3.0},
        "vdfb": {("USA", "Food"): 4.0},
        "vmfb": {("USA", "Food"): 5.0},
        "vst": {("USA", "Food"): 6.0},
    }
    tx = {
        "rtf": {("USA", "Land", "Food"): 0.1},
        "kappaf_activity": {("USA", "Land", "Food"): 0.2},
    }
    for k, v in over.items():
        (bm if k in bm else tx)[k] = v
    return SimpleNamespace(
        benchmark=SimpleNamespace(**bm),
        taxes=SimpleNamespace(**tx),
        elasticities=SimpleNamespace(esubva={("USA", "SER"): 1.26}),
        shifts=SimpleNamespace(lambdava={}),
    )


def _fp(params=None, closure=None, namespace="model", **flags):
    return fingerprint.fingerprint(
        namespace,
        params=params or _params(),
        closure=closure or _Closure(),
        residual_region="ROW",
        **flags,
    )


def test_same_inputs_same_key():
    assert _fp() == _fp()


@pytest.mark.parametrize(
    "group", ["evfb", "vfm", "vkb", "vdfb", "vmfb", "vst", "rtf", "kappaf_activity"]
)
def test_every_benchmark_group_moves_the_key(group):
    """vdfb/vmfb/vst were missing from the seed key."""
    assert _fp(_params(**{group: {("X",): 9.9}})) != _fp()


@pytest.mark.parametrize(
    "group,attr",
    [
        ("taxes", "rtms"),  # two runs differing only in tariffs shared a model
        ("taxes", "rtxs"),
        ("benchmark", "vom"),
        ("benchmark", "makb"),
        ("shares", "p_va"),
        ("calibrated", "and_param"),
    ],
)
def test_any_params_array_moves_the_key(group, attr):
    """Every array on params counts, not a hand-picked list of 8."""
    p = _params()
    if getattr(p, group, None) is None:
        setattr(p, group, SimpleNamespace())
    setattr(getattr(p, group), attr, {("USA",): 0.5})
    assert _fp(p) != _fp()


def test_nested_objects_and_sets_are_covered():
    p = _params()
    p.shares = SimpleNamespace(normalized=SimpleNamespace(value_added_share={"x": 1.0}))
    q = _params()
    q.shares = SimpleNamespace(normalized=SimpleNamespace(value_added_share={"x": 2.0}))
    assert _fp(p) != _fp(q)
    p.sets = SimpleNamespace(r=["USA", "EU"], i_to_a={"Food": "Food"})
    q2 = _params()
    q2.shares = p.shares
    q2.sets = SimpleNamespace(r=["USA", "ROW"], i_to_a={"Food": "Food"})
    assert _fp(p) != _fp(q2), "string data outside the numeric groups is covered"


def test_flags_written_onto_params_are_covered():
    """build_block_model writes va_subsidy_basis / _capflex_risk onto params."""
    p = _params()
    p.va_subsidy_basis = "gams"
    q = _params()
    q.va_subsidy_basis = "gempack"
    assert _fp(p) != _fp(q)


def test_elasticities_and_shifts_move_the_key():
    p = _params()
    p.shifts.lambdava[("USA", "SER")] = 1.10
    assert _fp(p) != _fp()
    q = _params()
    q.elasticities.esubva[("USA", "SER")] = 1.0
    assert _fp(q) != _fp()


def test_every_closure_field_moves_the_key():
    """The seed key kept 5 hand-picked fields and missed va_subsidy_basis."""
    assert _fp(closure=_Closure(va_subsidy_basis="other")) != _fp()
    assert _fp(closure=_Closure(new_field=1)) != _fp()


def test_residual_region_flags_and_namespace_move_the_key():
    assert _fp(base_calibrated=True) != _fp(base_calibrated=False)
    assert _fp(namespace="seed") != _fp(namespace="model")
    k = fingerprint.fingerprint(
        "model", params=_params(), closure=_Closure(), residual_region="EU"
    )
    assert k != _fp()


def test_key_starts_with_namespace():
    assert _fp(namespace="seed2").startswith("seed2-")


def test_uncoverable_inputs_give_none():
    assert _fp(closure=object()) is None, "closure without model_dump"
    assert (
        fingerprint.fingerprint(
            "model",
            params=SimpleNamespace(benchmark=None, taxes=None),
            closure=_Closure(),
            residual_region="ROW",
        )
        is None
    )
    assert _fp(_params(evfb={("USA",): object()})) is None, "unreadable value"
    assert _fp(_params(evfb={("USA",): "x"})) is None, "string in a numeric group"
    empty = _params()
    for name in vars(empty.benchmark):
        setattr(empty.benchmark, name, {})
    assert _fp(empty) is None, "an all-empty benchmark covers no data"
    weird = _params()
    weird.extra = object()  # default repr carries a memory address
    assert _fp(weird) is None


def test_code_folders_cover_build_and_solve():
    """The seed comes out of a full settle solve, so solver code shapes it too."""
    for rel in ("blocks/gtap", "templates/gtap", "solver", "backends", "core"):
        assert rel in fingerprint.CODE_FOLDERS
    # Imported by the build from outside those folders (PR #101 review).
    for rel in ("contracts", "babel/gdx"):
        assert rel in fingerprint.CODE_FOLDERS
    assert "model.py" in fingerprint.CODE_FILES


def test_any_py_file_in_the_code_folders_moves_the_key(monkeypatch, tmp_path):
    root = tmp_path / "equilibria"
    files = [
        "blocks/gtap/floors.py",  # missing from the old hand list
        "templates/gtap/gtap_multiperiod_driver.py",
        "templates/gtap/altertax/outer_loop.py",  # nested folder
        "solver/path_capi.py",
        "backends/pyomo_backend.py",
        "core/variables.py",
        "blocks/base.py",
        "model.py",
        "contracts/base.py",
    ]
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("original\n")
    monkeypatch.setattr(fingerprint, "_package_root", lambda: root)

    for rel in files:
        before = _fp()
        (root / rel).write_text("edited\n")
        assert _fp() != before, f"{rel} does not affect the key"
        (root / rel).write_text("original\n")

    before = _fp()
    new = root / "solver" / "new_module.py"
    new.write_text("x = 1\n")
    assert _fp() != before, "a new module in a code folder must move the key"


def test_non_python_files_do_not_move_the_key(monkeypatch, tmp_path):
    root = tmp_path / "equilibria"
    (root / "solver").mkdir(parents=True)
    (root / "solver" / "path_capi.py").write_text("x\n")
    monkeypatch.setattr(fingerprint, "_package_root", lambda: root)
    before = _fp()
    (root / "solver" / "__pycache__").mkdir()
    (root / "solver" / "__pycache__" / "path_capi.cpython-312.pyc").write_bytes(b"\0")
    assert _fp() == before


def test_real_tree_has_every_code_path():
    root = fingerprint._package_root()
    paths = fingerprint.CODE_FOLDERS + fingerprint.CODE_FILES
    missing = [rel for rel in paths if not (root / rel).exists()]
    assert not missing, f"CODE_FOLDERS lists absent paths: {missing}"
