"""Executable cache: F and J through serialized executables equal the jit path bit for bit."""

import warnings

import numpy as np
import pytest

pytest.importorskip("jax")

from tests.family._data import load_params  # noqa: E402
from tests.family._oracle import comp_stat_closure  # noqa: E402


def _system(dataset):
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.templates.gtap.gtap_block_model import compose_block_state

    p = load_params(dataset)
    model, state = compose_block_state(
        p, p.sets, comp_stat_closure(True), list(p.sets.r)[-1]
    )
    b = FamilyBackend()
    b.build(model, skip=state.ifsub_skipped)
    layout = b.layout(state)
    z = layout.free_values * (
        1.0 + 0.05 * np.random.default_rng(0).uniform(-1.0, 1.0, layout.n_cols)
    )
    return b.system(layout), z


@pytest.fixture
def aot(tmp_path):
    from equilibria.family import aot_cache

    aot_cache.set_dir(None)
    aot_cache.reset()
    yield aot_cache, tmp_path / "jax_aot"
    aot_cache.set_dir(None)
    aot_cache.reset()


def _same(a, b):
    assert np.array_equal(a.row, b.row) and np.array_equal(a.col, b.col)
    assert np.array_equal(a.data, b.data)


@pytest.mark.parametrize("dataset", ["gtap7_3x3", "gtap7_10x7"])
def test_bitwise_equal_to_jit_and_round_trip(aot, dataset):
    from equilibria.family import recorder

    cache, root = aot
    recorder.reset_fold_stats()
    sysm, z = _system(dataset)
    # the fixed-zero folding ran: the bitwise equality below covers folded terms
    assert recorder.fold_stats["folded"] > 0
    F0, J0 = sysm.F(z), sysm.jac(z)  # jit path: no directory set

    cache.set_dir(root)
    F1, J1 = sysm.F(z), sysm.jac(z)  # miss -> compile + write
    assert cache.stats["miss"] > 0 and cache.stats["hit"] == 0
    assert list(root.glob("*.aot"))
    assert np.array_equal(F0, F1)
    _same(J0, J1)

    cache.reset()  # a new process: nothing in memory
    F2, J2 = sysm.F(z), sysm.jac(z)  # hit from disk
    assert cache.stats["hit"] > 0 and cache.stats["miss"] == 0
    assert cache.stats["mem"] > 0  # repeated calls in a run hit memory
    assert np.array_equal(F0, F2)
    _same(J0, J2)
    n_mem = cache.stats["mem"]
    sysm.F(z)  # same process again: served from memory
    assert cache.stats["mem"] > n_mem
    # all three kernels went through the cache
    kinds = {p.name.split("-")[0] for p in root.glob("*.aot")}
    assert kinds == {"eval", "jac", "scatter"}


def test_corrupt_entry_recompiles_with_warning(aot):
    cache, root = aot
    sysm, z = _system("gtap7_3x3")
    F0 = sysm.F(z)
    cache.set_dir(root)
    sysm.F(z)
    files = sorted(root.glob("*.aot"))
    for i, f in enumerate(files):
        data = bytearray(f.read_bytes())
        if i % 2:
            f.write_bytes(bytes(data[: len(data) // 2]))  # truncated
        else:
            data[-5] ^= 0xFF  # flipped byte, checksum must catch it
            f.write_bytes(bytes(data))
    cache.reset()
    with pytest.warns(UserWarning, match="executable cache"):
        F1 = sysm.F(z)
    assert np.array_equal(F0, F1)
    assert cache.stats["corrupt"] == len(files)
    # the entries were rewritten: the next fresh process hits
    cache.reset()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        F2 = sysm.F(z)
    assert np.array_equal(F0, F2) and cache.stats["miss"] == 0


def test_version_or_platform_change_misses(aot, monkeypatch):
    cache, root = aot
    sysm, z = _system("gtap7_3x3")
    cache.set_dir(root)
    sysm.F(z)
    n = len(list(root.glob("*.aot")))
    cache.reset()
    sysm.F(z)
    assert cache.stats["miss"] == 0
    for field, value in (("jax", "0.0.0"), ("jaxlib", "0.0.0"), ("platform", "tpu")):
        cache.reset()
        cache.environment()  # fills the memoized host part
        monkeypatch.setitem(cache._env, field, value)
        sysm.F(z)
        assert cache.stats["hit"] == 0 and cache.stats["miss"] > 0, field
        monkeypatch.undo()
    assert len(list(root.glob("*.aot"))) == 4 * n


def test_function_without_template_key_uses_plain_jit(aot):
    cache, root = aot
    from tests.family.test_core import _toy, _z

    cache.set_dir(root)
    _toy().F(_z())
    assert not list(root.glob("eval-*")) and not list(root.glob("jac-*"))


_SCRIPT = """
import sys, numpy as np
import jax
from tests.family.test_aot_cache import _system
from equilibria.family import aot_cache
mode, root = sys.argv[1], sys.argv[2]
jax.config.update("jax_compilation_cache_dir", root + "/jax")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
aot_cache.set_dir(root + "/aot")
sysm, z = _system("gtap7_3x3")
F, J = sysm.F(z), sysm.jac(z)
np.save(f"{root}/{mode}.npy", np.concatenate([F, J.data]))
print(aot_cache.stats)
"""


def test_entries_run_in_a_new_process_after_the_xla_cache_filled(tmp_path):
    """An executable taken from JAX's persistent cache serializes into one that cannot
    run in another process; entries are compiled fresh, so a new process still hits."""
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    env = {**__import__("os").environ, "PYTHONPATH": f"{root / 'src'}:{root}"}

    def run(mode):
        p = subprocess.run(
            [sys.executable, "-c", _SCRIPT, mode, str(tmp_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=root,
        )
        assert p.returncode == 0, p.stderr[-2000:]
        assert "UserWarning" not in p.stderr, p.stderr[-2000:]
        return p.stdout

    run("a")  # fills the XLA cache and the executable cache
    shutil.rmtree(tmp_path / "aot")
    out = run("b")  # the XLA cache hits; the entries must still be written fresh
    assert "'miss': 0" not in out
    out = run("c")  # a new process reads them
    assert "'miss': 0" in out and "'corrupt': 0" in out
    a, b, c = (np.load(tmp_path / f"{m}.npy") for m in "abc")
    assert np.array_equal(a, b) and np.array_equal(a, c)


def test_environment_key_covers_xla_flags_config_and_cpu(aot, monkeypatch):
    cache, _ = aot
    cache._env = None
    base = cache.environment()
    assert isinstance(base["xla_flags"], str)
    assert isinstance(base["cpu"], str)
    assert "machine" in base and "processor" in base
    for name in ("jax_default_matmul_precision", "jax_numpy_dtype_promotion"):
        assert name in base
    name0 = cache._name("eval", ("s",))
    monkeypatch.setenv("XLA_FLAGS", "--xla_cpu_enable_fast_math=true")
    cache._env = None
    try:
        assert cache.environment()["xla_flags"] == "--xla_cpu_enable_fast_math=true"
        assert cache._name("eval", ("s",)) != name0
    finally:
        cache._env = None


def test_compile_fresh_without_jax_internals_warns_once_and_disables(aot, monkeypatch):
    import types

    import jax
    import jax._src
    import jax.numpy as jnp

    cache, root = aot
    cache.set_dir(root)
    # the module as a future JAX might shape it: without the two latch names
    monkeypatch.setattr(jax._src, "compilation_cache", types.SimpleNamespace())
    f = jax.jit(lambda x: x + 1.0)
    x = jnp.ones(3)
    with pytest.warns(UserWarning, match="executable cache is disabled") as rec:
        out = cache._compile_fresh(lambda: f.lower(x))
    assert (
        len([w for w in rec if "executable cache is disabled" in str(w.message)]) == 1
    )
    assert np.array_equal(out(x), np.full(3, 2.0))
    assert cache.get_dir() is None
    with warnings.catch_warnings(record=True) as rec2:
        warnings.simplefilter("always")
        cache._compile_fresh(lambda: f.lower(x))
    assert not [w for w in rec2 if "executable cache is disabled" in str(w.message)]


def test_fallback_through_the_real_path_warns_once_and_runs(aot, monkeypatch):
    import types

    import jax
    import jax._src
    import jax.numpy as jnp

    cache, root = aot
    cache.set_dir(root)
    monkeypatch.setattr(jax._src, "compilation_cache", types.SimpleNamespace())
    jitted = jax.jit(lambda v, r, n: jnp.zeros(n).at[r].add(v), static_argnums=2)
    v, r = jnp.arange(3.0), jnp.array([0, 1, 1])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = cache.run_scatter(jitted, v, r, 2)
        again = cache.run_scatter(jitted, v, r, 2)  # plain jit now
    assert len(rec) == 1 and "disabled" in str(rec[0].message), [
        str(w.message) for w in rec
    ]
    assert np.array_equal(out, [0.0, 3.0]) and np.array_equal(again, out)
    assert cache.get_dir() is None


def test_environment_follows_a_later_jax_config_update(aot):
    import jax

    cache, _ = aot
    before = jax.config.jax_default_matmul_precision
    env0, name0 = cache.environment(), cache._name("f", ("spec",))
    try:
        jax.config.update("jax_default_matmul_precision", "highest")
        assert cache.environment() != env0  # no reset() in between
        assert cache._name("f", ("spec",)) != name0
    finally:
        jax.config.update("jax_default_matmul_precision", before)
    assert cache._name("f", ("spec",)) == name0
