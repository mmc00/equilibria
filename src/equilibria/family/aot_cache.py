"""Serialized XLA executables of the family kernels: a warm start without a JAX trace.

JAX's persistent compilation cache skips the XLA compile but not the Python trace nor
the lowering to MLIR, which it needs to compute its key. Here the compiled executable
itself is stored (``jax.experimental.serialize_executable``) under a key made of the
template shape, the argument shapes and dtypes, the JAX / jaxlib versions, the platform,
the host (machine, processor, CPU brand), ``XLA_FLAGS``, the compilation-affecting JAX
config and the code that writes the kernels, so a hit runs the program without tracing it.

Never wrong, never fatal: an entry that is missing, from another environment (the key
changes), truncated, altered (checksum) or that fails to load is replaced by a fresh
``lower().compile()`` of the very same jitted function, with a warning for the damaged
ones. The executables are loaded with pickle: the directory is as trusted as the code.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import platform
import subprocess
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax

_MAGIC = b"EQAOT1\n"
_FAMILY_DIR = Path(__file__).resolve().parent

_dir: Path | None = None
_mem: dict[str, Any] = {}
_verified: set[str] = set()  # executables that ran once in this process
_env: dict[str, Any] | None = None
_warned_internals = False  # the "JAX internals changed" warning, once per process
stats: dict[str, int] = {"hit": 0, "miss": 0, "corrupt": 0, "mem": 0}


def set_dir(path: Path | str | None) -> None:
    """Enable the cache under ``path`` (created on the first write) or disable it."""
    global _dir
    _dir = None if path is None else Path(path)


def get_dir() -> Path | None:
    return _dir


def reset() -> None:
    """Forget the executables held in memory and zero the counters (a fresh process)."""
    _mem.clear()
    _verified.clear()
    global _warned_internals
    _warned_internals = False
    for k in stats:
        stats[k] = 0


_JAX_CONFIG = ("jax_default_matmul_precision", "jax_numpy_dtype_promotion")


def _cpu_brand() -> str:
    """The CPU model string of this host ("" when it cannot be read)."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            return out.stdout.strip()
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return ""


def environment() -> dict[str, Any]:
    """What a serialized executable depends on besides its program (part of every key)."""
    global _env
    if _env is None:
        import jaxlib

        dev = jax.devices()[0]
        h = hashlib.sha256()
        for name in ("recorder.py", "core.py", "aot_cache.py"):
            h.update((_FAMILY_DIR / name).read_bytes())
        _env = {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "platform": dev.client.platform,
            "platform_version": dev.client.platform_version,
            "device_kind": dev.device_kind,
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu": _cpu_brand() if dev.client.platform == "cpu" else "",
            "code": h.hexdigest()[:24],
        }
    # JAX config can change after the host part was memoized: read it every time
    live = {
        "x64": bool(jax.config.jax_enable_x64),
        **{n: repr(getattr(jax.config, n, None)) for n in _JAX_CONFIG},
    }
    return {**_env, **live}


def _sig(a: Any) -> tuple:
    return (tuple(a.shape), str(a.dtype))


def _name(kind: str, spec: tuple) -> str:
    desc = repr((spec, sorted(environment().items())))
    return f"{kind}-{hashlib.sha256(desc.encode()).hexdigest()[:32]}.aot"


def _read(path: Path) -> Any:
    """The compiled executable stored at ``path``; raises on any damage."""
    from jax.experimental.serialize_executable import deserialize_and_load

    raw = path.read_bytes()
    n = len(_MAGIC)
    if raw[:n] != _MAGIC or len(raw) < n + 32:
        raise ValueError("bad header")
    digest, payload = raw[n : n + 32], raw[n + 32 :]
    if hashlib.sha256(payload).digest() != digest:
        raise ValueError("checksum mismatch")
    blob, in_tree, out_tree = pickle.loads(payload)
    return deserialize_and_load(blob, in_tree, out_tree)


def _write(path: Path, compiled: Any) -> None:
    from jax.experimental.serialize_executable import serialize

    payload = pickle.dumps(serialize(compiled), protocol=pickle.HIGHEST_PROTOCOL)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(_MAGIC + hashlib.sha256(payload).digest() + payload)
    os.replace(tmp, path)


def _compile_fresh(lower: Callable[[], Any]) -> Any:
    """``lower().compile()`` that neither reads nor writes JAX's persistent cache.

    An executable that came out of that cache serializes into one that fails to run
    in another process (``Function ... not found``), so entries are always compiled.
    JAX latches "cache in use" after the first compile, so the flag is switched
    in its module (private names, guarded: if they change, the first-run check in
    ``_call`` still catches an entry that cannot run)."""
    global _warned_internals
    saved: tuple[Any, Any] | None = None
    cc: Any = None
    fallback = False
    try:
        from jax._src import compilation_cache as cc

        saved = (cc._cache_checked, cc._cache_used)
        cc._cache_checked, cc._cache_used = True, False
    except Exception:
        saved = None
        fallback = True
    try:
        return lower().compile()
    finally:
        if saved is not None:
            cc._cache_checked, cc._cache_used = saved
        if fallback:
            # JAX's persistent cache may have served this one: it could fail in another
            # process, so stop storing executables and let later calls take the jit path
            if not _warned_internals:
                _warned_internals = True
                warnings.warn(
                    "executable cache is disabled: JAX's internals changed and its "
                    "persistent cache cannot be bypassed",
                    stacklevel=6,
                )
            set_dir(None)


def _build(name: str, lower: Callable[[], Any]) -> Any:
    assert _dir is not None
    path = _dir / name
    compiled = _compile_fresh(lower)
    if _dir is not None:  # the compile may have disabled the cache (JAX internals)
        try:
            _write(path, compiled)
        except Exception as exc:  # unserializable or disk full: keep going without it
            warnings.warn(
                f"executable cache could not write {name} ({exc!r})", stacklevel=4
            )
    _mem[name] = compiled
    _verified.add(name)
    return compiled


def _obtain(name: str, lower: Callable[[], Any]) -> Any:
    """The executable of entry ``name``: from memory, from disk, or freshly compiled."""
    cached = _mem.get(name)
    if cached is not None:
        stats["mem"] += 1
        return cached
    assert _dir is not None
    path = _dir / name
    if path.exists():
        try:
            loaded = _read(path)
        except Exception as exc:
            _damaged(path, exc)
        else:
            stats["hit"] += 1
            _mem[name] = loaded
            return loaded
    stats["miss"] += 1
    return _build(name, lower)


def _call(kind: str, spec: tuple, lower: Callable[[], Any], *dyn: Any) -> Any:
    """Run the executable of ``spec`` on the dynamic arguments ``dyn``."""
    name = _name(kind, spec)
    compiled = _obtain(name, lower)
    path = None if _dir is None else _dir / name
    if name in _verified:
        return compiled(*dyn)
    try:  # the first run of a loaded executable: a program that loads may still not run
        out = compiled(*dyn)
        out.block_until_ready()  # a dispatch error can surface asynchronously
    except Exception as exc:
        _damaged(path or Path(name), exc)
        if _dir is None:  # disabled meanwhile: no entry to rebuild
            _mem.pop(name, None)  # do not serve the bad executable again
            return lower().compile()(*dyn)
        return _build(name, lower)(*dyn)
    _verified.add(name)
    return out


def _damaged(path: Path, exc: Exception) -> None:
    stats["corrupt"] += 1
    warnings.warn(
        f"executable cache entry {path.name} unusable ({exc!r}); recompiling",
        stacklevel=4,
    )


def run_part(
    kind: str, jitted: Any, fn: Callable, x: Any, cols: Any, args: dict
) -> Any:
    """``jitted(fn, x, cols, args)`` through a stored executable when ``fn`` is a
    template with a stable key (``fn._aot_key``); the plain jit call otherwise."""
    tkey = getattr(fn, "_aot_key", None)
    if _dir is None or tkey is None:
        return jitted(fn, x, cols, args)
    spec = (
        kind,
        tkey,
        _sig(x),
        _sig(cols),
        tuple((k, _sig(v)) for k, v in sorted(args.items())),
    )
    return _call(kind, spec, lambda: jitted.lower(fn, x, cols, args), x, cols, args)


def run_scatter(jitted: Any, values: Any, rows: Any, num_rows: int) -> Any:
    """``jitted(values, rows, num_rows)`` through a stored executable."""
    if _dir is None:
        return jitted(values, rows, num_rows)
    spec = ("scatter", _sig(values), _sig(rows), int(num_rows))
    lower = lambda: jitted.lower(values, rows, num_rows)  # noqa: E731
    return _call("scatter", spec, lower, values, rows)
