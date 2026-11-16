"""On-disk cache of the recorded families and of JAX's compiled programs.

The records depend on the params, the closure, the residual region and the code
that writes the equations; the key (``blocks/gtap/fingerprint``) hashes all four.
The composed state (levels, fixed cells, Parameters) is kept next to them under the
same key: it is a function of the same inputs, and composing it does not touch the
params.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

_FAMILY_DIR = Path(__file__).resolve().parent


def _family_code_digest() -> str:
    """The recorder itself: ``fingerprint`` covers the blocks, templates, backends,
    core and contracts, not ``equilibria/family``."""
    h = hashlib.sha256()
    for f in sorted(_FAMILY_DIR.rglob("*.py")):
        h.update(f.relative_to(_FAMILY_DIR).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:24]


def cache_key(params: Any, closure: Any, residual_region: str | None) -> str | None:
    """Key over the params CONTENT, the closure, the residual region and the code,
    or ``None`` (do not cache) when that cannot be covered or @overwrite hooks are
    active (they change the equations and are not in the key)."""
    import jax

    from equilibria.blocks.gtap.fingerprint import fingerprint
    from equilibria.blocks.gtap.overwrite import overwrite

    if overwrite.active():
        return None
    return fingerprint(
        "family-records",
        params=params,
        closure=closure,
        residual_region=str(residual_region),
        family_code=_family_code_digest(),
        jax=jax.__version__,
    )


def load_or_build(
    params: Any,
    closure: Any,
    residual_region: str | None,
    root: Path,
    *,
    key: Any = ...,
) -> tuple[Any, Any]:
    """``(backend, state)``: records from ``root/<key>/records.pkl`` and the composed
    state from ``root/<key>/state.pkl`` when present.

    Without a key (see ``cache_key``) both are built and not saved. ``key``: the
    caller's ``cache_key`` of these inputs (``None`` included), so it is not computed
    twice. The Model of the blocks is only composed when something is missing."""
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.family.phases import phase
    from equilibria.templates.gtap.gtap_block_model import compose_block_state

    if key is ...:
        with phase("sp_cache_key"):
            key = cache_key(params, closure, residual_region)
    folder = None if key is None else Path(root) / key
    path = None if folder is None else folder / "records.pkl"
    state_path = None if folder is None else folder / "state.pkl"
    state = None
    if state_path is not None and state_path.exists():
        with phase("state_load"):
            state = _load_pickle(state_path)
    model = None
    if state is None:
        with phase("compose_state"):
            model, state = compose_block_state(
                params, params.sets, closure, residual_region
            )
        if state_path is not None:
            _save_pickle(state, state_path)
    if path is not None and path.exists():
        with phase("sp_records_load"):
            return FamilyBackend.load(path), state
    if model is None:  # records gone, state kept: the recording needs the Model
        with phase("compose_state"):
            model, state = compose_block_state(
                params, params.sets, closure, residual_region
            )
    with phase("sp_records_build"):
        backend = FamilyBackend()
        backend.build(model, skip=state.ifsub_skipped)
        if path is not None:
            backend.save(path)
    return backend, state


def _save_pickle(obj: Any, path: Path) -> None:
    """Pickle ``obj`` to ``path`` through a temp file of our own (atomic replace)."""
    import os
    import pickle
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(obj, fh, protocol=5)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _load_pickle(path: Path) -> Any:
    """The pickled object, or ``None`` when the file cannot be read (then rebuilt)."""
    import pickle

    from equilibria.solver.gc_tuning import _gc_freeze_in_scope, _gc_paused

    try:
        with open(path, "rb") as fh, _gc_paused():
            obj = pickle.load(fh)
            _gc_freeze_in_scope()
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError):
        return None
    return obj


def enable_jax_cache(root: Path) -> None:
    """Persist every compiled family program under ``root/jax`` and the serialized
    executables of the F/J kernels under ``root/jax_aot`` (no trace on a warm start).

    Process-wide JAX config; call it before the first compile, or programs already
    compiled in this process are not cached."""
    import jax

    jax.config.update("jax_compilation_cache_dir", str(Path(root) / "jax"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    from equilibria.family import aot_cache

    aot_cache.set_dir(Path(root) / "jax_aot")
