"""On-disk cache of the recorded families and of JAX's compiled programs.

The records depend on the params, the closure, the residual region and the code
that writes the equations; the key (``blocks/gtap/fingerprint``) hashes all four.
The state (this run's levels, fixed cells, Parameters) is always composed fresh.
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
    params: Any, closure: Any, residual_region: str | None, root: Path
) -> tuple[Any, Any]:
    """``(backend, state)``: records from ``root/<key>/records.pkl`` when present.

    Without a key (see ``cache_key``) the records are built and not saved."""
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.templates.gtap.gtap_block_model import compose_block_state

    model, state = compose_block_state(params, params.sets, closure, residual_region)
    key = cache_key(params, closure, residual_region)
    path = None if key is None else Path(root) / key / "records.pkl"
    if path is not None and path.exists():
        return FamilyBackend.load(path), state
    backend = FamilyBackend()
    backend.build(model, skip=state.ifsub_skipped)
    if path is not None:
        backend.save(path)
    return backend, state


def enable_jax_cache(root: Path) -> None:
    """Persist every compiled family program under ``root/jax``.

    Process-wide JAX config; call it before the first compile, or programs already
    compiled in this process are not cached."""
    import jax

    jax.config.update("jax_compilation_cache_dir", str(Path(root) / "jax"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
