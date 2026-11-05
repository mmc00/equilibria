"""On-disk cache of the recorded families and of JAX's compiled programs.

The records depend on the dataset, the closure, the residual region and the code
that writes the equations; the key hashes all four. The state (this run's levels,
fixed cells, Parameters) is always composed fresh.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1]
# The code a recording depends on: the recorder and the equations it reads.
_CODE_GLOBS = ("family/*.py", "blocks/gtap/*.py", "templates/gtap/gtap_block_model.py")
_DATA_FILES = ("basedata.har", "sets.har", "default.prm", "baserate.har")


def _code_version() -> str:
    h = hashlib.sha256()
    for pattern in _CODE_GLOBS:
        for f in sorted(_SRC.glob(pattern)):
            h.update(f.relative_to(_SRC).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def cache_key(dataset: str, closure: Any, residual_region: str | None) -> str:
    from equilibria.sources import dataset_dir

    h = hashlib.sha256()
    h.update(dataset.encode())
    d = dataset_dir(dataset)
    for name in _DATA_FILES:
        f = d / name
        h.update(name.encode())
        h.update(f.read_bytes() if f.exists() else b"<missing>")
    h.update(b"" if closure is None else closure.model_dump_json().encode())
    h.update(str(residual_region).encode())
    h.update(_code_version().encode())
    return h.hexdigest()[:32]


def load_or_build(
    dataset: str, params: Any, closure: Any, residual_region: str | None, root: Path
) -> tuple[Any, Any]:
    """``(backend, state)``: records from ``root/<key>/records.pkl`` when present."""
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.templates.gtap.gtap_block_model import compose_block_state

    model, state = compose_block_state(params, params.sets, closure, residual_region)
    path = Path(root) / cache_key(dataset, closure, residual_region) / "records.pkl"
    if path.exists():
        return FamilyBackend.load(path), state
    backend = FamilyBackend()
    backend.build(model, skip=state.ifsub_skipped)
    backend.save(path)
    return backend, state


def enable_jax_cache(root: Path) -> None:
    """Persist every compiled family program under ``root/jax``."""
    import jax

    jax.config.update("jax_compilation_cache_dir", str(Path(root) / "jax"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
