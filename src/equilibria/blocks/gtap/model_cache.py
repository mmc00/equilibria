"""Disk cache for the built Pyomo model (opt-in).

``build_block_model`` on gtap7_20x41 costs ~4 min and produces a ConcreteModel with
631,512 active constraints / 3,435,916 vars. Loading that same model back from a
cloudpickle file costs ~1.35 min — MEASURED 3.6x, with the solve reaching
byte-identical parity (94.5% within-1pp, median 0.1523pp, code=1) from both paths.

Why cloudpickle and not pickle: ``GTAPBlockMultiPeriodModel.build_vars`` initializes
vars through a nested closure (``_mk_init.<locals>._init``), which stdlib pickle cannot
serialize ("Can't get local object"). cloudpickle serializes closures by value. Reading
back works with plain ``pickle.load`` — the file is self-contained.

THE CACHE KEY IS THE WHOLE POINT. A stale model would solve silently against outdated
data or outdated equations, which is the same class of failure as a stale name cache:
no exception, just a wrong answer. So the key covers BOTH inputs and code; it is
built by blocks/gtap/fingerprint.py, the same key the seed cache uses.  Content,
not file paths: a path digest missed load_from_gdx, filter_config, and
restored-mtime datasets.

When any of those cannot be computed, ``cache_key`` returns ``None`` and the
caller skips the cache. Refusing to cache is safe; a key that does not cover what
it claims to cover is not.

The code digest is what makes this safe to leave on: edit any equation, any block, or
the composer, and the key changes, so the next run rebuilds instead of loading a model
that no longer matches the code.

OPT-IN: set ``EQUILIBRIA_GTAP_MODEL_CACHE=1``. Off by default — a 0.88 GB artifact per
key is not something to write behind the user's back. Cache dir defaults to
``~/.cache/equilibria/model`` or ``$EQUILIBRIA_GTAP_MODEL_CACHE_DIR``.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

from equilibria.blocks.gtap.fingerprint import file_digest, fingerprint

_log = logging.getLogger(__name__)


def enabled() -> bool:
    """The cache is OFF unless explicitly turned on."""
    return os.environ.get("EQUILIBRIA_GTAP_MODEL_CACHE") == "1"


def _cache_dir() -> Path:
    d = os.environ.get("EQUILIBRIA_GTAP_MODEL_CACHE_DIR")
    p = Path(d) if d else Path.home() / ".cache" / "equilibria" / "model"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_key(
    params,
    closure,
    residual_region: str,
    base_calibrated: bool,
    ref_gdx=None,
) -> str | None:
    """Key covering the inputs AND the code that turn into the built model.

    Returns ``None`` when any component cannot be computed -- the caller must then
    skip the cache entirely.
    """
    # The ref GDX only seeds, but its values land in the built model: key on its
    # CONTENT (two different GDX at one path, or one GDX at two paths).
    gdx = "nogdx"
    if ref_gdx is not None:
        gdx = file_digest(ref_gdx)
        if gdx is None:
            return None
    return fingerprint(
        "model",
        params=params,
        closure=closure,
        residual_region=residual_region,
        base_calibrated=bool(base_calibrated),
        ref_gdx=gdx,
    )


def load(key: str):
    """Return the cached model, or None on a miss or any read failure.

    A corrupt or unreadable cache file must never fail the run — it degrades to a
    rebuild, which is exactly what the caller would have done without a cache.
    """
    if not enabled():
        return None
    f = _cache_dir() / f"{key}.pkl"
    if not f.exists():
        return None
    try:
        with open(f, "rb") as fh:
            m = pickle.load(fh)
    except Exception as exc:  # corrupt file, version skew, truncated write
        _log.warning("model cache: unreadable %s (%s) — rebuilding", f.name, exc)
        return None
    _log.info("model cache HIT: %s (%.2f GB)", f.name, f.stat().st_size / 1e9)
    return m


def save(key: str, model) -> None:
    """Write the model under key. Never raises: a cache write must not fail a run.

    Writes to a temp file and renames, so a crash mid-write cannot leave a truncated
    file that a later run would read as valid.
    """
    if not enabled():
        return
    try:
        import cloudpickle  # ty: ignore[unresolved-import]  (optional dep)
    except ImportError:
        _log.warning(
            "model cache: cloudpickle not installed — cannot save "
            "(stdlib pickle fails on the var-init closures). pip install cloudpickle"
        )
        return
    d = _cache_dir()
    tmp = d / f"{key}.pkl.tmp{os.getpid()}"
    try:
        with open(tmp, "wb") as fh:
            cloudpickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(d / f"{key}.pkl")
        _log.info(
            "model cache SAVED: %s.pkl (%.2f GB)",
            key,
            (d / f"{key}.pkl").stat().st_size / 1e9,
        )
    except Exception as exc:
        _log.warning("model cache: save failed (%s) — continuing", exc)
        try:
            tmp.unlink()
        except OSError:
            pass
