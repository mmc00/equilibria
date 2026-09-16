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
no exception, just a wrong answer. So the key covers BOTH inputs and code:

  * a sha256 of the benchmark CONTENT the build reads (not file paths: a path
    digest missed load_from_gdx, filter_config, and restored-mtime datasets)
  * a sha256 of every closure field, from model_dump() rather than an allowlist
  * a sha256 of the SOURCE of every module that builds the model

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

import hashlib
import logging
import os
import pickle
from pathlib import Path

_log = logging.getLogger(__name__)

_SEP = "\x1f"

# Every module whose source can change the built model. Paths are relative to the
# equilibria package root. A module listed here that is missing at runtime makes the
# key fall back to "missing" for that entry rather than raising: a wrong-but-stable
# key would be dangerous, an absent file is simply a different key.
_CODE_MODULES = (
    "templates/gtap/gtap_block_model.py",
    "templates/gtap/gtap_model_multiperiod.py",
    "templates/gtap/gtap_model_equations.py",
    "blocks/gtap/__init__.py",
    "blocks/gtap/_derived_params.py",
    "blocks/gtap/_ifsub_macros.py",
    "blocks/gtap/closure.py",
    "blocks/gtap/demand_utility.py",
    "blocks/gtap/factor.py",
    "blocks/gtap/income.py",
    "blocks/gtap/production_supply.py",
    "blocks/gtap/trade_armington_bilateral.py",
    "blocks/gtap/trade_cet.py",
    # The blocks above describe the model symbolically; these translate that
    # description into Pyomo objects, so they decide the built model just as
    # directly.  A bounds/domain policy change in pyomo_backend.py changes every
    # Var in the model without touching a single block.
    "backends/pyomo_backend.py",
    "backends/pyomo_equations.py",
    "blocks/base.py",
    "core/symbolic_equations.py",
    "core/parameters.py",
    "core/variables.py",
    "core/sets.py",
)

_DATA_FILES = ("basedata.har", "sets.har", "default.prm", "baserate.har")


def enabled() -> bool:
    """The cache is OFF unless explicitly turned on."""
    return os.environ.get("EQUILIBRIA_GTAP_MODEL_CACHE") == "1"


def _cache_dir() -> Path:
    d = os.environ.get("EQUILIBRIA_GTAP_MODEL_CACHE_DIR")
    p = Path(d) if d else Path.home() / ".cache" / "equilibria" / "model"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _package_root() -> Path:
    # .../equilibria/blocks/gtap/model_cache.py -> .../equilibria
    return Path(__file__).resolve().parent.parent.parent


def _code_digest() -> str:
    """sha256 over the source of every module that builds the model."""
    h = hashlib.sha256()
    root = _package_root()
    for rel in _CODE_MODULES:
        f = root / rel
        h.update(rel.encode())
        try:
            h.update(f.read_bytes())
        except OSError:
            h.update(b"missing")
    return h.hexdigest()[:24]


def _params_digest(params) -> str | None:
    """Digest the benchmark CONTENT the build reads -- not the file paths.

    Returns ``None`` when the content cannot be reached, which DISABLES the cache
    for that call.  That fallback is the whole point: the earlier version digested
    ``params._source_paths`` and defaulted to ``{}``, so any loader that did not
    populate it (``load_from_gdx`` does not) produced the sha256 of the empty
    string -- a perfectly valid, perfectly STABLE key with zero data coverage.
    Two different datasets collided and the second silently solved against the
    first one's model.  A degenerate-but-stable key is the one failure mode a
    cache key must never have; refusing to cache is always safe.

    Content-addressing also closes two holes that path+mtime could not:
    ``filter_config`` rewrites ``params.benchmark`` after load (same files, different
    model), and a restored/rsync'd dataset keeps its mtime while changing content.
    This mirrors ``seed_cache.cache_key``, which digests the same arrays.
    """
    bm = getattr(params, "benchmark", None)
    tx = getattr(params, "taxes", None)
    if bm is None and tx is None:
        return None
    parts = []
    srcs = (
        ("evfb", getattr(bm, "evfb", None)),
        ("vfm", getattr(bm, "vfm", None)),
        ("vkb", getattr(bm, "vkb", None)),
        ("vdfb", getattr(bm, "vdfb", None)),
        ("vmfb", getattr(bm, "vmfb", None)),
        ("vst", getattr(bm, "vst", None)),
        ("rtf", getattr(tx, "rtf", None)),
        ("kappaf_activity", getattr(tx, "kappaf_activity", None)),
    )
    seen_any = False
    for name, src in srcs:
        parts.append(name)
        if src is None:
            parts.append("none")
            continue
        try:
            items = sorted((str(k), round(float(v), 10)) for k, v in dict(src).items())
        except (TypeError, ValueError):
            return None  # unreadable -> do not cache rather than key on nothing
        seen_any = True
        parts.append(hashlib.sha256(repr(items).encode()).hexdigest()[:16])
    if not seen_any:
        return None
    return hashlib.sha256(_SEP.join(parts).encode()).hexdigest()[:24]


def _closure_digest(closure) -> str | None:
    """Digest EVERY closure field, not a hand-kept allowlist.

    ``GTAPClosureConfig`` is a frozen pydantic model with ``extra="forbid"``, so
    ``model_dump()`` enumerates the fields exhaustively and a newly added field is
    covered the day it appears.  The previous allowlist of 9 fields omitted
    ``va_subsidy_basis``, which selects ``ftrv + fbep`` vs ``ftrv - fbep`` in
    ``_derived_params._va_wedge`` and moves the VA-vs-intermediates weighting
    (0.571 vs 0.679 on subsidised agriculture) -- two closures differing only in
    that field shared a key.
    """
    dump = getattr(closure, "model_dump", None)
    if dump is None:
        return None
    try:
        data = dump()
    except Exception:
        return None
    try:
        return hashlib.sha256(
            repr(sorted((str(k), repr(v)) for k, v in data.items())).encode()
        ).hexdigest()[:24]
    except Exception:
        return None


def cache_key(
    params,
    closure,
    residual_region: str,
    base_calibrated: bool,
    ref_gdx=None,
) -> str | None:
    """Key covering the inputs AND the code that turn into the built model.

    Returns ``None`` when any component cannot be computed -- the caller must then
    skip the cache entirely.  Every ``None`` path here is a case where a key could
    still have been produced but would not have covered what it claims to cover.
    """
    pd = _params_digest(params)
    cd = _closure_digest(closure)
    if pd is None or cd is None:
        return None
    fields = [
        residual_region,
        str(bool(base_calibrated)),
        # The ref GDX only seeds; its presence still changes the built model's values.
        "gdx" if ref_gdx is not None else "nogdx",
        cd,
        pd,
        _code_digest(),
    ]
    return "model-" + hashlib.sha256(_SEP.join(fields).encode()).hexdigest()[:24]


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
