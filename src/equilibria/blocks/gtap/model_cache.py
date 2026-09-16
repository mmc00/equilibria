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

  * the dataset id and the mtime+size of every .har/.prm it reads
  * every closure field that reaches the build
  * a sha256 of the SOURCE of every module that builds the model

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


def _data_digest(dataset_dir) -> str:
    """mtime+size of each input file. Cheap, and enough to catch a regenerated dataset.

    Hashing the .har contents would be stricter but costs seconds on every run; mtime
    changes whenever the file is rewritten, which is how these datasets get updated.

    Accepts either a directory holding the standard filenames, or a mapping of
    ``{label: path}`` as recorded on ``params._source_paths`` by ``load_from_har``
    (the datasets are not always laid out as one directory per model).
    """
    h = hashlib.sha256()
    if isinstance(dataset_dir, dict):
        items = [(k, dataset_dir.get(k)) for k in sorted(dataset_dir)]
    else:
        d = Path(dataset_dir)
        items = [(n, d / n) for n in _DATA_FILES]
    for name, f in items:
        h.update(str(name).encode())
        if f is None:
            h.update(b"none")
            continue
        try:
            st = Path(f).stat()
            h.update(f"{st.st_mtime_ns}:{st.st_size}".encode())
        except OSError:
            h.update(b"missing")
    return h.hexdigest()[:24]


def cache_key(
    dataset_id: str,
    dataset_dir,
    closure,
    residual_region: str,
    base_calibrated: bool,
    ref_gdx=None,
) -> str:
    """Key covering the inputs AND the code that turn into the built model."""
    fields = [
        dataset_id,
        residual_region,
        str(bool(base_calibrated)),
        # The ref GDX only seeds; its presence still changes the built model's values.
        "gdx" if ref_gdx is not None else "nogdx",
        str(getattr(closure, "name", "")),
        str(getattr(closure, "closure_type", "")),
        str(getattr(closure, "savf_flag", "")),
        str(bool(getattr(closure, "if_sub", False))),
        str(getattr(closure, "capital_mobility", "")),
        str(getattr(closure, "numeraire", "")),
        str(bool(getattr(closure, "fix_endowments", False))),
        str(bool(getattr(closure, "fix_taxes", False))),
        str(bool(getattr(closure, "fix_technology", False))),
        _data_digest(dataset_dir),
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
