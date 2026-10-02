"""The one cache key shared by the model cache and the seed cache.

Both caches used to build their key by hand and drifted apart: the same
"shifts/elasticities missing from the key" fix had to land twice in one day
(b15f9b9 seed, b5b191e model), the seed key kept a 5-field closure allowlist
after the model key had moved to ``model_dump()`` because that allowlist let
``va_subsidy_basis`` collide, and the seed key covered no code at all.

The key covers:

  * every attribute of ``params``: the data CONTENT, not file paths
  * every closure field, via ``model_dump()``
  * the residual region and the caller's flags
  * the source of every ``.py`` under ``CODE_FOLDERS`` (+ ``CODE_FILES``): the
    code that builds the model AND the code that solves it, because the settled
    seed comes out of a full settle solve.  Modules outside these paths that the
    build imports must be added here.

When any of that cannot be read, ``fingerprint`` returns ``None`` and the caller
must skip the cache.  Refusing to cache is safe; a key that does not cover what
it claims to cover is not.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path

_SEP = "\x1f"

# Folders (relative to the equilibria package root) whose Python source can change
# the built model or the settled seed.  Walked recursively, so a new module is
# covered the day it appears -- the old hand list missed floors.py,
# declarations.py, agents.py and the whole driver/solver.
CODE_FOLDERS = (
    "blocks/gtap",
    "templates/gtap",
    "solver",
    "backends",
    "core",
    "contracts",  # gtap_contract builds on it
    "babel/gdx",  # reads the ref GDX that seeds the model
)
CODE_FILES = ("blocks/base.py", "model.py")  # model.py: the block composer

# Groups of numeric data on GTAPParameters.  Every attribute of every group is
# digested (the old keys hashed 8 hand-picked arrays and missed rtms, rtxs, vom,
# makb...: two runs differing only in tariffs shared a model).  A non-numeric
# value inside these groups means a broken load: refuse to key it.
_NUMERIC_GROUPS = (
    "benchmark",
    "taxes",
    "elasticities",
    "shifts",
    "shares",
    "calibrated",
)

_MAX_DEPTH = 12


class _Uncoverable(Exception):
    """A value the digest cannot cover deterministically."""


def _package_root() -> Path:
    # .../equilibria/blocks/gtap/fingerprint.py -> .../equilibria
    return Path(__file__).resolve().parent.parent.parent


def _repr_key(obj) -> str:
    """repr used for ordering/keys; refuses the default repr, whose memory
    address would change the key from one process to the next."""
    r = repr(obj)
    if " at 0x" in r:
        raise _Uncoverable(r)
    return r


def _feed(h, obj, numeric: bool, depth: int = 0) -> None:
    """Stream a deterministic encoding of ``obj`` into ``h``.

    Dicts are walked in sorted-key order, objects by their sorted attributes,
    numbers rounded to 10 decimals (so ``1`` and ``1.0`` match, as do a list and
    a tuple with the same items).  A ``Path`` counts by the CONTENT of its file:
    loaders store their source path on params, and the key must not change when
    the same data is loaded from another location.  ``numeric`` is set inside
    ``_NUMERIC_GROUPS``, where only numbers (or None) may sit at the leaves.

    Raises ``_Uncoverable`` (= do not cache) for anything it cannot encode
    deterministically: a non-number inside a numeric group, an unreadable path,
    a callable or class, or a default repr carrying a memory address.
    """
    if depth > _MAX_DEPTH:
        raise _Uncoverable("too deep")
    if obj is None or isinstance(obj, bool):
        h.update(repr(obj).encode())
    elif isinstance(obj, Enum):
        h.update(b"e" + type(obj).__name__.encode() + b".")
        _feed(h, obj.value, numeric, depth + 1)
    elif isinstance(obj, (int, float)):
        h.update(b"n" + repr(round(float(obj), 10)).encode())
    elif hasattr(obj, "item") and hasattr(obj, "dtype") and not hasattr(obj, "__len__"):
        _feed(h, obj.item(), numeric, depth + 1)  # numpy scalar
    elif isinstance(obj, str):
        if numeric:
            raise _Uncoverable(f"non-numeric value {obj!r}")
        h.update(b"s" + obj.encode())
    elif isinstance(obj, dict):
        h.update(b"{")
        for k, v in sorted(obj.items(), key=lambda kv: _repr_key(kv[0])):
            h.update(_repr_key(k).encode() + b":")
            _feed(h, v, numeric, depth + 1)
        h.update(b"}")
    elif isinstance(obj, (list, tuple)):
        h.update(b"[")
        for v in obj:
            _feed(h, v, numeric, depth + 1)
        h.update(b"]")
    elif isinstance(obj, (set, frozenset)):
        h.update(b"<")
        for v in sorted(obj, key=_repr_key):
            _feed(h, v, numeric, depth + 1)
        h.update(b">")
    elif isinstance(obj, Path):
        digest = file_digest(obj)
        if digest is None:
            raise _Uncoverable(f"unreadable path {obj}")
        h.update(b"f" + digest.encode())
    elif hasattr(obj, "index") and hasattr(obj, "to_dict"):
        _feed(h, obj.to_dict(), numeric, depth + 1)  # pandas: keep the index
    elif hasattr(obj, "tolist"):
        _feed(h, obj.tolist(), numeric, depth + 1)  # numpy array
    elif callable(obj) or isinstance(obj, type):
        # vars() of a function is empty, so every callable would hash alike.
        raise _Uncoverable(f"callable {obj!r}")
    elif hasattr(obj, "__dict__"):
        h.update(type(obj).__name__.encode() + b"(")
        for name, val in sorted(vars(obj).items()):
            h.update(name.encode() + b"=")
            _feed(h, val, numeric, depth + 1)
        h.update(b")")
    else:
        h.update(b"r" + _repr_key(obj).encode())


def _has_benchmark(params) -> bool:
    bm = getattr(params, "benchmark", None)
    if bm is None or not hasattr(bm, "__dict__"):
        return False
    return any(isinstance(v, dict) and v for v in vars(bm).values())


def _params_digest(params) -> str | None:
    """Digest EVERY attribute of ``params``: the numeric groups strictly, the rest
    (sets, closure-derived flags written by build_block_model...) generically."""
    if not _has_benchmark(params):
        # No benchmark content: a key here would be stable and cover nothing.
        return None
    h = hashlib.sha256()
    try:
        for name, val in sorted(vars(params).items()):
            h.update(name.encode() + b"=")
            _feed(h, val, numeric=name in _NUMERIC_GROUPS)
    except _Uncoverable:
        return None
    return h.hexdigest()[:24]


def file_digest(path) -> str | None:
    """Digest of a file's CONTENT (a ref GDX, say), or None when unreadable."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:24]
    except (OSError, TypeError):
        return None


def _closure_digest(closure) -> str | None:
    dump = getattr(closure, "model_dump", None)
    if dump is None:
        return None
    h = hashlib.sha256()
    try:
        _feed(h, dump(), numeric=False)
    except Exception:
        return None
    return h.hexdigest()[:24]


def _code_files(root: Path) -> list[Path]:
    files = []
    for rel in CODE_FOLDERS:
        files.extend((root / rel).rglob("*.py"))
    files.extend(root / rel for rel in CODE_FILES)
    return sorted(set(files))


def _code_digest() -> str:
    root = _package_root()
    h = hashlib.sha256()
    for f in _code_files(root):
        h.update(f.relative_to(root).as_posix().encode())
        try:
            h.update(f.read_bytes())
        except OSError:
            h.update(b"missing")
    return h.hexdigest()[:24]


def fingerprint(
    namespace: str, *, params, closure, residual_region: str, **flags
) -> str | None:
    """``"<namespace>-<hash>"`` over data, closure, flags and code, or ``None``
    when any of them cannot be covered (the caller then skips the cache)."""
    pd = _params_digest(params)
    cd = _closure_digest(closure)
    if pd is None or cd is None:
        return None
    fields = [
        namespace,
        str(residual_region),
        repr(sorted((k, repr(v)) for k, v in flags.items())),
        cd,
        pd,
        _code_digest(),
    ]
    return f"{namespace}-" + hashlib.sha256(_SEP.join(fields).encode()).hexdigest()[:24]
