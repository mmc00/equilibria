"""The one cache key shared by the model cache and the seed cache.

Both caches used to build their key by hand and drifted apart: the same
"shifts/elasticities missing from the key" fix had to land twice in one day
(b15f9b9 seed, b5b191e model), the seed key kept a 5-field closure allowlist
after the model key had moved to ``model_dump()`` because that allowlist let
``va_subsidy_basis`` collide, and the seed key covered no code at all.

The key covers:

  * the benchmark CONTENT the build reads (every group in ``_PARAM_GROUPS``),
    plus every attribute of ``params.elasticities`` and ``params.shifts``
  * every closure field, via ``model_dump()``
  * the residual region and the caller's flags
  * the source of every ``.py`` under ``CODE_FOLDERS`` (+ ``CODE_FILES``): the
    code that builds the model AND the code that solves it, because the settled
    seed comes out of a full settle solve

When any of that cannot be read, ``fingerprint`` returns ``None`` and the caller
must skip the cache.  Refusing to cache is safe; a key that does not cover what
it claims to cover is not.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_SEP = "\x1f"

# Folders (relative to the equilibria package root) whose Python source can change
# the built model or the settled seed.  Walked recursively, so a new module is
# covered the day it appears -- the old hand list missed floors.py,
# declarations.py, agents.py and the whole driver/solver.
CODE_FOLDERS = ("blocks/gtap", "templates/gtap", "solver", "backends", "core")
CODE_FILES = ("blocks/base.py",)

_PARAM_GROUPS = (
    ("benchmark", ("evfb", "vfm", "vkb", "vdfb", "vmfb", "vst")),
    ("taxes", ("rtf", "kappaf_activity")),
)


def _package_root() -> Path:
    # .../equilibria/blocks/gtap/fingerprint.py -> .../equilibria
    return Path(__file__).resolve().parent.parent.parent


def _sha(obj) -> str:
    return hashlib.sha256(repr(obj).encode()).hexdigest()[:16]


def _items(d) -> list:
    """Sorted (key, value) pairs, numbers rounded to 10 digits.  Raises on a
    non-numeric value: the caller turns that into "do not cache"."""
    return sorted((str(k), round(float(v), 10)) for k, v in dict(d).items())


def _params_digest(params) -> str | None:
    parts = []
    seen_any = False
    for group_name, names in _PARAM_GROUPS:
        group = getattr(params, group_name, None)
        for name in names:
            src = getattr(group, name, None)
            parts.append(name)
            if src is None:
                parts.append("none")
                continue
            try:
                parts.append(_sha(_items(src)))
            except (TypeError, ValueError):
                return None
            seen_any = True
    if not seen_any:
        # No benchmark at all: a key here would be stable and cover nothing.
        return None
    # The build and the settle bake the elasticities and the shifters into every
    # equation; every attribute counts, so a field added later is covered too.
    for group_name in ("elasticities", "shifts"):
        group = getattr(params, group_name, None)
        parts.append(group_name)
        if group is None:
            parts.append("none")
            continue
        try:
            parts.append(
                _sha(
                    sorted(
                        (attr, _items(val) if isinstance(val, dict) else repr(val))
                        for attr, val in vars(group).items()
                    )
                )
            )
        except (TypeError, ValueError):
            return None
    return _sha(parts)


def _closure_digest(closure) -> str | None:
    dump = getattr(closure, "model_dump", None)
    if dump is None:
        return None
    try:
        return _sha(sorted((str(k), repr(v)) for k, v in dump().items()))
    except Exception:
        return None


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
