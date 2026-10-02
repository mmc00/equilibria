"""Disk cache for calibrate_base's settled_seed.

Key = blocks/gtap/fingerprint.py over the settle's benchmark data, every closure
field, the residual region and the build+solve code.  Value =
``{var_name: {index_tuple_or_scalar: float}}`` stored as JSON (index keys encoded
as JSON themselves, so their type survives the round trip -- see ``_enc_key``).

``EQUILIBRIA_SEED_CACHE_DISABLE=1`` bypasses read+write. Cache dir defaults to
``~/.cache/equilibria/settled_seed`` or ``$EQUILIBRIA_SEED_CACHE``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from equilibria.blocks.gtap.fingerprint import file_digest, fingerprint

_SEP = "\x1f"  # legacy key separator (read-only, see _dec_key_legacy)

# Bumped when the on-disk layout changes; an unversioned file is the original
# \x1f-joined format and is still read.
_VERSION_FIELD = "_fmt"
_VERSION = 2

# Keys from the shared fingerprint start with KEY_PREFIX; the hand-built key they
# replaced started with "seed" and covered no code (see prune_old_keys).
KEY_PREFIX = "seed2"
_OLD_KEY_PREFIX = "seed"
_PRUNE_AGE_DAYS = 30


def disabled() -> bool:
    return os.environ.get("EQUILIBRIA_SEED_CACHE_DISABLE") == "1"


def _cache_dir() -> Path:
    d = os.environ.get("EQUILIBRIA_SEED_CACHE")
    p = Path(d) if d else Path.home() / ".cache" / "equilibria" / "settled_seed"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_key(
    dataset_id: str, closure, residual_region: str, params, ref_gdx=None
) -> str | None:
    """Key over the settle's data, closure, ref GDX and code -- see
    blocks/gtap/fingerprint.py.

    ``None`` means the inputs cannot be fully covered: skip the cache.
    """
    # The settle seeds from and solves against the ref GDX: key on its CONTENT.
    gdx = "nogdx"
    if ref_gdx is not None:
        gdx = file_digest(ref_gdx)
        if gdx is None:
            return None
    return fingerprint(
        KEY_PREFIX,
        params=params,
        closure=closure,
        residual_region=residual_region,
        dataset=dataset_id,
        ref_gdx=gdx,
    )


def prune_old_keys() -> list[str]:
    """Delete old-key seed files (``seed-*.json``) not written for 30 days.

    The cache dir is shared by every worktree, and a branch still on the old key
    keeps writing ``seed-*.json``; a recent one may be live, so only abandoned
    files go.  "Abandoned" is measured by the last WRITE (mtime): a read does not
    touch the file, and old branches run the old ``load``, so a seed another
    branch only reads is still pruned after 30 days -- that branch then
    recomputes it once.  Returns the removed file names.  Never raises.
    """
    if disabled():
        return []
    cutoff = time.time() - _PRUNE_AGE_DAYS * 86400
    removed = []
    for f in _cache_dir().glob(f"{_OLD_KEY_PREFIX}-*.json"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed.append(f.name)
        except OSError:
            pass
    return removed


def _enc_key(k):
    """Encode an index key as JSON so its TYPE survives the round trip.

    The previous encoding joined a tuple with ``\\x1f`` and split it back, which
    silently rewrote ``("USA", 2020)`` as ``("USA", "2020")`` and collapsed the
    1-tuple ``("USA",)`` to the bare string ``"USA"``.  That matters because the
    consumer (``gtap_multiperiod_driver``, base-calibrated seeding) does the
    lookup inside ``except (KeyError, TypeError, ValueError): pass`` -- a mistyped
    key raises nothing, it just seeds NOTHING, and the model solves from a
    different starting point with no diagnostic.

    JSON round-trips str/int/float/bool exactly; a tuple becomes a list, which
    ``_dec_key`` turns back into a tuple.  Today's GTAP index sets are all
    strings, so this changes no current behaviour -- it removes the trap.
    """
    return json.dumps(list(k) if isinstance(k, tuple) else k)


def _dec_key(s: str):
    """Inverse of :func:`_enc_key`."""
    dec = json.loads(s)
    return tuple(dec) if isinstance(dec, list) else dec


def _dec_key_legacy(s: str):
    """Decode a key written by the pre-versioning ``\\x1f`` encoder.

    Every key in such a file is a string (that encoder stringified everything),
    so this never guesses a type -- which is exactly why the format is chosen by
    the FILE's version marker and not sniffed per key.  Sniffing would decode a
    legacy ``"2020"`` as the int ``2020`` and reintroduce the very bug being
    fixed here.
    """
    return tuple(s.split(_SEP)) if _SEP in s else s


def load(key: str):
    if disabled():
        return None
    f = _cache_dir() / f"{key}.json"
    if not f.exists():
        return None
    raw = json.loads(f.read_text())
    # A versioned file wraps the payload; anything else is a pre-versioning file
    # whose keys must be read with the legacy decoder.
    if isinstance(raw, dict) and raw.get(_VERSION_FIELD) == _VERSION:
        raw, dec = raw["seed"], _dec_key
    else:
        dec = _dec_key_legacy
    return {
        name: {dec(k): float(v) for k, v in cells.items()}
        for name, cells in raw.items()
    }


def save(key: str, seed: dict) -> None:
    if disabled():
        return
    prune_old_keys()
    enc = {
        name: {_enc_key(k): float(v) for k, v in cells.items()}
        for name, cells in seed.items()
    }
    (_cache_dir() / f"{key}.json").write_text(
        json.dumps({_VERSION_FIELD: _VERSION, "seed": enc})
    )
