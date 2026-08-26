"""Disk cache for calibrate_base's settled_seed.

Key = hash of everything that changes the settle: dataset id, closure fields,
residual region, and a digest of the benchmark params the settle reads. Value =
``{var_name: {index_tuple_or_scalar: float}}`` stored as JSON (tuple keys encoded
with a \\x1f separator).

``EQUILIBRIA_SEED_CACHE_DISABLE=1`` bypasses read+write. Cache dir defaults to
``~/.cache/equilibria/settled_seed`` or ``$EQUILIBRIA_SEED_CACHE``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

_SEP = "\x1f"


def disabled() -> bool:
    return os.environ.get("EQUILIBRIA_SEED_CACHE_DISABLE") == "1"


def _cache_dir() -> Path:
    d = os.environ.get("EQUILIBRIA_SEED_CACHE")
    p = Path(d) if d else Path.home() / ".cache" / "equilibria" / "settled_seed"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_key(dataset_id: str, closure, residual_region: str, params) -> str:
    fields = [
        dataset_id, residual_region,
        str(getattr(closure, "closure_type", "")),
        str(getattr(closure, "savf_flag", "")),
        str(bool(getattr(closure, "if_sub", False))),
        str(getattr(closure, "capital_mobility", "")),
        str(getattr(closure, "numeraire", "")),
    ]
    # Digest of the benchmark inputs the settle depends on (evfb/vfm/vkb + tax rates).
    bm = getattr(params, "benchmark", None)
    tx = getattr(params, "taxes", None)
    for src in (getattr(bm, "evfb", None), getattr(bm, "vfm", None),
                getattr(bm, "vkb", None), getattr(tx, "rtf", None),
                getattr(tx, "kappaf_activity", None)):
        if src is None:
            fields.append("none")
            continue
        items = sorted((str(k), round(float(v), 10)) for k, v in dict(src).items())
        fields.append(hashlib.sha256(repr(items).encode()).hexdigest()[:16])
    return "seed-" + hashlib.sha256(_SEP.join(fields).encode()).hexdigest()[:24]


def _enc_key(k):
    return _SEP.join(map(str, k)) if isinstance(k, tuple) else str(k)


def _dec_key(s: str):
    return tuple(s.split(_SEP)) if _SEP in s else s


def load(key: str):
    if disabled():
        return None
    f = _cache_dir() / f"{key}.json"
    if not f.exists():
        return None
    raw = json.loads(f.read_text())
    return {name: {_dec_key(k): float(v) for k, v in cells.items()}
            for name, cells in raw.items()}


def save(key: str, seed: dict) -> None:
    if disabled():
        return
    enc = {name: {_enc_key(k): float(v) for k, v in cells.items()}
           for name, cells in seed.items()}
    (_cache_dir() / f"{key}.json").write_text(json.dumps(enc))
