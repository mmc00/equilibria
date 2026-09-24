"""Disk cache for calibrate_base's settled_seed.

Key = hash of everything that changes the settle: dataset id, closure fields,
residual region, and a digest of the benchmark params the settle reads. Value =
``{var_name: {index_tuple_or_scalar: float}}`` stored as JSON (index keys encoded
as JSON themselves, so their type survives the round trip -- see ``_enc_key``).

``EQUILIBRIA_SEED_CACHE_DISABLE=1`` bypasses read+write. Cache dir defaults to
``~/.cache/equilibria/settled_seed`` or ``$EQUILIBRIA_SEED_CACHE``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

_SEP = "\x1f"  # legacy key separator (read-only, see _dec_key_legacy)

# Bumped when the on-disk layout changes; an unversioned file is the original
# \x1f-joined format and is still read.
_VERSION_FIELD = "_fmt"
_VERSION = 2


def disabled() -> bool:
    return os.environ.get("EQUILIBRIA_SEED_CACHE_DISABLE") == "1"


def _cache_dir() -> Path:
    d = os.environ.get("EQUILIBRIA_SEED_CACHE")
    p = Path(d) if d else Path.home() / ".cache" / "equilibria" / "settled_seed"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_key(dataset_id: str, closure, residual_region: str, params) -> str:
    # DEUDA CONOCIDA (2026-09-22): esta clave cubre los DATOS de entrada (dataset,
    # closure, benchmark, tasas) pero NO el CODIGO que calcula el seed.  Editar el
    # settle o cualquier ecuacion que lo alimenta no invalida la entrada, asi que
    # un seed viejo se sigue sirviendo en silencio — y el cache vive fuera del
    # repo (~/.cache/equilibria/settled_seed), asi que sobrevive a checkouts y
    # ramas.  Medido: 12 entradas de agosto-septiembre en una maquina de trabajo.
    #
    # El cache de MODELOS hermano si lo cubre — ver blocks/gtap/model_cache.py:
    # "La clave cubre los archivos de entrada Y el source de cada modulo que
    # construye el modelo, asi que una ecuacion editada nunca puede recibir un
    # modelo obsoleto".  Aqui falta ese mismo digest de fuentes.
    #
    # Mientras tanto: EQUILIBRIA_SEED_CACHE_DISABLE=1 lo desactiva, y conviene
    # borrar el directorio al medir contra una referencia.
    fields = [
        dataset_id,
        residual_region,
        str(getattr(closure, "closure_type", "")),
        str(getattr(closure, "savf_flag", "")),
        str(bool(getattr(closure, "if_sub", False))),
        str(getattr(closure, "capital_mobility", "")),
        str(getattr(closure, "numeraire", "")),
    ]
    # Digest of the benchmark inputs the settle depends on (evfb/vfm/vkb + tax rates).
    bm = getattr(params, "benchmark", None)
    tx = getattr(params, "taxes", None)
    for src in (
        getattr(bm, "evfb", None),
        getattr(bm, "vfm", None),
        getattr(bm, "vkb", None),
        getattr(tx, "rtf", None),
        getattr(tx, "kappaf_activity", None),
    ):
        if src is None:
            fields.append("none")
            continue
        items = sorted((str(k), round(float(v), 10)) for k, v in dict(src).items())
        fields.append(hashlib.sha256(repr(items).encode()).hexdigest()[:16])
    return "seed-" + hashlib.sha256(_SEP.join(fields).encode()).hexdigest()[:24]


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
    enc = {
        name: {_enc_key(k): float(v) for k, v in cells.items()}
        for name, cells in seed.items()
    }
    (_cache_dir() / f"{key}.json").write_text(
        json.dumps({_VERSION_FIELD: _VERSION, "seed": enc})
    )
