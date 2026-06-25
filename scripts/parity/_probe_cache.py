"""Cache key + store/load for the probe's solve point. Pure (no Pyomo)."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]
_GTAP = _ROOT / "src" / "equilibria" / "templates" / "gtap"

# Equation-source files whose edits must invalidate the cached solve point.
KEY_FILES = [
    _GTAP / "gtap_model_equations.py",
    _GTAP / "altertax" / "parameter_overrides.py",
    _GTAP / "altertax" / "calibration_sequence.py",
    _GTAP / "altertax" / "postmodel.py",
    _GTAP / "gtap_parameters.py",
]

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "equilibria_probe"


def compute_cache_key(dataset: str, scenario: str, closure_name: str,
                      key_files: list[Path] | None = None) -> str:
    files = KEY_FILES if key_files is None else key_files
    h = hashlib.sha256()
    h.update(dataset.encode())
    h.update(b"\0")
    h.update(scenario.encode())
    h.update(b"\0")
    h.update(closure_name.encode())
    for p in files:
        h.update(b"\0")
        try:
            h.update(Path(p).read_bytes())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()[:16]


def store_solution(key: str, solution: dict, cache_dir: Path | None = None) -> Path:
    d = cache_dir or DEFAULT_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{key}.pkl"
    with path.open("wb") as fh:
        pickle.dump(solution, fh)
    return path


def load_solution(key: str, cache_dir: Path | None = None) -> Optional[dict]:
    d = cache_dir or DEFAULT_CACHE_DIR
    path = d / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None  # corrupt cache → treat as miss


def clear_cache(cache_dir: Path | None = None) -> int:
    d = cache_dir or DEFAULT_CACHE_DIR
    if not d.exists():
        return 0
    n = 0
    for p in d.glob("*.pkl"):
        p.unlink()
        n += 1
    return n
