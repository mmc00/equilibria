from __future__ import annotations

from pathlib import Path

import numpy as np

from equilibria.babel.har.symbols import HeaderArray


def read_har(
    filepath: str | Path,
    select_headers: list[str] | None = None,
) -> dict[str, HeaderArray]:
    """Read a GEMPACK HAR file and return its header arrays.

    Args:
        filepath: Path to the .har or .prm file.
        select_headers: If provided, only these headers are loaded.

    Returns:
        Dict mapping header name → HeaderArray.

    Raises:
        FileNotFoundError: If filepath does not exist.
        ImportError: If harpy3 is not installed.
    """
    try:
        import harpy
    except ImportError as e:
        raise ImportError(
            "harpy3 is required to read HAR files. "
            "Install it with: pip install harpy3"
        ) from e

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"HAR file not found: {filepath}")

    _load = getattr(harpy.HarFileObj, "loadFromDisk", None) or getattr(harpy.HarFileObj, "_loadFromDisk")
    hf = _load(str(filepath))

    all_names = hf.getHeaderArrayNames()
    names_to_load = (
        [n for n in all_names if n in select_headers]
        if select_headers is not None
        else all_names
    )

    def _get(obj, key, default=None):
        # harpy3 newer API: dict-like via Mapping; older API: attribute access.
        # Older HeaderArrayObj uses __getitem__ for SET-ELEMENT indexing, not keys —
        # so only treat as dict if it's an actual Mapping/dict.
        if isinstance(obj, dict):
            v = obj.get(key, default)
            return v if v is not None else default
        return getattr(obj, key, default)

    result: dict[str, HeaderArray] = {}
    for name in names_to_load:
        _geth = getattr(hf, "getHeaderArrayObj", None) or getattr(hf, "_getHeaderArrayObj")
        obj = _geth(name)
        if obj is None:
            continue
        raw_arr = _get(obj, "array", None)
        if raw_arr is None:
            continue
        arr = np.array(raw_arr)

        # New harpy3 API: 'sets' is a list[{'name', 'dim_desc'}]
        sets_list = _get(obj, "sets", None)
        is_list_of_dicts = (
            isinstance(sets_list, list) and sets_list and isinstance(sets_list[0], dict)
        )
        if is_list_of_dicts:
            set_names = [str(s.get("name", "")).strip() for s in sets_list]
            set_elements = [
                [str(e).strip() for e in (s.get("dim_desc") or [])]
                for s in sets_list
            ]
        else:
            # Older harpy API
            raw_set_names = list(_get(obj, "setNames", []) or [])
            raw_set_elements = list(_get(obj, "setElements", []) or [])
            set_names = [
                str(n).strip() if n is not None else "" for n in raw_set_names
            ]
            set_elements = [
                [str(e).strip() for e in (elems or [])]
                for elems in raw_set_elements
            ]

        long_name = _get(obj, "long_name", "") or ""
        coeff_name = _get(obj, "coeff_name", "") or name
        result[name] = HeaderArray(
            name=name,
            coeff_name=str(coeff_name).strip() if coeff_name else name,
            long_name=str(long_name).strip(),
            array=arr,
            set_names=set_names,
            set_elements=set_elements,
        )
    return result


def get_header_names(filepath: str | Path) -> list[str]:
    """Return all header names in a HAR file without loading array data."""
    try:
        import harpy
    except ImportError as e:
        raise ImportError("harpy3 is required. Install with: pip install harpy3") from e
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"HAR file not found: {filepath}")
    _load = getattr(harpy.HarFileObj, "loadFromDisk", None) or getattr(harpy.HarFileObj, "_loadFromDisk")
    hf = _load(str(filepath))
    return hf.getHeaderArrayNames()


def read_header_array(filepath: str | Path, name: str) -> HeaderArray:
    """Read a single named header array from a HAR file."""
    data = read_har(filepath, select_headers=[name])
    if name not in data:
        raise KeyError(f"Header '{name}' not found in {filepath}")
    return data[name]
