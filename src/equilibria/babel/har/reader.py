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

    hf = harpy.HarFileObj.loadFromDisk(str(filepath))

    all_names = hf.getHeaderArrayNames()
    names_to_load = (
        [n for n in all_names if n in select_headers]
        if select_headers is not None
        else all_names
    )

    result: dict[str, HeaderArray] = {}
    for name in names_to_load:
        obj = hf.getHeaderArrayObj(name)
        if obj is None:
            continue
        raw_arr = obj["array"] if hasattr(obj, "__getitem__") else obj.array
        if raw_arr is None:
            continue
        arr = np.array(raw_arr)
        sets_meta = obj.get("sets") if hasattr(obj, "get") else None
        sets_meta = sets_meta or []
        set_names: list[str] = []
        set_elements: list[list[str]] = []
        for d in sets_meta:
            set_names.append(str(d.get("name", "")))
            descs = d.get("dim_desc") or []
            set_elements.append([str(e).strip() for e in descs])
        long_name = (obj.get("long_name") if hasattr(obj, "get") else "") or ""
        coeff_name = (obj.get("coeff_name") if hasattr(obj, "get") else None) or name
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
    hf = harpy.HarFileObj.loadFromDisk(str(filepath))
    return hf.getHeaderArrayNames()


def read_header_array(filepath: str | Path, name: str) -> HeaderArray:
    """Read a single named header array from a HAR file."""
    data = read_har(filepath, select_headers=[name])
    if name not in data:
        raise KeyError(f"Header '{name}' not found in {filepath}")
    return data[name]
