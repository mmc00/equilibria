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

    hf = harpy.HarFileObj._loadFromDisk(str(filepath))

    all_names = hf.getHeaderArrayNames()
    names_to_load = (
        [n for n in all_names if n in select_headers]
        if select_headers is not None
        else all_names
    )

    result: dict[str, HeaderArray] = {}
    for name in names_to_load:
        obj = hf[name]
        if obj.array is None:
            continue
        arr = np.array(obj.array)
        # setElements is a list of lists of strings; some metadata headers have None
        set_elements: list[list[str]] = []
        for elems in obj.setElements:
            if elems is None:
                set_elements.append([])
            else:
                set_elements.append([str(e).strip() for e in elems])
        result[name] = HeaderArray(
            name=name,
            coeff_name=obj.coeff_name.strip() if obj.coeff_name else name,
            long_name=obj.long_name.strip() if obj.long_name else "",
            array=arr,
            set_names=list(obj.setNames),
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
    hf = harpy.HarFileObj._loadFromDisk(str(filepath))
    return hf.getHeaderArrayNames()


def read_header_array(filepath: str | Path, name: str) -> HeaderArray:
    """Read a single named header array from a HAR file."""
    data = read_har(filepath, select_headers=[name])
    if name not in data:
        raise KeyError(f"Header '{name}' not found in {filepath}")
    return data[name]
