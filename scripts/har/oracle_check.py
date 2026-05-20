"""Manual sandbox harness: harpy3 as black-box oracle for the HAR writer.

This script is NOT imported by any equilibria source code, NEVER runs in
CI, and exists only to capture golden JSON fixtures under
tests/babel/har/golden/.

Workflow:
    1. Create a fresh venv:    python3 -m venv .oracle_venv
    2. Activate:               source .oracle_venv/bin/activate
    3. Install harpy3 only:    pip install harpy3 numpy
    4. From the repo root:     python scripts/har/oracle_check.py refresh
       or compare:             python scripts/har/oracle_check.py compare

The "refresh" subcommand reads each GTAP fixture with harpy3 and writes
the structured dump (per-header: shape, set_names, set_elements, sum,
first/last value, dtype kind) to tests/babel/har/golden/<fixture>.json.

The "compare" subcommand:
    - runs equilibria's writer over the fixture (read_har -> write_har),
    - reads the writer output with harpy3,
    - reads the original GEMPACK file with harpy3,
    - asserts the two harpy3 reads are identical (this is the L3-via-oracle
      guarantee: harpy3 sees our output the same way it sees the original).

This file deliberately uses only the harpy3 public Python API. The
implementation MUST NOT import any module from harpy3._internal or
read any harpy3 source. If a needed capability isn't exposed, capture
the gap and stop — never paper over it by reading harpy3 source.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "src/equilibria/templates/reference/gtap/data"
GOLDEN = REPO_ROOT / "tests/babel/har/golden"

FIXTURES = [
    DATA / "nus333/sets.har",
    DATA / "nus333/basedata.har",
    DATA / "nus333/baserate.har",
    DATA / "nus333/default.prm",
    DATA / "9x10/sets.har",
    DATA / "9x10/basedata.har",
]


def _import_harpy_or_die():
    try:
        import harpy3 as harpy  # type: ignore
        return harpy, "harpy3"
    except ImportError:
        try:
            import harpy  # type: ignore
            return harpy, "harpy"
        except ImportError:
            sys.exit(
                "ERROR: harpy3 (or harpy) is not installed. "
                "This script is meant to run in a sandbox venv "
                "(see module docstring)."
            )


def _dump_har_with_harpy(harpy, path: Path) -> dict:
    """Read a HAR file with harpy and capture a structured per-header dump.

    Only public attributes are read: shape, set names, set elements,
    array statistics. No internals.
    """
    obj = harpy.HarFileObj.loadFromDisk(str(path))
    out: dict = {"path": str(path), "headers": {}}
    for name in obj.getHeaderArrayNames():
        ha = obj.getHeaderArrayObj(name)
        arr = ha.array
        shape = list(arr.shape)
        dtype_kind = arr.dtype.kind
        set_names = list(getattr(ha, "set_names", []) or [])
        set_elements = [list(map(str, e)) for e in (getattr(ha, "set_elements", []) or [])]
        if dtype_kind in ("f", "i"):
            stats = {
                "sum": float(arr.sum()),
                "min": float(arr.min()) if arr.size else 0.0,
                "max": float(arr.max()) if arr.size else 0.0,
                "first": float(arr.ravel()[0]) if arr.size else 0.0,
                "last": float(arr.ravel()[-1]) if arr.size else 0.0,
            }
        else:
            stats = {
                "first": str(arr.ravel()[0]) if arr.size else "",
                "last": str(arr.ravel()[-1]) if arr.size else "",
            }
        out["headers"][name] = {
            "long_name": getattr(ha, "long_name", ""),
            "dtype_kind": dtype_kind,
            "shape": shape,
            "set_names": set_names,
            "set_elements": set_elements,
            "stats": stats,
        }
    return out


def cmd_refresh() -> int:
    harpy, _ = _import_harpy_or_die()
    GOLDEN.mkdir(parents=True, exist_ok=True)
    for fixture in FIXTURES:
        if not fixture.exists():
            print(f"SKIP {fixture} (not present)")
            continue
        dump = _dump_har_with_harpy(harpy, fixture)
        out_path = GOLDEN / f"{fixture.parent.name}_{fixture.stem}.json"
        out_path.write_text(json.dumps(dump, indent=2, sort_keys=True))
        print(f"wrote {out_path}")
    return 0


def cmd_compare() -> int:
    import tempfile

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from equilibria.babel.har import read_har, write_har  # noqa

    harpy, _ = _import_harpy_or_die()
    fails = 0
    for fixture in FIXTURES:
        if not fixture.exists():
            print(f"SKIP {fixture} (not present)")
            continue
        with tempfile.NamedTemporaryFile(suffix=fixture.suffix, delete=False) as tf:
            tmp_path = Path(tf.name)
        try:
            write_har(tmp_path, read_har(fixture))
            orig_dump = _dump_har_with_harpy(harpy, fixture)
            new_dump = _dump_har_with_harpy(harpy, tmp_path)
            orig_dump.pop("path", None)
            new_dump.pop("path", None)
            if orig_dump == new_dump:
                print(f"OK   {fixture.name}")
            else:
                print(f"DIFF {fixture.name}")
                fails += 1
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return 1 if fails else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", choices=["refresh", "compare"])
    args = p.parse_args()
    if args.cmd == "refresh":
        return cmd_refresh()
    if args.cmd == "compare":
        return cmd_compare()
    return 2


if __name__ == "__main__":
    sys.exit(main())
