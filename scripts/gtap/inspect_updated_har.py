"""Inspect a RunGTAP updated.har: print every header's name, rank, shape, and
set names, so VAR_TO_HEADER in gempack_reference.py can be completed against the
REAL header inventory (never by guessing).

Usage:
    uv run python scripts/gtap/inspect_updated_har.py path/to/updated.har
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from equilibria.babel.har.reader import read_har  # noqa: E402


def inspect(har_path: str) -> None:
    headers = read_har(har_path)
    print(f"{har_path}\n{len(headers)} headers\n")
    print(f"{'HEADER':<10} {'RANK':<5} {'SHAPE':<18} {'SETS':<22} LONG_NAME")
    print("-" * 90)
    for name in sorted(headers):
        ha = headers[name]
        shape = "x".join(str(s) for s in ha.array.shape) or "scalar"
        sets = ", ".join(ha.set_names) if ha.set_names else "(none)"
        # sltoht SL4 dumps number headers 0001.. but keep the variable name in
        # long_name ("qfd # description #") — surface it so numbered headers are legible.
        long_name = (ha.long_name or "").strip()
        print(f"{name:<10} {ha.rank:<5} {shape:<18} {sets:<22} {long_name}")
    print(
        "\nMap ONLY headers that correspond cell-by-cell to a post-shock Var "
        "level.\nAdd them to VAR_TO_HEADER in scripts/gtap/gempack_reference.py.\n"
        "Aggregate-only headers: leave out (gempack_levels raises KeyError)."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: inspect_updated_har.py path/to/updated.har", file=sys.stderr)
        raise SystemExit(2)
    inspect(sys.argv[1])
