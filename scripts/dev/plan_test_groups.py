"""plan_test_groups.py: split a CI test suite into groups by FILE under a time budget.

Why by file: splitting test by test (pytest-split) scatters the tests of one
file over many groups, so every group repeats the module/session preparation
and imports and collects the whole suite.  Here a group is a list of whole
files; only a file longer than the budget is split, and only across groups
dedicated to that one file (run with pytest-split restricted to it).

Inputs (stdlib only, no pytest import):

* ``.github/test_durations.json``: seconds per pytest nodeid, measured on CI.
* ``.github/test_files_<suite>.txt``: the test files with at least one test
  selected by that suite's markers.  Regenerate with ``--refresh-files``.

Plan the matrix for a suite (one JSON line on stdout, for ``fromJSON``)::

    python scripts/dev/plan_test_groups.py --suite python --budget 360 --max-groups 10

Output: ``{"include": [{"group": ..., "paths": "a.py b.py", "split": "i k" or ""}]}``.
A non-empty ``split`` means: run ``paths`` (one file) with
``--splits k --group i --splitting-algorithm least_duration --durations-path ...``.

Regenerate the two file lists (runs pytest; use the project environment)::

    python scripts/dev/plan_test_groups.py --refresh-files
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GITHUB = ROOT / ".github"
NEW_FILE_SECONDS = 30.0

# Marker expressions per suite; they must match the ones in tests.yml.
SUITE_MARKERS = {
    "python": "not gams and not slow and not needs_gdxdump",
    "gdxdump": "needs_gdxdump and not gams and not slow",
}


def file_seconds(durations: dict[str, float]) -> dict[str, float]:
    """Sum of the test durations per test file (nodeid part before ``::``)."""
    totals: dict[str, float] = {}
    for nodeid, seconds in durations.items():
        file = nodeid.split("::", 1)[0]
        totals[file] = totals.get(file, 0.0) + seconds
    return totals


def _pack(files: list[tuple[str, float]], budget: float) -> list[list[tuple[str, float]]]:
    """LPT bin packing into the fewest groups whose heaviest is <= budget."""
    if not files:
        return []
    total = sum(s for _, s in files)
    for count in range(max(1, math.ceil(total / budget)), len(files) + 1):
        bins: list[list[tuple[str, float]]] = [[] for _ in range(count)]
        loads = [0.0] * count
        for item in files:  # already sorted by (-seconds, path)
            lightest = min(range(count), key=lambda i: (loads[i], i))
            bins[lightest].append(item)
            loads[lightest] += item[1]
        if max(loads) <= budget:
            return bins
    return [[item] for item in files]  # unreachable: each file <= budget


def plan_groups(
    files: list[str], seconds: dict[str, float], budget: float, max_groups: int
) -> list[dict[str, Any]]:
    """Groups ``{"name", "paths", "split"}`` covering every file exactly once."""
    sized = sorted(
        ((f, seconds.get(f, NEW_FILE_SECONDS)) for f in set(files)),
        key=lambda item: (-item[1], item[0]),
    )
    big = [(f, s) for f, s in sized if s > budget]
    small = [(f, s) for f, s in sized if s <= budget]
    stems = [Path(f).stem for f, _ in big]
    groups: list[dict[str, Any]] = []
    for path, secs in sorted(big):
        k = math.ceil(secs / budget)
        stem = Path(path).stem
        if stems.count(stem) > 1:
            stem = path.removesuffix(".py").replace("/", "_")
        for i in range(1, k + 1):
            groups.append({"name": f"{stem} {i}/{k}", "paths": [path], "split": [i, k]})
    packed = _pack(small, budget)
    for index, members in enumerate(packed, start=1):
        paths = sorted(f for f, _ in members)
        groups.append({"name": str(index), "paths": paths, "split": None})
    if len(groups) > max_groups:
        raise ValueError(
            f"{len(groups)} groups needed ({len(groups) - len(packed)} for big files, "
            f"{len(packed)} packed) but max_groups={max_groups} at budget={budget:g}s"
        )
    return groups


def predicted_seconds(group: dict[str, Any], seconds: dict[str, float]) -> float:
    total = sum(seconds.get(p, NEW_FILE_SECONDS) for p in group["paths"])
    return total / group["split"][1] if group["split"] else total


def _read_files(path: Path) -> list[str]:
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def refresh_files(tests_root: str = "tests") -> None:
    """Rewrite .github/test_files_<suite>.txt from a real pytest collection."""
    for suite, markers in SUITE_MARKERS.items():
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-o", "addopts=",
             "-q", "-m", markers, tests_root],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        nodeids = [ln for ln in result.stdout.splitlines() if "::" in ln]
        if result.returncode not in (0, 5) or not nodeids:
            sys.stderr.write(result.stdout[-2000:] + result.stderr[-2000:])
            raise SystemExit(f"collection failed for suite {suite}")
        files = sorted({n.split("::", 1)[0] for n in nodeids})
        out = GITHUB / f"test_files_{suite}.txt"
        out.write_text("".join(f + "\n" for f in files))
        print(f"{out.relative_to(ROOT)}: {len(files)} files, {len(nodeids)} tests")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--suite", choices=sorted(SUITE_MARKERS))
    parser.add_argument("--files-list", type=Path, help="override .github/test_files_<suite>.txt")
    parser.add_argument("--durations", type=Path, default=GITHUB / "test_durations.json")
    parser.add_argument("--budget", type=float, default=360.0)
    parser.add_argument("--max-groups", type=int, default=10)
    parser.add_argument("--refresh-files", action="store_true")
    parser.add_argument("--explain", action="store_true", help="print the plan to stderr")
    args = parser.parse_args(argv)

    if args.refresh_files:
        refresh_files()
        return 0
    listing = args.files_list or (GITHUB / f"test_files_{args.suite}.txt" if args.suite else None)
    if listing is None:
        parser.error("--suite or --files-list is required")
    seconds = file_seconds(json.loads(args.durations.read_text()))
    files = _read_files(listing)
    groups = plan_groups(files, seconds, args.budget, args.max_groups)
    if args.explain:
        for g in groups:
            sys.stderr.write(
                f"{g['name']:>34}  {predicted_seconds(g, seconds):7.1f}s  "
                f"{len(g['paths'])} file(s)  split={g['split']}\n"
            )
    include = [
        {
            "group": g["name"],
            "paths": " ".join(g["paths"]),
            "split": " ".join(map(str, g["split"])) if g["split"] else "",
        }
        for g in groups
    ]
    print(json.dumps({"include": include}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
