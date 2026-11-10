"""check_test_shards.py: guard for a test suite split across CI shards.

Two checks, both stdlib only:

``shards``: every test collected up front ran exactly once across the JUnit
files the shards produced, and none failed::

    python scripts/dev/check_test_shards.py shards --expected collected.txt junit-*.xml

``collected.txt`` has one pytest nodeid per line, as printed by
``pytest --collect-only -q -o addopts="" <same markers>`` (the repo's own
``addopts`` carries ``-q``, which turns the listing into ``file: count``; lines
that are not nodeids, such as the trailing summary, are ignored).

``compare``: a serial baseline run and a sharded run agree test by test, on
status AND skip/xfail reason (identity rule: even skipped -> passed is a
difference)::

    python scripts/dev/check_test_shards.py compare --baseline base.xml --current junit-*.xml

Exit status is 1 when any problem is found; problems are printed one per line.

Nodeid reconstruction.  pytest writes ``classname="tests.pkg.test_mod.TestCls"``
and ``name="test_x[param]"``.  Dots in a classname cannot be told apart from
path separators in general, so run pytest with ``-o junit_family=xunit1``: that
adds ``file="tests/pkg/test_mod.py"`` and the nodeid is exactly
``file::<classname minus the module part>::name``, the form ``--collect-only``
prints.  Without the ``file`` attribute a heuristic is used (the module path
ends at the first classname component that starts with an uppercase letter),
which fails only for test classes whose name does not start with one.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# (status, reason); status in passed, failed, error, skipped, xfailed, xpassed.
Outcome = tuple[str, str]


def _nodeid(case: ET.Element) -> str:
    classname = case.get("classname", "")
    name = case.get("name", "")
    file = case.get("file")
    parts = classname.split(".") if classname else []
    if file:
        module = file.removesuffix(".py").replace("/", ".")
        if classname == module:
            classes: list[str] = []
        elif classname.startswith(module + "."):
            classes = classname[len(module) + 1 :].split(".")
        else:  # unexpected shape: keep every component after the path
            classes = parts
        return "::".join([file, *classes, name])
    # Heuristic: the module path ends before the first CamelCase component.
    split = next((i for i, p in enumerate(parts) if p[:1].isupper()), len(parts))
    module_path = "/".join(parts[:split]) + ".py"
    return "::".join([module_path, *parts[split:], name])


def _outcome(case: ET.Element) -> Outcome:
    for child in case:
        message = child.get("message", "")
        if child.tag == "failure":
            return ("failed", message)
        if child.tag == "error":
            return ("error", message)
        if child.tag == "skipped":
            if child.get("type") == "pytest.xfail":
                return ("xfailed", message)
            return ("skipped", message)
    return ("passed", "")


def _is_module_entry(case: ET.Element) -> bool:
    """A whole module skipped or broken at collection, not a test: pytest writes
    it with an empty classname and the dotted module as name, once per shard."""
    file = case.get("file") or ""
    return not case.get("classname") and case.get("name") == file.removesuffix(
        ".py"
    ).replace("/", ".")


def _load(
    paths: list[Path],
) -> tuple[dict[str, Outcome], Counter[str], set[str]]:
    outcomes: dict[str, Outcome] = {}
    seen: Counter[str] = Counter()
    modules: set[str] = set()
    for path in paths:
        for case in ET.parse(path).getroot().iter("testcase"):
            nodeid = _nodeid(case)
            seen[nodeid] += 1
            outcomes[nodeid] = _outcome(case)
            if _is_module_entry(case):
                modules.add(nodeid)
    return outcomes, seen, modules


def load_junit(paths: list[Path]) -> dict[str, Outcome]:
    """Map pytest nodeid -> (status, reason) across the given JUnit files."""
    return _load(paths)[0]


def check_shards(expected: list[str], junit_paths: list[Path]) -> list[str]:
    """Problems with a sharded run; an empty list means it is complete and green."""
    outcomes, seen, modules = _load(junit_paths)
    problems: list[str] = []
    expected_set = set(expected)
    for nodeid in expected:
        if seen[nodeid] == 0:
            problems.append(f"missing: {nodeid} did not run in any shard")
    for nodeid, count in sorted(seen.items()):
        if nodeid in modules:  # repeats in every shard by design; failures below
            continue
        if count > 1:
            problems.append(f"more than once: {nodeid} ran {count} times")
        if nodeid not in expected_set:
            problems.append(f"unexpected: {nodeid} ran but was not collected")
    for nodeid, (status, reason) in sorted(outcomes.items()):
        if status in ("failed", "error"):
            problems.append(f"failed: {nodeid} ({status}) {reason}".rstrip())
    return problems


def compare_runs(baseline: dict[str, Outcome], current: dict[str, Outcome]) -> list[str]:
    """Any difference in status or reason, or presence on one side only."""
    problems: list[str] = []
    for nodeid in sorted(baseline.keys() | current.keys()):
        if nodeid not in current:
            problems.append(f"only in baseline: {nodeid} {baseline[nodeid]}")
        elif nodeid not in baseline:
            problems.append(f"only in current: {nodeid} {current[nodeid]}")
        elif baseline[nodeid] != current[nodeid]:
            problems.append(f"changed: {nodeid}: {baseline[nodeid]} -> {current[nodeid]}")
    return problems


def _read_expected(path: Path) -> list[str]:
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line for line in lines if "::" in line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    shards = sub.add_parser("shards", help="every expected test ran exactly once")
    shards.add_argument("--expected", type=Path, required=True)
    shards.add_argument("junit", nargs="*", type=Path)
    compare = sub.add_parser("compare", help="baseline run vs sharded run")
    compare.add_argument("--baseline", nargs="+", type=Path, required=True)
    compare.add_argument("--current", nargs="+", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "shards":
        problems = check_shards(_read_expected(args.expected), args.junit)
    else:
        problems = compare_runs(load_junit(args.baseline), load_junit(args.current))
    for problem in problems:
        print(problem)
    if problems:
        print(f"{len(problems)} problem(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
