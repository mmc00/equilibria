#!/usr/bin/env python3
"""Enforce lowercase filename policy with explicit legacy exceptions.

This check is intentionally strict for new files but keeps a small whitelist
for legacy/contractual names (e.g. GAMS reference artifacts).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# Files that are intentionally kept with canonical uppercase naming.
ALLOWED_EXACT: set[str] = {
    "AGENTS.md",
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "skills/gams-to-equilibria/SKILL.md",
    "src/equilibria/babel/gdx/README.md",
    "tests/babel/gdx/fixtures/SAM-V2_0.gdx",
}

# Directories where source/reference artifacts must keep upstream naming.
ALLOWED_PREFIXES: tuple[str, ...] = (
    "src/equilibria/templates/data/pep/",
    "src/equilibria/templates/reference/pep/",
    "src/equilibria/templates/reference/pep2/data/",
    "src/equilibria/templates/reference/pep2/scripts/",
)


def _tracked_files() -> list[str]:
    out = subprocess.check_output(["git", "ls-files"], text=True)
    return [line.strip() for line in out.splitlines() if line.strip()]


def _has_uppercase(name: str) -> bool:
    return any(ch.isalpha() and ch.isupper() for ch in name)


def _is_allowed(path: str) -> bool:
    if path in ALLOWED_EXACT:
        return True
    return any(path.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def _violations(paths: list[str]) -> list[str]:
    bad: list[str] = []
    for p in paths:
        basename = Path(p).name
        if not _has_uppercase(basename):
            continue
        if _is_allowed(p):
            continue
        bad.append(p)
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that tracked filenames are lowercase unless whitelisted.",
    )
    parser.add_argument(
        "--show-allowed-uppercase",
        action="store_true",
        help="Print tracked files that contain uppercase but are allowed.",
    )
    args = parser.parse_args()

    files = _tracked_files()
    violations = _violations(files)

    if args.show_allowed_uppercase:
        allowed = []
        for p in files:
            basename = Path(p).name
            if _has_uppercase(basename) and _is_allowed(p):
                allowed.append(p)
        print("Allowed uppercase tracked files:")
        for p in sorted(allowed):
            print(f"  - {p}")
        print(f"Total allowed uppercase files: {len(allowed)}")
        print()

    if violations:
        print("Lowercase filename policy failed. Non-whitelisted uppercase filenames:")
        for p in sorted(violations):
            print(f"  - {p}")
        print()
        print("Rename these files to lowercase or add a justified whitelist entry in:")
        print("  scripts/check_lowercase_filenames.py")
        return 1

    print("Lowercase filename policy passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
