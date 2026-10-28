#!/usr/bin/env python3
"""Reject commit messages written in Spanish (maintainer's rule, 2026-10-28).

As a ``commit-msg`` hook it gets the message file; in CI it checks every
commit of a range (``--range origin/main..HEAD``). A message is Spanish when it
has two or more words of ``check_layout.SPANISH_WORDS`` or any accented word;
one borrowed word (``sin(x)``) does not fail it. Trailers, git's ``#`` lines and
code in backticks are not read.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "check_layout", _HERE / "check_layout.py"
)
assert _spec is not None and _spec.loader is not None
_layout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_layout)
SPANISH_WORDS: frozenset[str] = _layout.SPANISH_WORDS

_TRAILER = re.compile(r"^[A-Za-z-]+-by:", re.IGNORECASE)
_BACKTICKS = re.compile(r"`[^`]*`")
_WORD = re.compile(r"[^\W\d_]+")
_ACCENTED = re.compile("[ñáéíóúü]", re.IGNORECASE)


def spanish_words(message: str) -> list[str]:
    """The Spanish words of ``message`` when it reads as Spanish, else ``[]``."""
    lines = [
        _BACKTICKS.sub(" ", line)
        for line in message.splitlines()
        if not line.startswith("#") and not _TRAILER.match(line.strip())
    ]
    words = _WORD.findall(" ".join(lines).lower())
    listed = [w for w in words if w in SPANISH_WORDS]
    accented = [w for w in words if _ACCENTED.search(w)]
    if accented or len(listed) >= 2:
        return sorted(set(listed + accented))
    return []


def _range_messages(rev_range: str, repo: Path) -> list[tuple[str, str]]:
    out = subprocess.check_output(
        ["git", "-C", str(repo), "log", "--format=%h%x00%B%x01", rev_range], text=True
    )
    pairs = []
    for chunk in out.split("\x01"):
        if "\x00" in chunk:
            sha, body = chunk.split("\x00", 1)
            pairs.append((sha.strip(), body))
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("msg_file", nargs="?", type=Path)
    ap.add_argument("--range", dest="rev_range")
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    args = ap.parse_args(argv)

    if args.rev_range:
        messages = _range_messages(args.rev_range, args.repo)
    elif args.msg_file:
        messages = [("this commit", args.msg_file.read_text())]
    else:
        ap.error("give a message file or --range")

    bad = [(sha, msg, w) for sha, msg in messages if (w := spanish_words(msg))]
    for sha, msg, words in bad:
        first = msg.strip().splitlines()[0] if msg.strip() else ""
        print(f"{sha}: commit message is not English ({', '.join(words)}): {first}")
    if bad:
        print("Commit messages are written in English (see CONTRIBUTING.md).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
