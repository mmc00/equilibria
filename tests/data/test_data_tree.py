"""The repository's data/ is clean: every file has a verified source (ADR-0010).

A branch whose phase is still open may set EQUILIBRIA_ALLOW_PENDING=1; CI on
main never does.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from equilibria.sources import DATA_ROOT, check_tree, iter_sources


def test_data_tree_is_clean() -> None:
    allow = os.environ.get("EQUILIBRIA_ALLOW_PENDING") == "1"
    assert check_tree(allow_pending=allow) == []


def test_record_outputs_are_not_gitignored() -> None:
    # a record whose output git ignores is verified on the machine that ran it
    # and broken everywhere else (*.lst / *.log hid the NEOS listings)
    repo = DATA_ROOT.parent
    git = shutil.which("git")
    if git is None or not (repo / ".git").exists():
        pytest.skip("not a git checkout")
    outputs = [
        str(rec.path / rel)
        for rec in iter_sources()
        for rel in (rec.raw.get("outputs") or {})
    ]
    ignored = subprocess.run(
        [str(git), "check-ignore", "--stdin"],
        input="\n".join(outputs), capture_output=True, text=True, cwd=repo,
    ).stdout.split()  # fmt: skip
    assert ignored == []


def test_git_keeps_the_bytes_of_every_hashed_file() -> None:
    # a source.yaml hashes the exact bytes. With core.autocrlf, git stored the
    # Windows GEMPACK consoles with LF while their record had hashed CRLF, so all
    # 66 records failed on any other machine: git must not touch data/'s bytes
    repo = DATA_ROOT.parent
    git = shutil.which("git")
    if git is None or not (repo / ".git").exists():
        pytest.skip("not a git checkout")
    hashed = [
        str((rec.path / rel).relative_to(repo))
        for rec in iter_sources()
        for rel in (rec.raw.get("outputs") or {})
    ]
    hashed += [
        str(p.relative_to(repo)) for p in (DATA_ROOT / "gtap7" / "gams").glob("*.gms")
    ]
    attrs = subprocess.run(
        [str(git), "check-attr", "text", "--stdin"],
        input="\n".join(hashed), capture_output=True, text=True, cwd=repo,
    ).stdout.splitlines()  # fmt: skip
    converted = [line for line in attrs if not line.endswith(": text: unset")]
    assert len(attrs) == len(hashed)
    assert converted == [], converted[:3]
