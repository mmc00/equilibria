"""Fast staleness check for the mandatory local GTAP parity gates (stdlib-only).

The full NLP-vs-NLP + MCP-vs-MCP integration gates (`run_parity_gates.py`) are
LOCAL-ONLY (they need the PATH C-API and the fixture GDXs; CI cannot run them).
This checker is the cheap (<50 ms) guard that the Claude PreToolUse hook and
humans use before `git push` / `gh pr create`:

  - It hashes the git TREES of every path whose content can change gate results
    (model source, scripts, gate tests, reference fixtures). Doc-only commits do
    not move this hash, so committing regenerated matrices never invalidates a
    fresh stamp.
  - If the hash at HEAD equals the hash at origin/main, the push contains no
    gate-relevant GTAP change and no stamp is required.
  - Otherwise `.git/gtap-parity-gates.stamp` (written by run_parity_gates.py on
    a fully green run) must exist and match HEAD's hash, and the watched paths
    plus the generated docs must be clean in the working tree.

Exit 0 = ok to push. Exit 1 = stale (reason on stdout).
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

# Paths whose tracked content determines gate results.
INPUT_TREES = [
    "src/equilibria/templates/gtap",
    "scripts/gtap",
    "tests/templates/gtap",
    "tests/fixtures/gtap7",
    "tests/fixtures/gtap7_altertax",
]

# Generated docs that must be committed in sync with a gate run.
GENERATED_DOCS = [
    "docs/site/guide/gtap7_coverage_matrix.md",
    "docs/site/_static/gtap7_mcp_matrix.html",
    "docs/site/_static/gtap7_nlp_matrix.html",
]

STAMP_NAME = "gtap-parity-gates.stamp"


def stamp_path(repo: Path) -> Path:
    """Per-worktree stamp location. In a linked worktree `.git` is a FILE
    (gitdir pointer), so resolve the real git dir via rev-parse."""
    git_dir = _git(repo, "rev-parse", "--git-dir") or ".git"
    p = Path(git_dir)
    if not p.is_absolute():
        p = repo / p
    return p / STAMP_NAME


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if out.returncode != 0:
        return ""
    return out.stdout.strip()


def repo_root(start: Path | None = None) -> Path:
    top = _git(start or Path.cwd(), "rev-parse", "--show-toplevel")
    return Path(top) if top else Path.cwd()


def input_hash(repo: Path, rev: str = "HEAD") -> str:
    parts = []
    for tree in INPUT_TREES:
        parts.append(_git(repo, "rev-parse", f"{rev}:{tree}") or "-")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def dirty_watched(repo: Path) -> list[str]:
    out = _git(repo, "status", "--porcelain", "--", *INPUT_TREES, *GENERATED_DOCS)
    return [line for line in out.splitlines() if line.strip()]


def check(repo: Path | None = None) -> tuple[bool, str]:
    repo = repo or repo_root()
    dirty = dirty_watched(repo)
    if dirty:
        return False, (
            "gate-relevant paths have uncommitted changes (commit first, "
            "then run the gates):\n  " + "\n  ".join(dirty[:10])
        )
    head_hash = input_hash(repo, "HEAD")
    main_hash = input_hash(repo, "origin/main")
    if head_hash == main_hash:
        return True, "no gate-relevant GTAP changes vs origin/main — stamp not required"
    sp = stamp_path(repo)
    if not sp.exists():
        return False, (
            "GTAP gate-relevant changes vs origin/main but no parity-gates stamp.\n"
            "Run: uv run python scripts/gtap/run_parity_gates.py"
        )
    stamp = sp.read_text().split()[0] if sp.read_text().strip() else ""
    if stamp != head_hash:
        return False, (
            "parity-gates stamp is STALE (gate-relevant content changed since the "
            "last green run).\nRun: uv run python scripts/gtap/run_parity_gates.py"
        )
    return True, "parity-gates stamp fresh for HEAD"


def main() -> int:
    ok, msg = check()
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
