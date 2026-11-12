"""Claude Code PreToolUse hook: block `git push` / `gh pr create` without fresh gates.

Wired in `.claude/settings.json` on the Bash matcher. Claude Code invokes this
script BEFORE every Bash tool call, passing the pending call as JSON on stdin:

    {"hook_event_name": "PreToolUse", "tool_name": "Bash",
     "tool_input": {"command": "git push origin main", ...}, ...}

Contract (Claude Code hook exit codes):
    exit 0 -> allow the tool call
    exit 2 -> BLOCK the tool call; stderr is fed back to Claude, which then
              knows to run scripts/gtap/run_parity_gates.py first
    other  -> non-blocking error (shown, call proceeds)

The actual staleness logic lives in scripts/gtap/check_parity_gates_stamp.py
(stdlib-only, <50 ms): pushes that do not change gate-relevant GTAP trees vs
origin/main are always allowed, so non-GTAP work is never blocked.

Escape hatch (emergencies only): GTAP_GATES_SKIP=1 in the environment.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Only match at command position (start of line or after ; & | ` ( ) so the
# literal text "git push" inside an echo/string does not trip the guard.
GUARDED = re.compile(
    r"(?:^|[;&|`(\n]\s*)(?:git\s+(?:[\w.-]+\s+)*push\b|gh\s+pr\s+(?:create|merge)\b)"
)


def target_dir(command: str, cwd: str) -> Path:
    """The directory the push runs in: the path of ``git -C <path>`` or of a leading
    ``cd <path>``, else ``cwd``. A session started in one worktree pushes another one
    that way, and the stamp to check is that worktree's, not the hook's own."""
    import shlex

    base = Path(cwd)
    try:
        words = shlex.split(command)
    except ValueError:
        return base
    for i, w in enumerate(words[:-1]):
        if w == "git" and words[i + 1] == "-C" and i + 2 < len(words):
            return base / words[i + 2]
    if len(words) > 1 and words[0] == "cd":
        return base / words[1]
    return base


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # malformed payload — never block on our own bug

    if payload.get("tool_name") != "Bash":
        return 0
    command = (payload.get("tool_input") or {}).get("command", "")
    if not GUARDED.search(command):
        return 0
    if os.environ.get("GTAP_GATES_SKIP") == "1":
        print("GTAP_GATES_SKIP=1 — parity-gates check bypassed", file=sys.stderr)
        return 0

    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "scripts/gtap"))
    try:
        from check_parity_gates_stamp import check, repo_root

        cwd = payload.get("cwd") or os.getcwd()
        ok, msg = check(repo_root(target_dir(command, cwd)))
    except Exception as exc:  # checker broken — fail open, but say so
        print(f"parity-gates hook: checker error ({exc}) — allowing", file=sys.stderr)
        return 0

    if ok:
        return 0
    print(
        "BLOCKED: mandatory local GTAP parity gates (NLP-vs-NLP + MCP-vs-MCP) "
        "are not fresh for this push.\n" + msg +
        "\nRun: uv run python scripts/gtap/run_parity_gates.py  (then commit any "
        "regenerated docs and retry).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
