#!/usr/bin/env python3
"""Claude Code PostToolUse hook: run the skills checker on the skill a Write or
Edit just touched (specs/2026-11-26-skills-uber-method-design.md).

A PostToolUse hook cannot undo the write, so its findings are feedback:
- a skill not on scripts/dev/skills_pending.txt: exit 2, findings on stderr, so
  Claude fixes them in the same turn;
- a pending skill: exit 0, findings as additionalContext.
Anything else (another file, a path outside the repo, bad input) exits 0 silently.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEV))

from check_skills import ROOT, check_skill, load_pending  # noqa: E402

PREFIXES = ((".agents", "skills"), (".claude", "skills"))


def skill_of(path: str, root: Path) -> str | None:
    """The skill a file belongs to, or None when it is not under a skill folder."""
    try:
        rel = Path(path).resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        try:
            rel = Path(path).relative_to(root)
        except ValueError:
            return None
    parts = rel.parts
    if len(parts) >= 3 and parts[:2] in PREFIXES:
        return parts[2]
    return None


def run(payload: dict, root: Path) -> tuple[int, str]:
    """(exit code, text to print) for one hook payload."""
    path = (payload.get("tool_input") or {}).get("file_path")
    if not isinstance(path, str):
        return 0, ""
    name = skill_of(path, root)
    if name is None or name == "README.md":
        return 0, ""
    findings = check_skill(root, name)
    if not findings:
        return 0, ""
    lines = "\n".join(str(f) for f in findings)
    if name in load_pending(root):
        context = f"Skill {name} is pending migration; it does not meet the method yet:\n{lines}"
        return 0, json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": context,
                }
            }
        )
    return (
        2,
        f"Skill {name} does not meet the method (scripts/dev/check_skills.py):\n{lines}",
    )


def main(root: Path = ROOT) -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    if not isinstance(payload, dict):
        return 0
    code, text = run(payload, root)
    if text:
        print(text, file=sys.stderr if code == 2 else sys.stdout)
    return code


if __name__ == "__main__":
    sys.exit(main())
