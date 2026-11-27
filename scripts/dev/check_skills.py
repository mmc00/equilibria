"""check_skills.py: hold the repo's agent skills to Uber's skills method.

The method and the rules: specs/2026-11-26-skills-uber-method-design.md.

    python scripts/dev/check_skills.py              # every skill, plus the pending ones
    python scripts/dev/check_skills.py port-to-jax  # only the named skills

Exits 1 with one ``<skill>: <rule>: <detail>`` line per finding. A skill listed
in ``scripts/dev/skills_pending.txt`` (not migrated yet) only gets warnings, and
must leave that list once it passes. Stdlib only: prek runs it with no project
install.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILLS = Path(".agents") / "skills"
PENDING = Path("scripts") / "dev" / "skills_pending.txt"
MAX_DESCRIPTION_WORDS = 100
MAX_BODY_LINES = 500


@dataclass(frozen=True)
class Finding:
    skill: str
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.skill}: {self.rule}: {self.detail}"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def split_skill(text: str) -> tuple[dict[str, str] | None, str]:
    """(frontmatter as ``key: value`` pairs, body); None when there is none."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return None, text
    for end in range(1, len(lines)):
        if lines[end].strip() == "---":
            meta = {}
            for line in lines[1:end]:
                key, sep, value = line.partition(":")
                if sep:
                    meta[key.strip()] = value.strip()
            return meta, "\n".join(lines[end + 1 :])
    return None, text


def _folder(root: Path, name: str) -> Path:
    return root / SKILLS / name


def rule_frontmatter(root: Path, name: str) -> list[Finding]:
    meta, _ = split_skill(read_text(_folder(root, name) / "SKILL.md"))
    if meta is None:
        return [Finding(name, "frontmatter", "SKILL.md has no --- frontmatter")]
    out = []
    if meta.get("name") != name:
        out.append(
            Finding(
                name,
                "frontmatter",
                f"name {meta.get('name')!r} is not the folder {name!r}",
            )
        )
    desc = meta.get("description", "")
    if not desc:
        out.append(Finding(name, "frontmatter", "description is missing"))
    else:
        words = len(desc.split())
        if words > MAX_DESCRIPTION_WORDS:
            out.append(
                Finding(
                    name,
                    "frontmatter",
                    f"description has {words} words (max {MAX_DESCRIPTION_WORDS})",
                )
            )
        if "use when" not in desc.lower():
            out.append(
                Finding(
                    name,
                    "frontmatter",
                    'description does not say when to use it ("Use when")',
                )
            )
    return out


def rule_length(root: Path, name: str) -> list[Finding]:
    _, body = split_skill(read_text(_folder(root, name) / "SKILL.md"))
    n = len(body.splitlines())
    if n >= MAX_BODY_LINES:
        return [
            Finding(
                name, "length", f"body has {n} lines (must be under {MAX_BODY_LINES})"
            )
        ]
    return []


def rule_wiring(root: Path, name: str) -> list[Finding]:
    out = []
    link = root / ".claude" / "skills" / name
    if not (link.exists() and link.resolve() == _folder(root, name).resolve()):
        out.append(
            Finding(
                name,
                "wiring",
                f".claude/skills/{name} does not link to {SKILLS / name}",
            )
        )
    gi = root / ".gitignore"
    if f"!/.claude/skills/{name}" not in (
        read_text(gi).split("\n") if gi.exists() else []
    ):
        out.append(Finding(name, "wiring", f".gitignore lacks !/.claude/skills/{name}"))
    readme = root / SKILLS / "README.md"
    if f"({name}/SKILL.md)" not in (read_text(readme) if readme.exists() else ""):
        out.append(
            Finding(name, "wiring", f"{SKILLS / 'README.md'} has no row for {name}")
        )
    return out


RULES: list[Callable[[Path, str], list[Finding]]] = [
    rule_frontmatter,
    rule_length,
    rule_wiring,
]


def check_skill(root: Path, name: str) -> list[Finding]:
    """Every rule's findings for the skill ``name`` under ``root``."""
    if not (_folder(root, name) / "SKILL.md").is_file():
        return [
            Finding(name, "frontmatter", f"{SKILLS / name / 'SKILL.md'} is missing")
        ]
    return [f for rule in RULES for f in rule(root, name)]


def load_pending(root: Path) -> set[str]:
    path = root / PENDING
    if not path.exists():
        return set()
    names = (line.split("#", 1)[0].strip() for line in read_text(path).split("\n"))
    return {n for n in names if n}


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    args = sys.argv[1:] if argv is None else argv
    pending = load_pending(root)
    if args:
        names = list(dict.fromkeys(args))
    else:
        base = root / SKILLS
        here = (
            sorted(p.name for p in base.iterdir() if p.is_dir())
            if base.is_dir()
            else []
        )
        names = here + sorted(pending - set(here))
    bad = False
    for name in names:
        if name in pending and not (_folder(root, name)).is_dir():
            print(f"warning: {name}: pending: not in the repo yet")
            continue
        findings = check_skill(root, name)
        if name in pending:
            if not findings:
                print(f"{name}: pending: passes every rule; remove it from pending")
                bad = True
            for f in findings:
                print(f"warning: {f}")
        else:
            for f in findings:
                print(f)
            bad = bad or bool(findings)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
