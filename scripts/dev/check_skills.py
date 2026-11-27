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


COMMAND_PREFIXES = ("uv run", "git ", "gh ", "python3 ", "bash ", "scripts/")
SPEC_HEADINGS = ("Goal", "Outputs", "Tools", "Forbidden", "Agents")
_FENCE = re.compile(r"^\s*(```|~~~)")
_STEP = re.compile(r"^\d+\. ")
_INLINE = re.compile(r"`([^`\n]+)`")
_RETRY_ONLY = re.compile(r"^(try again|retry|re-?run( it| the command)?)\.?$")
_LINK = re.compile(r"\]\(([^)\s]+)\)")
_NOT_PORTABLE = (
    (re.compile(r"\.venv/bin/python"), ".venv/bin/python"),
    (re.compile(r"(?<!uv run )(?<![\w/.])python (?!3)"), "a bare python"),
    (re.compile(r"/Users/"), "an absolute /Users/ path"),
)


def _outside_fences(text: str) -> list[tuple[str, bool]]:
    """Each line with whether it is inside a fenced block (fence lines count as in)."""
    out, inside = [], False
    for line in text.split("\n"):
        if _FENCE.match(line):
            out.append((line, True))
            inside = not inside
        else:
            out.append((line, inside))
    return out


def section(body: str, heading: str) -> str | None:
    """The text under ``## <heading>`` up to the next ``## `` heading; headings
    inside fences do not count. None when there is no such heading."""
    lines, found = [], False
    for line, fenced in _outside_fences(body):
        if not fenced and line.startswith("## "):
            if found:
                break
            found = line[3:].strip().lower() == heading.lower()
            continue
        if found:
            lines.append(line)
    return "\n".join(lines) if found else None


def _code(text: str) -> list[str]:
    """Fenced lines and inline code spans of ``text``."""
    out = []
    for line, fenced in _outside_fences(text):
        if fenced:
            if not _FENCE.match(line):
                out.append(line)
        else:
            out.extend(_INLINE.findall(line))
    return out


def _is_command(code: str, fenced: bool) -> bool:
    return fenced or code.strip().startswith(COMMAND_PREFIXES)


def rule_steps(root: Path, name: str) -> list[Finding]:
    _, body = split_skill(read_text(_folder(root, name) / "SKILL.md"))
    text = section(body, "Steps")
    if text is None:
        return [Finding(name, "steps", "no ## Steps section")]
    steps: list[list[tuple[str, bool]]] = []
    for line, fenced in _outside_fences(text):
        if not fenced and _STEP.match(line):
            steps.append([])
        if steps:
            steps[-1].append((line, fenced))
    if not steps:
        return [Finding(name, "steps", "## Steps has no numbered step")]
    out = []
    for n, lines in enumerate(steps, start=1):
        fenced_code = any(
            f and not _FENCE.match(line) and line.strip() for line, f in lines
        )
        inline = [c for line, f in lines if not f for c in _INLINE.findall(line)]
        if not (fenced_code or any(_is_command(c, False) for c in inline)):
            out.append(Finding(name, "steps", f"step {n} has no command"))
        checks = [line for line, f in lines if not f and "Check:" in line]
        if not checks or all(not c.split("Check:", 1)[1].strip() for c in checks):
            out.append(
                Finding(name, "steps", f"step {n} has no Check: line with content")
            )
    return out


def rule_errors(root: Path, name: str) -> list[Finding]:
    _, body = split_skill(read_text(_folder(root, name) / "SKILL.md"))
    text = section(body, "Errors")
    if text is None:
        return [Finding(name, "errors", "no ## Errors section")]
    rows = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in text.split("\n")
        if line.strip().startswith("|")
    ]
    rows = [r for r in rows[1:] if not all(set(c) <= set("-: ") for c in r)]
    out = []
    if len(rows) < 2:
        out.append(
            Finding(name, "errors", f"the errors table has {len(rows)} row(s) (min 2)")
        )
    for r in rows:
        if len(r) >= 2 and _RETRY_ONLY.match(r[-1].lower().strip()):
            out.append(Finding(name, "errors", f"fix {r[-1]!r} only retries"))
    return out


def rule_spec(root: Path, name: str) -> list[Finding]:
    path = _folder(root, name) / "spec.md"
    if not path.is_file():
        return [Finding(name, "spec", "spec.md is missing")]
    text = read_text(path)
    out = []
    for heading in SPEC_HEADINGS:
        body = section(text, heading)
        if body is None or not body.strip():
            out.append(
                Finding(name, "spec", f"spec.md has no content under ## {heading}")
            )
    return out


def _skill_docs(folder: Path) -> list[Path]:
    docs = [folder / "SKILL.md"]
    for sub in ("references", "scripts"):
        if (folder / sub).is_dir():
            docs += sorted(p for p in (folder / sub).rglob("*") if p.is_file())
    return docs


def rule_links(root: Path, name: str) -> list[Finding]:
    folder = _folder(root, name)
    out = []
    for doc in [folder / "SKILL.md", folder / "spec.md"]:
        if not doc.is_file():
            continue
        for target in _LINK.findall(read_text(doc)):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            path = target.split("#", 1)[0]
            if path and not (doc.parent / path).exists():
                out.append(
                    Finding(name, "links", f"{doc.name} links to missing {path}")
                )
    return out


def rule_portable(root: Path, name: str) -> list[Finding]:
    folder = _folder(root, name)
    out = []
    for doc in _skill_docs(folder):
        try:
            text = read_text(doc)
        except UnicodeDecodeError:
            continue
        code = text.split("\n") if doc.suffix not in (".md", "") else _code(text)
        for snippet in code:
            for pattern, what in _NOT_PORTABLE:
                if pattern.search(snippet):
                    rel = doc.relative_to(folder)
                    out.append(
                        Finding(
                            name,
                            "portable",
                            f"{rel} uses {what}: {snippet.strip()[:60]}",
                        )
                    )
    return out


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
    rule_steps,
    rule_errors,
    rule_spec,
    rule_links,
    rule_portable,
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
