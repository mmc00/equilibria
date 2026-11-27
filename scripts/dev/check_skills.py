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

from skill_yaml import LAST_RUN, SubsetError, read_last_run, read_prompts, skill_hash

ROOT = Path(__file__).resolve().parents[2]
SKILLS = Path(".agents") / "skills"
PENDING = Path("scripts") / "dev" / "skills_pending.txt"
MAX_DESCRIPTION_WORDS = 100
MAX_BODY_LINES = 500
MIN_PROMPTS = 5
MIN_PASS_RATE = 0.70
PROMPT_KEYS = ("id", "kind", "prompt", "expect")
PROMPT_KINDS = ("golden", "trajectory", "author")


@dataclass(frozen=True)
class Finding:
    skill: str
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.skill}: {self.rule}: {self.detail}"


def read_text(path: Path) -> str:
    """The file as text, without a UTF-8 BOM and with LF line ends."""
    return path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")


_BLOCK_SCALAR = re.compile(r"^[>|][+-]?$")


def _frontmatter(lines: list[str]) -> dict[str, str]:
    """``key: value`` pairs; a ``>`` or ``|`` value takes the indented lines
    under it, joined by spaces (``>``) or newlines (``|``)."""
    meta: dict[str, str] = {}
    i = 0
    while i < len(lines):
        key, sep, value = lines[i].partition(":")
        i += 1
        if not sep or lines[i - 1][:1].isspace():
            continue
        value = value.strip()
        if _BLOCK_SCALAR.match(value):
            block = []
            while i < len(lines) and (not lines[i].strip() or lines[i][:1].isspace()):
                block.append(lines[i].strip())
                i += 1
            value = (" " if value[0] == ">" else "\n").join(b for b in block if b)
        meta[key.strip()] = value
    return meta


def split_skill(text: str) -> tuple[dict[str, str] | None, str]:
    """(frontmatter as ``key: value`` pairs, body); None when there is none."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return None, text
    for end in range(1, len(lines)):
        if lines[end].strip() == "---":
            return _frontmatter(lines[1:end]), "\n".join(lines[end + 1 :])
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
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
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
    """Each line with whether it is inside a fenced block (fence lines count as in).
    As in CommonMark, a fence closes only on a bare run of its own character at
    least as long as the one that opened it, so a ```` block may hold ``` ones."""
    out: list[tuple[str, bool]] = []
    opened: str | None = None
    for line in text.split("\n"):
        m = _FENCE.match(line)
        if opened is None:
            if m:
                opened = m.group(1)
            out.append((line, opened is not None))
        else:
            out.append((line, True))
            run = m.group(1) if m else ""
            if (
                run[:1] == opened[0]
                and len(run) >= len(opened)
                and not line.strip()[len(run) :].strip()
            ):
                opened = None
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
        for line, fenced in _outside_fences(text)
        if not fenced and line.strip().startswith("|")
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
        code = _code(text) if doc.suffix == ".md" else text.split("\n")
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


def rule_evals(root: Path, name: str) -> list[Finding]:
    path = _folder(root, name) / "evals" / "prompts.yaml"
    if not path.is_file():
        return [Finding(name, "evals", "evals/prompts.yaml is missing")]
    try:
        prompts = read_prompts(read_text(path))
    except SubsetError as err:
        return [Finding(name, "evals", f"prompts.yaml {err}")]
    out = []
    if len(prompts) < MIN_PROMPTS:
        out.append(
            Finding(name, "evals", f"{len(prompts)} prompt(s) (min {MIN_PROMPTS})")
        )
    kinds = [p.get("kind") for p in prompts]
    for kind in ("golden", "trajectory"):
        if kind not in kinds:
            out.append(Finding(name, "evals", f"no {kind} prompt"))
    for i, p in enumerate(prompts, start=1):
        pid = p.get("id", f"#{i}")
        missing = [k for k in PROMPT_KEYS if not p.get(k)]
        if missing:
            out.append(
                Finding(name, "evals", f"prompt {pid} lacks {', '.join(missing)}")
            )
        if p.get("kind") and p.get("kind") not in PROMPT_KINDS:
            out.append(
                Finding(
                    name, "evals", f"prompt {pid} has unknown kind {p.get('kind')!r}"
                )
            )
        if p.get("kind") == "trajectory" and not p.get("must_run"):
            out.append(
                Finding(name, "evals", f"trajectory prompt {pid} has no must_run")
            )
    return out


def _pass_rate(root: Path, name: str) -> float | None:
    path = _folder(root, name) / LAST_RUN
    try:
        return float(read_last_run(path).get("pass_rate", ""))
    except (OSError, ValueError):
        return None


def rule_evals_run(root: Path, name: str) -> list[Finding]:
    folder = _folder(root, name)
    path = folder / LAST_RUN
    if not path.is_file():
        return [
            Finding(
                name,
                "evals_run",
                f"{LAST_RUN} is missing: run scripts/dev/run_skill_evals.py {name}",
            )
        ]
    stamp = read_last_run(path)
    out = []
    if stamp.get("skill_hash") != skill_hash(folder):
        out.append(
            Finding(
                name,
                "evals_run",
                f"{LAST_RUN} is stale: the skill changed since its last eval run",
            )
        )
    rate = _pass_rate(root, name)
    if rate is None:
        out.append(Finding(name, "evals_run", f"{LAST_RUN} has no pass_rate"))
    elif rate < MIN_PASS_RATE:
        out.append(
            Finding(
                name, "evals_run", f"pass rate {rate:.0%} is under {MIN_PASS_RATE:.0%}"
            )
        )
    return out


def notes(root: Path, name: str) -> list[str]:
    """What is allowed but worth a look: a pass rate of 100% may mean easy prompts."""
    if _pass_rate(root, name) == 1.0:
        return [f"{name}: evals_run: pass rate 100%; the prompts may be too easy"]
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
    rule_steps,
    rule_errors,
    rule_spec,
    rule_links,
    rule_portable,
    rule_evals,
    rule_evals_run,
    rule_wiring,
]


TEXT_FILES = ("SKILL.md", "spec.md", "evals/prompts.yaml", str(LAST_RUN))


def rule_encoding(root: Path, name: str) -> list[Finding]:
    """The files the rules read must be UTF-8; the rules do not run otherwise."""
    folder = _folder(root, name)
    out = []
    for rel in TEXT_FILES:
        path = folder / rel
        if path.is_file():
            try:
                read_text(path)
            except UnicodeDecodeError:
                out.append(Finding(name, "encoding", f"{rel} is not UTF-8"))
    return out


def check_skill(root: Path, name: str) -> list[Finding]:
    """Every rule's findings for the skill ``name`` under ``root``."""
    if not (_folder(root, name) / "SKILL.md").is_file():
        return [
            Finding(name, "frontmatter", f"{SKILLS / name / 'SKILL.md'} is missing")
        ]
    return rule_encoding(root, name) or [f for rule in RULES for f in rule(root, name)]


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
        for note in notes(root, name):
            print(f"note: {note}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
