"""skill_yaml.py: the stdlib reader of a skill's evals/prompts.yaml, and the
hash and stamp of a skill's folder (specs/2026-11-26-skills-uber-method-design.md).

The YAML subset: a list of flat maps whose values are strings or lists of
strings; plain, single- and double-quoted scalars; flow (``[a, b]``) and block
(``- a``) lists; ``#`` comments. Anything else raises ``SubsetError`` with its
line: it never gets a guessed parse.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

LAST_RUN = Path("evals") / "last_run.md"
_KEY = re.compile(r"^([A-Za-z_][\w-]*):(?:\s+(.*))?$")


class SubsetError(ValueError):
    def __init__(self, line: int, msg: str) -> None:
        super().__init__(f"line {line}: {msg}")
        self.line = line
        self.msg = msg


def _strip_comment(text: str) -> str:
    quote = None
    for i, ch in enumerate(text):
        if quote:
            if ch == "\\" and quote == '"':
                continue
            if ch == quote and (quote == "'" or i == 0 or text[i - 1] != "\\"):
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or text[i - 1].isspace()):
            return text[:i].rstrip()
    return text.rstrip()


def _scalar(raw: str, line: int) -> str:
    raw = raw.strip()
    if not raw:
        raise SubsetError(line, "empty value")
    if raw[0] in "|>&*!{":
        raise SubsetError(
            line, f"{raw[0]!r} (block scalar, anchor, tag or map) is outside the subset"
        )
    if raw[0] == '"':
        if len(raw) < 2 or raw[-1] != '"':
            raise SubsetError(line, "unclosed double quote")
        return re.sub(r'\\(["\\])', r"\1", raw[1:-1])
    if raw[0] == "'":
        if len(raw) < 2 or raw[-1] != "'":
            raise SubsetError(line, "unclosed single quote")
        return raw[1:-1].replace("''", "'")
    return raw


def _flow_list(raw: str, line: int) -> list[str]:
    inner = raw.strip()[1:-1]
    items, cur, quote = [], "", None
    for ch in inner:
        if quote:
            cur += ch
            if ch == quote and not cur.endswith("\\" + quote):
                quote = None
        elif ch in "'\"":
            quote = ch
            cur += ch
        elif ch == ",":
            items.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        items.append(cur)
    return [_scalar(i, line) for i in items]


def read_prompts(text: str) -> list[dict[str, str | list[str]]]:
    """The prompts of a prompts.yaml, as a list of flat maps."""
    out: list[dict[str, str | list[str]]] = []
    open_list: list[str] | None = None
    for n, original in enumerate(text.replace("\r\n", "\n").split("\n"), start=1):
        line = _strip_comment(original)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        body = line.strip()
        if indent == 0:
            if not body.startswith("- "):
                raise SubsetError(
                    n, "a top-level line must start a list item (- key: value)"
                )
            out.append({})
            body, open_list = body[2:].strip(), None
        elif open_list is not None and body.startswith("- ") and indent >= 4:
            open_list.append(_scalar(body[2:], n))
            continue
        elif indent != 2:
            raise SubsetError(
                n, "unexpected indentation (nested maps are outside the subset)"
            )
        if not out:
            raise SubsetError(n, "a key before the first list item")
        m = _KEY.match(body)
        if not m:
            raise SubsetError(n, f"not a key: value line: {body!r}")
        key, value = m.group(1), (m.group(2) or "").strip()
        if key in out[-1]:
            raise SubsetError(n, f"duplicate key {key!r}")
        if not value:
            open_list = []
            out[-1][key] = open_list
        elif value.startswith("["):
            if not value.endswith("]"):
                raise SubsetError(n, "a flow list must close on its line")
            out[-1][key], open_list = _flow_list(value, n), None
        else:
            out[-1][key], open_list = _scalar(value, n), None
    for item in out:
        for key, value in item.items():
            if value == []:
                raise SubsetError(0, f"{key!r} has no value")
    return out


def _skill_files(folder: Path) -> list[Path]:
    """The files git would carry for the folder (tracked, or untracked and not
    ignored); outside a git repo, every file. Dotfiles (a Finder .DS_Store) never."""
    out = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z", "."],
        cwd=folder,
        capture_output=True,
    )
    if out.returncode == 0:
        files = [folder / p for p in out.stdout.decode().split("\0") if p]
    else:
        files = [p for p in folder.rglob("*") if p.is_file()]
    return sorted(
        p
        for p in files
        if p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(folder).parts)
    )


def skill_hash(folder: Path) -> str:
    """sha256 over the sorted relative paths and bytes of the skill's files
    (``_skill_files``) but ``evals/last_run.md`` and ``__pycache__``."""
    h = hashlib.sha256()
    for path in _skill_files(folder):
        rel = path.relative_to(folder)
        if rel == LAST_RUN or "__pycache__" in rel.parts:
            continue
        h.update(rel.as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def read_last_run(path: Path) -> dict[str, str]:
    """The ``key: value`` lines of a skill's evals/last_run.md."""
    out = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        key, sep, value = line.partition(":")
        if sep and key.strip() and " " not in key.strip():
            out[key.strip()] = value.strip()
    return out
