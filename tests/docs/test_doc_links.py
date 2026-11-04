"""Links between documents resolve, and nothing points at a path phase 6 moves.

Phase 6 of ``specs/data/changes/one-data-folder`` moves most of ``docs/`` (to
``docs/archive/`` and ``docs/notes/``), ``GLOSSARY.md``/``ROADMAP.md`` into
``docs/`` and renames the last Spanish test files. A moved file leaves two kinds
of stale pointer behind:

- a relative Markdown link ``](../x.md)`` whose target no longer exists, which
  ``test_relative_links_resolve`` catches in every tracked ``*.md``;
- a path written as text (a docstring, a comment, a backticked path in prose),
  which ``test_no_mention_of_moved_paths`` catches with one pattern per moved
  family.

Both started with an allowance of the hits that existed when they were written;
each task of the phase shrank it, and the last task deleted it once empty.

The changes under ``specs/**/changes/`` (proposals, designs, plans) and
``docs/archive/`` are history, written then and not maintained, so neither test
reads them; ``CHANGELOG.md`` is history too. The rest of ``specs/`` (the
baselines, ``areas.toml``, the templates) is live and is read.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

HISTORY = ("docs/archive/", "CHANGELOG.md")
HISTORY_SEGMENTS = ("/changes/", "/archive/")
# Generators under data/ are fingerprinted by their source.yaml (ADR-0010): a
# comment edit there breaks the record, so the mention scan leaves them alone.
FINGERPRINTED = ("data/",)
# Text a person or a tool reads; data files (CSV, GDX, HAR, .gms) are not.
TEXT = (
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".yml",
    ".yaml",
    ".txt",
    ".cfg",
    ".sh",
    ".ps1",
    ".gitignore",
    "Makefile",
)

_FENCE = re.compile(r"^(```|~~~).*?^\1", re.MULTILINE | re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")

# One pattern per family of paths this phase moves (the "From" column of the
# plan). Relative links to them are the link test's job; these catch the text.
MOVED = {
    "docs-family": r"docs/(?:analysis|findings|plans|architecture|reference|technical|guides)/",
    "docs-loose": r"docs/(?:gtap_gams_mcp_parity_plan|ieem_to_pep_transformations)",
    "reports": r"(?<![\w./-])reports/(?:gdx_comparison_final_report|gdx_comparison_report|validation_report_final)",
    "glossary-root": r"GLOSSARY\.md",
    "roadmap-root": r"ROADMAP\.md",
    "renamed-doc": r"monolito_vs_bloques|registro_de_bloques\.md|gdx_reader_guide|mip_to_sam_guide_en",
    "renamed-test": (
        r"sin_rutas_fijas|test_sin_registro_de_bloques|_desempleo|default_es_bloques"
        r"|test_nus333_bienestar|me9_pasos"
    ),
}


def _tracked(pattern: str) -> list[str]:
    out = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z", pattern])
    return [p for p in out.decode("utf-8").split("\0") if p and not _history(p)]


def _history(path: str) -> bool:
    if path.startswith(HISTORY):
        return True
    return path.startswith("specs/") and any(s in path for s in HISTORY_SEGMENTS)


def _links(text: str) -> list[str]:
    text = _FENCE.sub("", text)
    text = _INLINE_CODE.sub("", text)
    return _LINK.findall(text)


def _local(target: str) -> str | None:
    if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE) or target.startswith(
        "#"
    ):
        return None
    return target.split("#", 1)[0] or None


def _broken_links() -> set[str]:
    broken = set()
    for rel in _tracked("*.md"):
        path = ROOT / rel
        for target in _links(path.read_text(encoding="utf-8")):
            local = _local(target)
            if local is None:
                continue
            base = ROOT if local.startswith("/") else path.parent
            if not (base / local.lstrip("/")).exists():
                broken.add(f"{rel}: {target}")
    return broken


def test_relative_links_resolve() -> None:
    broken = sorted(_broken_links())
    assert broken == [], "broken relative links:\n" + "\n".join(broken)


def _mentions() -> set[tuple[str, str]]:
    # These two name the old paths on purpose: the patterns here, and the
    # Spanish originals that the translation test reads from git.
    exempt = {
        Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "tests/docs/test_translations_keep_code.py",
    }
    hits = set()
    for rel in _tracked("*"):
        if rel in exempt or not rel.endswith(TEXT) or rel.startswith(FINGERPRINTED):
            continue
        try:
            text = (ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        for family, pattern in MOVED.items():
            if re.search(pattern, text):
                hits.add((rel, family))
    return hits


def test_no_mention_of_moved_paths() -> None:
    hits = sorted(_mentions())
    assert hits == [], f"text points at a moved path: {hits}"


def test_link_parser_sees_relative_and_skips_urls_and_code() -> None:
    text = (
        "[a](../x.md) [b](https://e.org) [c](#top) `[d](y.md)`\n```\n[e](z.md)\n```\n"
    )
    assert [t for t in _links(text) if _local(t)] == ["../x.md"]
