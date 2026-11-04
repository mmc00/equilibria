"""Links between documents resolve, and nothing points at a path phase 6 moves.

Phase 6 of ``specs/data/changes/one-data-folder`` moves most of ``docs/`` (to
``docs/archive/`` and ``docs/notes/``), ``GLOSSARY.md``/``ROADMAP.md`` into
``docs/`` and renames the last Spanish test files. A moved file leaves two kinds
of stale pointer behind:

- a relative Markdown link ``](../x.md)`` whose target no longer exists, which
  ``test_relative_links_resolve`` catches in every tracked ``*.md``;
- a path written as text (a docstring, a comment, a backticked path in prose),
  which ``test_no_mention_of_moved_paths`` catches with one pattern per moved
  family. ``MENTION_ALLOWANCE`` holds the hits that existed when the test was
  written; each task of the phase shrinks it, and an entry that no longer hits
  fails, so the allowance can only go down.

``specs/`` and ``docs/archive/`` are history, written then and not maintained, so
neither test reads them; ``CHANGELOG.md`` is history too.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

HISTORY = ("specs/", "docs/archive/", "CHANGELOG.md")
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

# "file: target" links already broken when the test was written (most sit in
# documents bound for docs/archive/). Each task removes the ones it fixes or
# archives; Task 6 deletes the set.
LINK_ALLOWANCE: set[str] = {
    "docs/guides/simple_open_gams_parity.md: scripts/parity/generate_simple_open_gams_reference.py",
    "docs/guides/simple_open_gams_parity.md: scripts/parity/run_simple_open_gams_parity.py",
    "docs/guides/simple_open_gams_parity.md: src/equilibria/templates/reference/simple_open/scripts/simple_open_v1_benchmark.gms",
    "docs/guides/simple_open_gams_parity.md: src/equilibria/templates/simple_open_parity_pipeline.py",
    "docs/guides/simple_open_parity_contract.md: src/equilibria/templates/simple_open_constraint_jacobian.py",
    "docs/guides/simple_open_parity_contract.md: src/equilibria/templates/simple_open_parity_spec.py",
    "docs/guides/simple_open_stdcge_mapping.md: src/equilibria/templates/reference/simple_open/scripts/simple_open_v1_benchmark.gms",
    "src/equilibria/babel/gdx/README.md: ../../CONTRIBUTING.md",
}

# (file, family) pairs that hit when the test was written. Each task removes
# the pairs it fixes; Task 6 deletes the dict.
MENTION_ALLOWANCE: set[tuple[str, str]] = {
    ("docs/adr/0001-blocks-are-the-model.md", "docs-family"),
    ("docs/adr/0001-blocks-are-the-model.md", "renamed-doc"),
    ("docs/adr/0003-shock-lives-in-shock-block.md", "docs-family"),
    ("docs/adr/0003-shock-lives-in-shock-block.md", "renamed-doc"),
    ("docs/adr/0006-blocks-do-not-know-periods.md", "docs-family"),
    ("docs/adr/0006-blocks-do-not-know-periods.md", "renamed-doc"),
    ("docs/adr/0007-no-block-registry.md", "docs-family"),
    ("docs/adr/0007-no-block-registry.md", "renamed-doc"),
    ("docs/adr/0008-reference-models-are-the-source-of-truth.md", "docs-family"),
    ("docs/adr/0008-reference-models-are-the-source-of-truth.md", "renamed-doc"),
    ("docs/guides/bolivia_mip_to_sam_complete_guide.md", "renamed-doc"),
    ("docs/guides/mip_to_sam_conversion.md", "docs-family"),
    ("docs/guides/mip_to_sam_guide_en.md", "docs-family"),
    ("docs/guides/pep_jacobian_modes.md", "docs-family"),
    ("docs/guides/pep_user_flow_examples.md", "docs-family"),
    ("docs/guides/simulations_api.md", "docs-family"),
    ("docs/ieem_to_pep_transformations.md", "docs-loose"),
    ("docs/readme.md", "renamed-doc"),
    ("docs/roadmap.md", "docs-family"),
    ("docs/roadmap.md", "renamed-doc"),
    ("examples/cge/example_04_custom_blocks.py", "docs-family"),
    ("examples/cge/example_04_custom_blocks.py", "renamed-doc"),
    ("examples/sam/readme.md", "docs-loose"),
    ("scripts/dev/check_layout.py", "docs-family"),
    ("scripts/dev/check_layout.py", "renamed-doc"),
    ("scripts/dev/check_layout.py", "renamed-test"),
    ("scripts/gtap/_parity_json.py", "docs-family"),
    ("scripts/gtap/run_burfisher.py", "renamed-test"),
    ("scripts/paper/run_tables.py", "renamed-test"),
    ("src/equilibria/babel/gdx/README.md", "docs-family"),
    ("src/equilibria/babel/gdx/README.md", "renamed-doc"),
    ("src/equilibria/solver/path_capi.py", "docs-family"),
    ("src/equilibria/templates/gtap/gtap_block_model.py", "docs-family"),
    ("src/equilibria/templates/gtap/gtap_block_model.py", "renamed-doc"),
    ("src/equilibria/templates/gtap/gtap_model_equations.py", "docs-family"),
    ("src/equilibria/templates/gtap/gtap_model_equations.py", "renamed-doc"),
    ("tests/babel/gdx/test_gdxdump_sin_rutas_fijas.py", "renamed-test"),
    ("tests/blocks/gtap/test_overwrite.py", "renamed-test"),
    ("tests/blocks/test_sin_registro_de_bloques.py", "docs-family"),
    ("tests/blocks/test_sin_registro_de_bloques.py", "renamed-doc"),
    ("tests/blocks/test_sin_registro_de_bloques.py", "renamed-test"),
    ("tests/scripts/test_check_commit_msg.py", "renamed-test"),
    ("tests/scripts/test_spec_tools.py", "docs-family"),
    ("tests/templates/gtap/_desempleo.py", "renamed-test"),
    ("tests/templates/gtap/test_gtap7_nlp_parity.py", "docs-family"),
    ("tests/templates/gtap/test_gtap7_nlp_parity.py", "renamed-doc"),
    ("tests/templates/gtap/test_nus333_instrument_shocks.py", "renamed-test"),
    ("tests/templates/gtap/test_nus333_tbl65b_desempleo.py", "renamed-test"),
}


def _tracked(pattern: str) -> list[str]:
    out = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z", pattern])
    return [
        p for p in out.decode("utf-8").split("\0") if p and not p.startswith(HISTORY)
    ]


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
    broken = _broken_links()
    new = sorted(broken - LINK_ALLOWANCE)
    stale = sorted(LINK_ALLOWANCE - broken)
    assert new == [], "broken relative links:\n" + "\n".join(new)
    assert stale == [], f"allowance entries no longer broken; remove them: {stale}"


def _mentions() -> set[tuple[str, str]]:
    this = Path(__file__).resolve().relative_to(ROOT).as_posix()
    hits = set()
    for rel in _tracked("*"):
        if rel == this or not rel.endswith(TEXT) or rel.startswith(FINGERPRINTED):
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
    hits = _mentions()
    new = sorted(hits - MENTION_ALLOWANCE)
    stale = sorted(MENTION_ALLOWANCE - hits)
    assert new == [], f"text points at a moved path: {new}"
    assert stale == [], f"allowance entries no longer hit; remove them: {stale}"


def test_link_parser_sees_relative_and_skips_urls_and_code() -> None:
    text = (
        "[a](../x.md) [b](https://e.org) [c](#top) `[d](y.md)`\n```\n[e](z.md)\n```\n"
    )
    assert [t for t in _links(text) if _local(t)] == ["../x.md"]
