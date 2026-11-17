"""The JAX backend page (docs/site/guide/jax_backend.md) stays true to the code.

The page names many files of the family route and draws its figures as committed
SVGs. Three things can rot without anyone noticing:

- a path the page cites is moved or renamed;
- a figure gets a hard-coded colour, so it vanishes on the dark theme (the SVGs
  are inlined and take every colour from the CSS variables of jax_backend.css);
- an em dash slips into the text (the docs style uses commas, colons or
  parentheses instead).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "site" / "guide" / "jax_backend.md"
FIGURES = ROOT / "docs" / "assets" / "jax-backend"

# Backticked repo paths: `src/...`, `scripts/...`, `tests/...`, `data/...`, `.github/...`
_PATH = re.compile(r"`((?:src|scripts|tests|data|\.github)/[\w./*<>-]+)`")
_HEX_COLOUR = re.compile(r"(?:fill|stroke|stop-color)\s*(?:=\s*\"|:\s*)#", re.I)


def _cited_paths() -> list[str]:
    return sorted(set(_PATH.findall(PAGE.read_text(encoding="utf-8"))))


def test_jax_backend_page_paths_exist() -> None:
    cited = _cited_paths()
    assert cited, "the page cites no repo paths"
    missing = []
    for path in cited:
        pattern = re.sub(r"<[^>]+>", "*", path).rstrip("/")
        if "*" in pattern:
            if not any(ROOT.glob(pattern)):
                missing.append(path)
        elif not (ROOT / pattern).exists():
            missing.append(path)
    assert missing == []


def test_jax_backend_svgs_use_theme_vars() -> None:
    svgs = sorted(FIGURES.glob("*.svg"))
    assert svgs, "no figures exported"
    for svg in svgs:
        text = svg.read_text(encoding="utf-8")
        assert not _HEX_COLOUR.search(text), f"{svg.name} has a hard-coded colour"
        # colours come from CSS variables or from the svg.jxb classes of jax_backend.css
        assert 'class="jxb' in text, f"{svg.name} is outside the jax_backend.css scope"


def test_jax_backend_page_includes_every_figure() -> None:
    text = PAGE.read_text(encoding="utf-8")
    included = set(
        re.findall(r":file: \.\./\.\./assets/jax-backend/([\w-]+\.svg)", text)
    )
    on_disk = {p.name for p in FIGURES.glob("*.svg")}
    assert included == on_disk


def test_jax_backend_page_has_no_em_dash() -> None:
    assert "—" not in PAGE.read_text(encoding="utf-8")


def test_jax_backend_svg_ids_are_unique_on_the_page() -> None:
    """All figures are inlined into one page, so an id must not repeat across them."""
    ids: list[str] = []
    for svg in sorted(FIGURES.glob("*.svg")):
        ids += re.findall(r'\bid="([^"]+)"', svg.read_text(encoding="utf-8"))
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    assert repeated == []


def test_jax_backend_svgs_have_a_text_alternative() -> None:
    for svg in sorted(FIGURES.glob("*.svg")):
        text = svg.read_text(encoding="utf-8")
        assert re.match(r'<svg [^>]*role="img"[^>]*><title>[^<]+</title>', text), (
            svg.name
        )


def test_jax_backend_svg_colours_are_defined_in_the_css() -> None:
    css = (ROOT / "docs" / "site" / "_static" / "jax_backend.css").read_text(
        encoding="utf-8"
    )
    defined = set(re.findall(r"(--[\w-]+)\s*:", css))
    used: set[str] = set()
    for svg in FIGURES.glob("*.svg"):
        used |= set(re.findall(r"var\((--[\w-]+)\)", svg.read_text(encoding="utf-8")))
    assert sorted(used - defined) == []
