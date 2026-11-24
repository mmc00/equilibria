"""The architecture guide (docs/site/guide/architecture.md) and its Lego figures.

The page is read by economists (the drawings), programmers (the code names next
to them) and LLMs, which do not see the images: every figure carries a <title>
that says what it shows. The figures are committed SVGs drawn by
scripts/dev/export_architecture_svgs.mjs with the kit of the JAX page, and take
every colour from jax_backend.css so they follow the light/dark toggle.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "site" / "guide" / "architecture.md"
FIGURES = ROOT / "docs" / "assets" / "architecture"
NAMES = {"tower", "periods", "overwrite", "recipe", "pipeline"}

_PATH = re.compile(r"`((?:src|scripts|tests|data|\.github)/[\w./*<>-]+)`")
_HEX_COLOUR = re.compile(r"(?:fill|stroke|stop-color)\s*(?:=\s*\"|:\s*)#", re.I)


def _svgs() -> list[Path]:
    svgs = sorted(FIGURES.glob("*.svg"))
    assert svgs, "no figures exported"
    return svgs


def test_architecture_figures_are_the_five_and_all_included() -> None:
    text = PAGE.read_text(encoding="utf-8")
    included = set(
        re.findall(r":file: \.\./\.\./assets/architecture/([\w-]+)\.svg", text)
    )
    on_disk = {p.stem for p in _svgs()}
    assert included == on_disk
    assert on_disk >= NAMES


def test_architecture_page_has_no_inline_svg() -> None:
    """Drawings live in files: an LLM reading the page meets text, not coordinates."""
    assert "<svg" not in PAGE.read_text(encoding="utf-8")


def test_architecture_svgs_use_theme_vars() -> None:
    for svg in _svgs():
        text = svg.read_text(encoding="utf-8")
        assert not _HEX_COLOUR.search(text), f"{svg.name} has a hard-coded colour"
        assert 'class="jxb' in text, f"{svg.name} is outside the jax_backend.css scope"


def test_architecture_svgs_have_a_text_alternative() -> None:
    for svg in _svgs():
        text = svg.read_text(encoding="utf-8")
        assert re.match(r'<svg [^>]*role="img"[^>]*><title>[^<]{20,}</title>', text), (
            svg.name
        )


def test_architecture_svg_ids_are_unique_on_the_page() -> None:
    ids: list[str] = []
    for svg in _svgs():
        ids += re.findall(r'\bid="([^"]+)"', svg.read_text(encoding="utf-8"))
    assert sorted({i for i in ids if ids.count(i) > 1}) == []


def test_the_tower_is_gtap_block_order() -> None:
    src = (ROOT / "src" / "equilibria" / "blocks" / "gtap" / "__init__.py").read_text(
        encoding="utf-8"
    )
    body = re.search(r"GTAP_BLOCK_ORDER = \[(.*?)\]", src, re.S)
    assert body
    order = re.findall(r"^\s*(\w+Block),", body.group(1), re.M)
    assert len(order) == 8
    tower = (FIGURES / "tower.svg").read_text(encoding="utf-8")
    positions = [tower.find(f">{name}<") for name in order]
    assert all(p >= 0 for p in positions), dict(zip(order, positions, strict=True))


def test_architecture_page_paths_exist() -> None:
    cited = sorted(set(_PATH.findall(PAGE.read_text(encoding="utf-8"))))
    missing = []
    for path in cited:
        pattern = re.sub(r"<[^>]+>", "*", path).rstrip("/")
        hit = any(ROOT.glob(pattern)) if "*" in pattern else (ROOT / pattern).exists()
        if not hit:
            missing.append(path)
    assert missing == []


def test_architecture_page_has_no_em_dash() -> None:
    assert "—" not in PAGE.read_text(encoding="utf-8")


def test_architecture_svg_colours_are_defined_in_the_css() -> None:
    css = (ROOT / "docs" / "site" / "_static" / "jax_backend.css").read_text(
        encoding="utf-8"
    )
    defined = set(re.findall(r"(--[\w-]+)\s*:", css))
    used: set[str] = set()
    for svg in _svgs():
        used |= set(re.findall(r"var\((--[\w-]+)\)", svg.read_text(encoding="utf-8")))
    assert used - defined == set()
