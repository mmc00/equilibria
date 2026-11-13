"""The documentation site uses pydata-sphinx-theme, with the project's own logos."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONF = ROOT / "docs" / "site" / "conf.py"


def _conf() -> dict:
    return runpy.run_path(str(CONF))


def test_theme_is_pydata() -> None:
    assert _conf()["html_theme"] == "pydata_sphinx_theme"


def test_light_and_dark_logos_exist() -> None:
    conf = _conf()
    logo = conf["html_theme_options"]["logo"]
    static = [CONF.parent / s for s in conf["html_static_path"]]
    for key in ("image_light", "image_dark"):
        name = Path(logo[key]).name
        assert any((d / name).is_file() for d in static), (key, logo[key])


# The navbar shows the top-level toctree entries of index.md, one link each;
# pydata ignores the toctree captions. Each area is therefore one landing page
# whose own toctree holds the area's pages (shown in the left sidebar).
NAVBAR = [
    "sections/getting_started",
    "sections/templates",
    "sections/data_solvers",
    "sections/validation",
    "gallery/index",
    "sections/reference",
]
SITE = CONF.parent


def _toctree_entries(page: Path) -> list[str]:
    """Entries of every ``{toctree}`` fence in a MyST page, in order."""
    entries: list[str] = []
    inside = False
    for line in page.read_text(encoding="utf-8").splitlines():
        if line.startswith("```{toctree}"):
            inside = True
        elif inside and line.startswith("```"):
            inside = False
        elif inside and line.strip() and not line.startswith(":"):
            entries.append(line.strip())
    return entries


def test_navbar_is_the_six_areas() -> None:
    assert _toctree_entries(SITE / "index.md") == NAVBAR


def test_navbar_has_no_more_dropdown() -> None:
    opts = _conf()["html_theme_options"]
    assert opts["header_links_before_dropdown"] >= len(NAVBAR)


def test_every_guide_page_sits_in_one_area() -> None:
    """Each guide page is reachable from exactly one area landing page."""
    pages = {
        f"guide/{p.stem}" for p in (SITE / "guide").glob("*.md") if p.stem != "index"
    }
    seen: list[str] = []
    for area in NAVBAR:
        if area.startswith("sections/"):
            seen += [
                f"guide/{e.split('/')[-1]}"
                for e in _toctree_entries(SITE / f"{area}.md")
                if "guide/" in e
            ]
    assert sorted(seen) == sorted(pages)
