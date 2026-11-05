"""Sphinx configuration for the equilibria documentation site."""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

# Make the package importable for autodoc when building docs locally.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# -- Project information ------------------------------------------------------

project = "equilibria"
author = "Marlon Molina"
copyright = f"{date.today().year}, {author}"

try:
    from equilibria.version import __version__ as release
except Exception:  # noqa: BLE001 — fallback for partial installs
    release = "0.6.1"
version = ".".join(release.split(".")[:2])

# -- General configuration ----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "examples_src"]

# MyST options — enable common extensions.
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# -- Autodoc / Napoleon -------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
# Render docstring "Attributes:" sections as :ivar: fields instead of
# .. attribute:: directives — avoids duplicate-object warnings when autodoc
# also documents the real (pydantic) class attributes.
napoleon_use_ivar = True

# -- Intersphinx --------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- sphinx-gallery -----------------------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": [str(Path(__file__).parent / "examples_src")],
    "gallery_dirs": ["gallery"],
    "filename_pattern": r"/example_",
    "remove_config_comments": True,
    "plot_gallery": "True",
    "download_all_examples": False,
}

# -- HTML output --------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"equilibria {version}"
# docs/assets is copied into _static, so the theme finds the logos there.
html_static_path = ["_static", "../assets"]
html_css_files = ["matrix.css"]
html_favicon = str(ROOT / "docs" / "assets" / "equilibria-favicon.svg")
# The navbar holds one landing page per area (sections/); the area's pages show
# in the left sidebar. The home page has no area, so no sidebar.
html_sidebars = {"index": []}
html_context = {
    "github_user": "mmc00",
    "github_repo": "equilibria",
    "github_version": "main",
    "doc_path": "docs/site",
}
html_theme_options = {
    "logo": {
        "text": "equilibria",
        "image_light": "_static/equilibria-mark.svg",
        "image_dark": "_static/equilibria-mono-white.svg",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/mmc00/equilibria",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/equilibria/",
            "icon": "fa-brands fa-python",
        },
    ],
    "use_edit_page_button": True,
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "show_toc_level": 2,
    "footer_start": ["copyright"],
    "footer_end": ["theme-version"],
}

# -- doctest ------------------------------------------------------------------

doctest_global_setup = """
import equilibria
"""

# -- Benchmarks page (regenerated from committed CSVs) -----------------------

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "_scripts"))
try:
    from render_benchmarks import render as _render_benchmarks  # noqa: E402
    _render_benchmarks(
        out_path=_HERE / "guide" / "benchmarks.md",
        data_dir=_HERE / "_data" / "benchmarks",
    )
except Exception as exc:  # noqa: BLE001 — never fail a docs build on this
    print(f"[conf.py] benchmarks render skipped: {exc}")
