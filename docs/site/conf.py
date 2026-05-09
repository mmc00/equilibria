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
    release = "0.2.0"
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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

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

html_theme = "furo"
html_title = f"equilibria {version}"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/mmc00/equilibria",
    "source_branch": "main",
    "source_directory": "docs/site/",
}

# -- doctest ------------------------------------------------------------------

doctest_global_setup = """
import equilibria
"""
