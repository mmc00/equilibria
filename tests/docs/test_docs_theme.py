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
