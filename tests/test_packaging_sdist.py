"""A wheel built from the sdist needs every force-included file inside the sdist."""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10
    import tomli as tomllib  # ty: ignore[unresolved-import]

ROOT = Path(__file__).resolve().parents[1]


def test_sdist_carries_every_force_included_file() -> None:
    hatch = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["hatch"][
        "build"
    ]["targets"]
    forced = hatch["wheel"].get("force-include", {})
    included = [p.strip("/") for p in hatch["sdist"]["include"]]
    missing = [
        s for s in forced if not any(s == i or s.startswith(i + "/") for i in included)
    ]
    assert not missing, f"force-include sources not in the sdist: {missing}"
