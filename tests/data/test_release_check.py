"""release_check.py: a tool refuses to record a run made by another GEMPACK release.

gtapv7.exe/gtapuv7.exe (RunGTAP) print 12.1.003; tablo/gemsim/sltoht of the GEMPACK
install print 11.3.004. Each record's tool_version is the release of the program that made it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from equilibria.sources import GEMPACK_MODEL_DIR

sys.path.insert(0, str(GEMPACK_MODEL_DIR / "tools"))
import release_check  # noqa: E402  # ty: ignore[unresolved-import]

MATRIX_3X3 = GEMPACK_MODEL_DIR.parent / "gtap7_3x3" / "gempack_tm10_s8-16-32_capflex"


def test_release_parses_both_banners() -> None:
    assert release_check.release(MATRIX_3X3 / "solve_console.txt") == "12.1.003"
    assert release_check.release(MATRIX_3X3 / "sltoht_console.txt") == "11.3.004"


def test_release_mismatch_refuses() -> None:
    with pytest.raises(release_check.ReleaseMismatch, match=r"11\.3\.004.*12\.1\.003"):
        release_check.require(MATRIX_3X3 / "sltoht_console.txt", "12.1.003")
    assert (
        release_check.require(MATRIX_3X3 / "solve_console.txt", "12.1.003")
        == "12.1.003"
    )


def test_console_without_banner_refuses(tmp_path: Path) -> None:
    console = tmp_path / "console.txt"
    console.write_text("no banner here\n")
    with pytest.raises(release_check.ReleaseMismatch, match="no GEMPACK Release"):
        release_check.require(console, "11.3.004")
