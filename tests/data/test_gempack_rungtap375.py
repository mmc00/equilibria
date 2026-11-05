"""RunGTAP 3.75's gtapv7.tab 7.0 and its patched levels TABs (phase 3a-2).

The model of the gtapv7.exe that made the 66 matrix records is kept unchanged in
data/gtap7/gempack_rungtap375/ (docs/archive/findings/gtapv7_tab_70_vs_71_2026-11-03.md).
Its patched versions in gempack_rungtap375_levels/ are exactly what
tools/make_levels_tab.py writes from it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from equilibria.sources import DATA_ROOT, GEMPACK_MODEL_DIR, load_source, sha256_file

sys.path.insert(0, str(GEMPACK_MODEL_DIR / "tools"))
import make_levels_tab  # noqa: E402  # ty: ignore[unresolved-import]

RUNGTAP = DATA_ROOT / "gtap7" / "gempack_rungtap375"
LEVELS = DATA_ROOT / "gtap7" / "gempack_rungtap375_levels"
EXPECTED = {
    "gtapv7.tab": "67cb2c6793fc40b3e14d8ce997ab1d66920b03800d0d0f3897c7a1bf0d5d2836",
    "gtapv7.sti": "a3279021f72413ed3519031dd5ffbe2a893f7c770c0d8267a174415d8abc7a87",
}
PATCHED = (
    "gtapv7_levels.tab",
    "gtapv7_levels.sti",
    "gtapuv7_levels.tab",
    "gtapuv7_levels.sti",
)


def test_rungtap375_is_unchanged() -> None:
    rec = load_source(RUNGTAP)
    assert rec.kind == "model"
    assert "not stated by the source; to confirm" in rec.raw["notes"]
    assert {f: sha256_file(RUNGTAP / f) for f in EXPECTED} == EXPECTED
    assert set(rec.raw["outputs"]) == set(EXPECTED)


def test_levels_tabs_are_derived(tmp_path: Path) -> None:
    rec = load_source(LEVELS)
    assert rec.kind == "model"
    assert rec.raw["derived_from"] == "gtap7/gempack_rungtap375"
    assert {c["file"] for c in rec.raw["changes"]} == set(PATCHED)
    assert all(c["change"] == "added" for c in rec.raw["changes"])
    for flag in ([], ["--uncondensed"]):
        out = tmp_path / ("u" if flag else "c")
        argv = [str(RUNGTAP / "gtapv7.tab"), "--sti", str(RUNGTAP / "gtapv7.sti")]
        assert make_levels_tab.main([*argv, "--out-dir", str(out), *flag]) == 0
        for f in out.iterdir():
            assert f.read_bytes() == (LEVELS / f.name).read_bytes(), f.name
