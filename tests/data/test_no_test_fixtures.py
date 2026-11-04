"""Nothing is left under tests/**/fixtures/ (ADR-0010, phase 4).

Each phase-4 task moves one family of fixtures to data/ and removes its paths
from ``ALLOWED_UNTIL_PHASE_4``; phase 3b removes the GEMPACK ones.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# GEMPACK results and what is computed from them: regenerated on Windows (3a-2),
# moved in phase 3b.
ALLOWED_UNTIL_PHASE_3B_PREFIX = "tests/fixtures/gtap7_gempack/"
ALLOWED_UNTIL_PHASE_3B = frozenset({"tests/fixtures/nus333_bienestar_gempack.json.gz"})

ALLOWED_UNTIL_PHASE_4 = frozenset(
    {
        "tests/compare/fixtures/eq_burfisher_tbl45a_levels.json.gz",
        "tests/compare/fixtures/eq_burfisher_tbl46a_levels.json.gz",
        "tests/compare/fixtures/eq_gtap7_3x3_levels.json.gz",
        "tests/compare/fixtures/gtap3x3_minimo.yaml",
        "tests/compare/fixtures/make_eq_burfisher_levels.py",
        "tests/compare/fixtures/make_eq_gtap7_levels.py",
    }
)


def _tracked_fixtures() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files", "tests/**/fixtures/**", "tests/fixtures/**"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {p for p in out.splitlines() if p}


def _allowed(path: str) -> bool:
    return (
        path.startswith(ALLOWED_UNTIL_PHASE_3B_PREFIX)
        or path in ALLOWED_UNTIL_PHASE_3B
        or path in ALLOWED_UNTIL_PHASE_4
    )


def test_no_files_under_tests_fixtures() -> None:
    extra = sorted(p for p in _tracked_fixtures() if not _allowed(p))
    assert not extra, f"move these to data/ (ADR-0010): {extra}"


def test_allowance_has_no_stale_entries() -> None:
    tracked = _tracked_fixtures()
    stale = sorted((ALLOWED_UNTIL_PHASE_4 | ALLOWED_UNTIL_PHASE_3B) - tracked)
    assert not stale, f"moved already; drop them from the allowance: {stale}"
