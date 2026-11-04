"""Nothing is left under tests/**/fixtures/ (ADR-0010, phase 4).

Phase 4 moved the reader, solver and snapshot fixtures to data/; phase 3b
moves the GEMPACK ones and empties the allowance.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# GEMPACK results and what is computed from them: regenerated on Windows (3a-2),
# moved in phase 3b.
ALLOWED_UNTIL_PHASE_3B_PREFIX = "tests/fixtures/gtap7_gempack/"
ALLOWED_UNTIL_PHASE_3B = frozenset({"tests/fixtures/nus333_bienestar_gempack.json.gz"})


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
        path.startswith(ALLOWED_UNTIL_PHASE_3B_PREFIX) or path in ALLOWED_UNTIL_PHASE_3B
    )


def test_no_files_under_tests_fixtures() -> None:
    extra = sorted(p for p in _tracked_fixtures() if not _allowed(p))
    assert not extra, f"move these to data/ (ADR-0010): {extra}"


def test_allowance_has_no_stale_entries() -> None:
    tracked = _tracked_fixtures()
    stale = sorted(ALLOWED_UNTIL_PHASE_3B - tracked)
    assert not stale, f"moved already; drop them from the allowance: {stale}"


# A path into any tests/**/fixtures folder, as text or as two Path segments.
_FIXTURE_REF = r'tests/([A-Za-z0-9_/]+/)?fixtures|"tests" / "fixtures"'
_CODE = ("*.py", "*.toml", "*.yml", "*.yaml", "*.cfg", "*.ps1", "*.sh", "*.ipynb")
# The GEMPACK results and their tools move in phase 3b; a source.yaml says where
# a file came from, which is history, not a path anything reads.
_UNTIL_PHASE_3B = ("gtap7_gempack", "nus333_bienestar_gempack")
_EXEMPT_FILES = ("data/gtap7/gempack/", "scripts/windows/", "specs/")


def test_no_code_points_into_tests_fixtures() -> None:
    """Code and user docs read fixtures from data/, except the GEMPACK ones (3b)."""
    out = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            "-E",
            _FIXTURE_REF,
            "--",
            *_CODE,
            "docs/site",
            "docs/guides",
            "examples",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout
    hits = [
        line
        for line in out.splitlines()
        if not line.startswith(_EXEMPT_FILES)
        and not line.startswith("tests/data/test_no_test_fixtures.py")
        and not re.match(r"data/.*source\.yaml:", line)
        and not any(k in line for k in _UNTIL_PHASE_3B)
    ]
    assert not hits, "\n".join(hits)
