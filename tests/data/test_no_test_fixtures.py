"""Nothing is left under tests/**/fixtures/ or runs/ (ADR-0010, phases 4 and 3b).

Phase 4 moved the reader, solver and snapshot fixtures to data/; phase 3b
moves the GEMPACK ones, one family at a time, and empties PENDING_3B.
"""

from __future__ import annotations

import fnmatch
import io
import re
import subprocess
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# GEMPACK results and what is computed from them: regenerated on Windows (3a-2),
# moved in phase 3b one family at a time. Each glob names the task of
# plan-phase-3b.md that removes it; an empty dict means 3b is done.
_GG = "tests/fixtures/gtap7_gempack/"
PENDING_3B: dict[str, str] = {
    _GG + "burfisher_gragg/*": "Task 5",
    "tests/fixtures/nus333_bienestar_gempack.json.gz": "Task 6",
    _GG + "sl4levels_sample_sic.txt": "Task 7",
    _GG + "updated_synthetic.har": "Task 7",
    _GG + "make_synthetic_updated_har.py": "Task 7",
    _GG + "table1_levels_v061_gtap7_3x3.json.gz": "Task 7",
    _GG + "burfisher/readme.md": "Task 7",
    _GG + "burfisher_gragg/readme.md": "Task 7",
    "runs/**": "Task 8",
}


def _tracked(*pathspecs: str) -> set[str]:
    out = subprocess.run(
        ["git", "ls-files", *pathspecs],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {p for p in out.splitlines() if p}


def _tracked_fixtures() -> set[str]:
    return _tracked("tests/**/fixtures/**", "tests/fixtures/**")


def _pending(path: str) -> bool:
    return any(fnmatch.fnmatchcase(path, glob) for glob in PENDING_3B)


def _names_pending(text: str) -> bool:
    """True if ``text`` writes fixture paths and every one is still pending.

    A path may be one string or ``Path`` segments (``"tests" / "fixtures" /
    ...``), and an f-string may cut it: a path that stops before the file name
    (the folder, or ``sl4levels6_`` before ``{ds}``) is pending while any glob
    under it is.
    """
    unquoted = re.sub(r"[\\\"']", "", re.sub(r"\b[rbfRBF]{1,2}(?=[\"'])", "", text))
    flat = re.sub(r"\s*/\s*", "/", unquoted)
    # "tests/" may sit on a line before the window that starts at "fixtures".
    paths = re.findall(r"(?:tests/(?:[A-Za-z0-9_]+/)*)?fixtures[A-Za-z0-9_./*-]*", flat)
    if not paths:
        return False
    for path in paths:
        path = path.rstrip(".")
        if not path.startswith("tests/"):
            path = "tests/" + path
        if not any(
            fnmatch.fnmatchcase(path, glob) or glob.split("*")[0].startswith(path)
            for glob in PENDING_3B
        ):
            return False
    return True


def test_no_files_under_tests_fixtures() -> None:
    extra = sorted(p for p in _tracked_fixtures() if not _pending(p))
    assert not extra, f"move these to data/ (ADR-0010): {extra}"


def test_pending_globs_match_something() -> None:
    """A glob whose files are gone is stale: drop it from PENDING_3B."""
    tracked = _tracked("tests/fixtures/**", "runs/**")
    stale = sorted(
        f"{glob} ({task})"
        for glob, task in PENDING_3B.items()
        if not any(fnmatch.fnmatchcase(p, glob) for p in tracked)
    )
    assert not stale, f"moved already; drop them from PENDING_3B: {stale}"


# A path into any tests/**/fixtures folder, as text or as two Path segments.
_FIXTURE_REF = r'tests/([A-Za-z0-9_/]+/)?fixtures|"tests" / "fixtures"'
_CODE = ("*.py", "*.toml", "*.yml", "*.yaml", "*.cfg", "*.ps1", "*.sh", "*.ipynb")
# A source.yaml says where a file came from, which is history, not a path
# anything reads.
_EXEMPT_FILES = ("data/gtap7/gempack/", "scripts/windows/", "specs/")


def _grep_code(pattern: str) -> list[str]:
    out = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            "-E",
            pattern,
            "--",
            *_CODE,
            "docs/site",
            "docs/notes",
            "examples",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout
    return [
        line
        for line in out.splitlines()
        if not line.startswith(_EXEMPT_FILES)
        and not line.startswith("tests/data/test_no_test_fixtures.py")
        and not re.match(r"data/.*source\.yaml:", line)
    ]


def test_no_code_points_into_tests_fixtures() -> None:
    """Code and user docs read fixtures from data/, except the families pending in 3b."""
    hits = [
        line
        for line in _grep_code(_FIXTURE_REF)
        if not _names_pending(line.split(":", 2)[2])
    ]
    assert not hits, "\n".join(hits)


def test_no_code_reads_runs() -> None:
    """Nothing reads runs/, which leaves the repository in phase 3b (Task 8)."""
    if "runs/**" in PENDING_3B:
        return
    hits = [
        line for line in _grep_code(r'\bruns/|"runs" /') if not line.startswith("runs/")
    ]
    assert not hits, "\n".join(hits)


# Folders named "fixtures" that are not tests/**/fixtures (a tmp_path subfolder).
_NOT_TESTS_FIXTURES = frozenset(
    {
        "tests/templates/gtap/test_linearization_study_gen.py",
        "tests/data/test_no_test_fixtures.py",  # this guard names the word
    }
)


def _fixtures_segments(path: Path) -> list[int]:
    """Lines where a Python string literal is exactly "fixtures" (a Path segment)."""
    tokens = tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
    return [
        t.start[0]
        for t in tokens
        if t.type == tokenize.STRING and t.string.strip("\"'") == "fixtures"
    ]


def test_no_python_path_segment_into_tests_fixtures() -> None:
    """Catches ``parent / "fixtures" / ...`` however it is split over lines."""
    files = _tracked("*.py")
    hits = []
    for rel in sorted(files):
        if rel.startswith(_EXEMPT_FILES) or rel in _NOT_TESTS_FIXTURES:
            continue
        lines = (ROOT / rel).read_text().splitlines()
        for n in _fixtures_segments(ROOT / rel):
            window = "\n".join(lines[n - 1 : n + 3])
            if not _names_pending(window):
                hits.append(f"{rel}:{n}: {lines[n - 1].strip()}")
    assert not hits, "\n".join(hits)
