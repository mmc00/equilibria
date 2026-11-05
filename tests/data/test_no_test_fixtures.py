"""Nothing is left under tests/**/fixtures/ or runs/ (ADR-0010, phases 4 and 3b).

Phase 4 moved the reader, solver and snapshot fixtures to data/; phase 3b moved
the GEMPACK ones and deleted tests/fixtures/ and runs/.
"""

from __future__ import annotations

import io
import re
import subprocess
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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


def test_no_files_under_tests_fixtures() -> None:
    extra = sorted(_tracked_fixtures())
    assert not extra, f"move these to data/ (ADR-0010): {extra}"


def test_tests_fixtures_is_gone() -> None:
    """Phase 3b deleted tests/fixtures/ and runs/: neither has a tracked file."""
    left = sorted(_tracked("tests/fixtures", "runs"))
    assert not left, f"phase 3b removed these folders: {left}"


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
    """Code and user docs read fixtures from data/."""
    hits = _grep_code(_FIXTURE_REF)
    assert not hits, "\n".join(hits)


# The git-ignored work folders of two hashed tools, which stay (plan-phase-3b
# Task 8): regenerate_paper.py runs its programs in runs/gempack_work and
# run_gempack_matrix.py writes under runs/gempack_matrix. Naming them is not
# reading the removed runs/.
_RUNS_WORK = re.compile(r'runs/gempack_(work|matrix)|"runs" / "gempack_work"')
# Reads git's bytes of the removed runs/ .cmf and skips once they are gone (Task 8).
_RUNS_HISTORY = ("tests/scripts/test_swap_job.py",)


def test_no_code_reads_runs() -> None:
    """Nothing reads runs/, which leaves the repository in phase 3b (Task 8)."""
    hits = [
        line
        # not \b: git grep -E on macOS does not know it and would match nothing
        for line in _grep_code(r'(^|[^[:alnum:]_])runs/|"runs" /')
        if not line.startswith(("runs/", *_RUNS_HISTORY))
        and re.search(
            r'(^|\W)runs/|"runs" /', _RUNS_WORK.sub("", line.split(":", 2)[2])
        )
    ]
    assert not hits, "\n".join(hits)


def test_runs_work_folders_are_the_only_exemption() -> None:
    """The exemption hides the two work folders and nothing else."""
    assert not _RUNS_WORK.sub("", 'ROOT / "runs" / "gempack_work"').count("runs")
    assert "runs/nus333_compare" in _RUNS_WORK.sub("", "runs/nus333_compare/x.py")


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
            hits.append(f"{rel}:{n}: {lines[n - 1].strip()}")
    assert not hits, "\n".join(hits)
