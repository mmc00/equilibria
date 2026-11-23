"""No file names the old GTAP drivers (gtap-on-core PR 4).

PR 4 deleted the old Pyomo multiperiod driver, its period recipe, the old family
driver and the family period recipe. No file in the repository (code, tests,
scripts, docs, specs, data, CI lists) names them or their entry points again; the
history is reworded too. The only exception is this file.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

OLD = (
    r"gtap_multiperiod_driver",
    r"period_prep",
    r"family[./]driver\b",
    r"family import driver\b",
    r"prep_recipe",
    r"solve_multiperiod",
    r"solve_gtap_family",
    r"PeriodPreparer",
    r"run_recipe",
)
SELF = "tests/test_no_old_driver_names.py"


def _hits(root: Path = ROOT) -> list[str]:
    # git grep's ERE drops \b inside an alternation on macOS (it then misses
    # "family.driver"): git grep takes the candidate lines without \b, Python's re
    # keeps those the patterns match
    loose = "|".join(p.replace(r"\b", "") for p in OLD)
    p = subprocess.run(
        ["git", "grep", "-nIE", loose, "--", ".", f":!{SELF}"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    # 0 = found, 1 = nothing found; anything else (no git, not a checkout) is an
    # error, never a pass
    assert p.returncode in (0, 1), p.stderr
    pattern = re.compile("|".join(OLD))
    return [h for h in p.stdout.splitlines() if pattern.search(h.split(":", 2)[-1])]


def test_no_file_names_the_old_drivers() -> None:
    hits = _hits()
    files = sorted({h.split(":", 1)[0] for h in hits})
    assert files == [], f"{len(files)} files, first hits:\n" + "\n".join(hits[:20])


def test_the_guard_sees_a_name() -> None:
    """The pattern really matches: a name written in a scratch file is found."""
    assert all(
        re.search("|".join(OLD), s)
        for s in (
            "from equilibria.templates.gtap.gtap_multiperiod_driver import x",
            "equilibria.family.driver",
            "from equilibria.family import driver",
            "src/equilibria/family/driver.py",
            "solve_gtap_family(params)",
        )
    )
    assert not re.search("|".join(OLD), "equilibria.family.driver_hook")


def test_the_scan_finds_every_name_in_a_checkout(tmp_path: Path) -> None:
    """``_hits`` itself (git grep, not only the pattern) finds each name in a tracked
    file, and not a longer identifier."""
    lines = [
        "from equilibria.family.driver import prepare_run",
        "from equilibria.family import driver",
        "from equilibria.templates.gtap.period_prep import x",
        "solve_multiperiod(m)",
        "equilibria.family.driver_hook",
    ]
    for i, line in enumerate(lines):
        (tmp_path / f"f{i}.py").write_text(line + "\n")
    for cmd in (["init", "-q"], ["add", "."]):
        subprocess.run(["git", *cmd], cwd=tmp_path, check=True)
    found = sorted(h.split(":", 1)[0] for h in _hits(tmp_path))
    assert found == ["f0.py", "f1.py", "f2.py", "f3.py"]
