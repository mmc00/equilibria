"""The core reaches the family run setup and Newton through their own modules.

PR 3d of gtap-on-core moved what the core driver and the ALTERTAX runner use out
of ``family/driver.py`` (``run_setup``: the run context and its preparation;
``period_newton``: the squared system of a period, Newton-TR, the ratio border).
What is left in ``driver.py`` is the old orchestration (``solve_gtap_family`` and
its period loop), which PR 4 deletes; nothing outside it may import it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE_USERS = (
    "src/equilibria/templates/gtap/template.py",
    "src/equilibria/solver/newton_gtap.py",
    "src/equilibria/family/altertax",
)


def test_the_core_does_not_import_the_old_family_driver() -> None:
    r = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            "-E",
            r"family\.driver( |$|\.|,)|family import driver( |$|,)",
            "--",
            *CORE_USERS,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.stdout.splitlines() == []


def test_driver_re_exports_the_moved_pieces() -> None:
    import equilibria.family.driver as d
    import equilibria.family.period_newton as pn
    import equilibria.family.run_setup as rs

    assert d.prepare_run is rs.prepare_run  # the same object, not a copy
    assert d.FamilyRun is rs.FamilyRun
    assert d._period_system is pn._period_system
    assert d.RatioBorder is pn.RatioBorder
