"""Only the listed scripts still name the old GTAP drivers (PR 3b of gtap-on-core).

Every other script and example runs on the core driver (``Exercise``). The list
shrinks with the code: PR 3c moves the notebooks' generator, PR 3d the family
altertax runner, PR 4 deletes the old drivers with the scripts that only make sense
for them. A new script that calls an old driver fails here; so does an entry that
no longer names one (take it off the list).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

OLD = (
    "solve_multiperiod",
    "gtap_multiperiod_driver",
    "period_prep",
    "PeriodPreparer",
    "solve_gtap_family",
    "family.driver",
    "family import driver",
    "prep_recipe",
    "run_recipe",
)

_TIMES_OLD = "times or dissects the old drivers' own steps (PR 4)"
_MONOLITH = (
    "runs the monolithic GTAPMultiPeriodModel, which the core does not build (PR 4)"
)
UNTIL_PR4: dict[str, str] = {
    "examples/notebooks/gtap7/burfisher/make_notebooks.py": "the notebooks' generator (PR 3c)",
    "scripts/dev/record_pr3b_scripts.py": "records the old driver's results: the PR 3b oracle",
    "scripts/family/altertax_rungtap_measure.py": "family.driver._world_income (PR 3d)",
    "scripts/family/budget_20x41.py": _TIMES_OLD,
    "scripts/family/fscale_check.py": _TIMES_OLD,
    "scripts/family/newton_checkpoint.py": _TIMES_OLD,
    "scripts/family/newton_cost_20x41.py": _TIMES_OLD,
    "scripts/family/parity_ladder.py": _TIMES_OLD,
    "scripts/family/prep_cost_20x41.py": _TIMES_OLD,
    "scripts/family/squared_check.py": _TIMES_OLD,
    "scripts/family/timing_20x41.py": _TIMES_OLD,
    "scripts/gtap/diag_mp_check_drift.py": _TIMES_OLD,
    "scripts/gtap/perf_family_20x41.py": _TIMES_OLD,
    "scripts/gtap/phase_times.py": _TIMES_OLD,
    "scripts/gtap/bench_nlp_timing.py": _MONOLITH,
    "scripts/gtap/diag_mp_3x3.py": _MONOLITH,
    "scripts/gtap/gen_linearization_study.py": _MONOLITH,
    "scripts/gtap/measure_gate_tols.py": _MONOLITH,
    "scripts/gtap/measure_gtap_pure_tols.py": _MONOLITH,
    "scripts/gtap/measure_nlp_vs_nlp.py": _MONOLITH,
    "scripts/gtap/measure_nlp_vs_nlp_altertax.py": _MONOLITH,
    "scripts/gtap/run_burfisher.py": "route='old' for the old-vs-core equality test (PR 4)",
}


def _names_an_old_driver(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return any(name in text for name in OLD)


def _tracked_python() -> list[str]:
    r = subprocess.run(
        [
            "git",
            "ls-files",
            "scripts/*.py",
            "scripts/**/*.py",
            "examples/*.py",
            "examples/**/*.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"not a git checkout: {r.stderr.strip()}")
    return sorted(set(r.stdout.split()))


def test_only_the_listed_scripts_name_an_old_driver() -> None:
    found = [f for f in _tracked_python() if _names_an_old_driver(ROOT / f)]
    assert sorted(set(found) - set(UNTIL_PR4)) == []


def test_every_listed_script_still_names_one() -> None:
    stale = [
        f
        for f in UNTIL_PR4
        if not (ROOT / f).exists() or not _names_an_old_driver(ROOT / f)
    ]
    assert stale == []
