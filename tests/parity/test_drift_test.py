"""drift_test solves the case the MCP gate builds, seeded at the data/ reference,
on the core driver, and ranks how far each cell of the period moved from the seed
(MAR-219). It used to solve the old single-period model, which never converged."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from equilibria.sources import find_reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
_spec = importlib.util.spec_from_file_location(
    "drift_test", ROOT / "scripts/gtap/drift_test.py"
)
assert _spec is not None and _spec.loader is not None
DT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DT)

_FOLDER = find_reference(
    "gtap7_3x3", "gams", form="mcp", mode="altertax", ifsub=0, fbep="header"
)


def test_drift_rows_rank_by_relative_move_and_skip_tiny_seeds_and_welfare():
    seeded = {
        ("pva", ("USA", "Food")): 2.0,
        ("pnd", ("USA", "Food")): 1.0,
        ("walras", ()): 1e-9,
        ("cv", ("ROW",)): 1.0,
    }
    solved = {
        ("pva", ("USA", "Food")): 2.2,
        ("pnd", ("USA", "Food")): 1.0,
        ("walras", ()): 1.0,
        ("cv", ("ROW",)): 2.0,
    }
    rows = DT.drift_rows(seeded, solved)
    assert [(v, c) for _, v, c, _, _ in rows] == [
        ("pva", ("USA", "Food")),
        ("pnd", ("USA", "Food")),
    ]
    assert rows[0][0] == pytest.approx(0.1)


def test_the_default_gdx_is_the_data_reference():
    assert _FOLDER is not None
    assert DT._default_gdx("gtap7_3x3", "altertax", 0) == _FOLDER / "out.gdx"


@pytest.mark.skipif(_FOLDER is None, reason="GAMS MCP altertax ref absent")
@pytest.mark.needs_path
@pytest.mark.needs_gdxdump
@pytest.mark.parametrize("period", ["check", "shock"])
def test_gtap7_3x3_altertax_is_clean(period):
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/gtap/drift_test.py",
            "--dataset",
            "gtap7_3x3",
            "--period",
            period,
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["status"] == "clean", out["headline"]
