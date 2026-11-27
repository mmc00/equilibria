"""diff_tautology perturbs the case the MCP gate builds, seeded at the data/
reference, and only the requested period's cells (MAR-219). It used to build the
old single-period model without the gempack FBEP basis and read the June refs."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pyomo.environ as pyo
import pytest

from equilibria.sources import find_reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
_spec = importlib.util.spec_from_file_location(
    "diff_tautology", ROOT / "scripts/gtap/diff_tautology.py"
)
assert _spec is not None and _spec.loader is not None
DT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DT)

_FOLDER = find_reference(
    "gtap7_3x3", "gams", form="mcp", mode="altertax", ifsub=0, fbep="header"
)


def _model():
    """eq_pdeq is a market clearing paired with pd in check (pd is not in it, as in
    GTAP's xds = sum xda) and a tautology in shock (pd == pd)."""
    m = pyo.ConcreteModel()
    m.t = pyo.Set(initialize=["check", "shock"])
    m.pd = pyo.Var(m.t, initialize=2.0)
    m.xds = pyo.Var(m.t, initialize=3.0)
    m.eq_pdeq = pyo.Constraint(
        m.t,
        rule=lambda m, t: (m.xds[t] == 3.0) if t == "check" else (m.pd[t] == m.pd[t]),
    )
    return m


def test_a_clearing_row_without_its_paired_price_is_anchored():
    rows = DT.scan_pairs(_model(), "check", [("eq_pdeq", "pd")], 1e-4)
    assert [(eq, v, cell) for _, eq, v, cell, _ in rows] == [("eq_pdeq", "pd", ())]
    assert rows[0][0] > 1e-6


def test_a_row_that_responds_to_nothing_is_a_tautology():
    rows = DT.scan_pairs(_model(), "shock", [("eq_pdeq", "pd")], 1e-4)
    assert rows[0][0] < 1e-6


def test_the_default_gdx_is_the_data_reference():
    assert _FOLDER is not None
    assert DT._default_gdx("gtap7_3x3", "altertax", 0) == _FOLDER / "out.gdx"


@pytest.mark.skipif(_FOLDER is None, reason="GAMS MCP altertax ref absent")
@pytest.mark.needs_gdxdump  # the seed reads the GAMS GDX through it
@pytest.mark.parametrize("period", ["check", "shock"])
def test_gtap7_3x3_altertax_is_clean(period):
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/gtap/diff_tautology.py",
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
