"""validate_reference reads the GAMS point on the case the MCP gate builds, seeded
at the data/ reference, over the requested period's rows (MAR-219). It used to seed
the old single-period model and read the June refs, so the gempack FBEP sign of
the reference showed up as "the reference violates its own eq_pfaeq"."""

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
    "validate_reference", ROOT / "scripts/gtap/validate_reference.py"
)
assert _spec is not None and _spec.loader is not None
VR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(VR)

_FOLDER = find_reference(
    "gtap7_3x3", "gams", form="mcp", mode="altertax", ifsub=0, fbep="header"
)


def _model():
    m = pyo.ConcreteModel()
    m.t = pyo.Set(initialize=["check", "shock"])
    m.x = pyo.Var(m.t, initialize=1.0)
    m.eq_x = pyo.Constraint(
        m.t, rule=lambda m, t: m.x[t] == (1.0 if t == "check" else 3.0)
    )
    return m


def test_residuals_read_only_the_requested_period():
    fams, _ = VR._equation_residuals(_model(), "check")
    assert fams["eq_x"]["max"] == 0.0 and fams["eq_x"]["n"] == 1
    fams, _ = VR._equation_residuals(_model(), "shock")
    assert fams["eq_x"]["max"] == 2.0 and fams["eq_x"]["worst"] == ()


def test_the_default_gdx_is_the_data_reference():
    assert _FOLDER is not None
    assert VR._default_gdx("gtap7_3x3", "altertax", 0) == _FOLDER / "out.gdx"


@pytest.mark.skipif(_FOLDER is None, reason="GAMS MCP altertax ref absent")
@pytest.mark.needs_gdxdump  # the seed reads the GAMS GDX through it
@pytest.mark.parametrize("period", ["check", "shock"])
def test_gtap7_3x3_altertax_is_clean(period):
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/gtap/validate_reference.py",
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


_GTAP_FOLDER = find_reference(
    "gtap7_3x3", "gams", form="mcp", mode="gtap", ifsub=0, fbep="header"
)


@pytest.mark.skipif(_GTAP_FOLDER is None, reason="GAMS MCP gtap ref absent")
@pytest.mark.needs_gdxdump
def test_gtap7_3x3_gtap_shock_reads_no_muted_welfare_row():
    """The driver mutes cv/ev/u/ug/us (steps.mute_welfare); their rows move with
    psave, which follows GTAP v7, not GAMS, so they are not read at the GAMS point."""
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/gtap/validate_reference.py",
            "--dataset",
            "gtap7_3x3",
            "--period",
            "shock",
            "--mode",
            "gtap",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["status"] == "clean", out["headline"]
