"""inheritance_table measures where Python's check starts from, at the core solve
of the case the MCP gate builds, instead of asserting it (MAR-219). It used to
print "Python seeds from the GAMS check value" for every var, from the June .gms."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from equilibria.sources import find_reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
_spec = importlib.util.spec_from_file_location(
    "inheritance_table", ROOT / "scripts/gtap/inheritance_table.py"
)
assert _spec is not None and _spec.loader is not None
IT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(IT)

_FOLDER = find_reference(
    "gtap7_3x3", "gams", form="mcp", mode="altertax", ifsub=0, fbep="header"
)


@pytest.mark.parametrize(
    "d_prior, d_gams, source",
    [
        (0.0, 0.2, "prior period"),
        (0.2, 0.0, "GAMS, same period"),
        (0.0, 0.0, "both (they agree)"),
        (0.2, 0.3, "neither"),
    ],
)
def test_the_source_follows_the_measured_distances(d_prior, d_gams, source):
    assert IT.seed_source(d_prior, d_gams) == source


def test_the_gms_is_the_data_reference():
    assert _FOLDER is not None
    assert IT._gms("gtap7_3x3", 0) == _FOLDER / "model.gms"


@pytest.mark.skipif(_FOLDER is None, reason="GAMS MCP altertax ref absent")
@pytest.mark.needs_path
@pytest.mark.needs_gdxdump
def test_runs_on_gtap7_3x3():
    proc = subprocess.run(
        [sys.executable, "scripts/gtap/inheritance_table.py", "--dataset", "gtap7_3x3"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "measured at the core solve of check" in proc.stdout
