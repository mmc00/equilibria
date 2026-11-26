"""diff_equation_form shows the row of the case the MCP gate builds, in the
requested period, next to the GAMS block of the data/ reference's model.gms
(MAR-219). It used to print the old single-period model and the June .gms."""

import subprocess
import sys
from pathlib import Path

import pytest

from equilibria.sources import find_reference

ROOT = Path(__file__).resolve().parents[2]
_FOLDER = find_reference(
    "gtap7_3x3", "gams", form="mcp", mode="altertax", ifsub=0, fbep="header"
)


@pytest.mark.skipif(_FOLDER is None, reason="GAMS MCP altertax ref absent")
def test_shows_the_gate_row_and_the_data_gms():
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/gtap/diff_equation_form.py",
            "--dataset",
            "gtap7_3x3",
            "--eq",
            "eq_xds",
            "--cell",
            "ROW,Svces",
            "--gams-eq",
            "xdseq",
            "--period",
            "check",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ps[ROW,Svces,check]" in proc.stdout
    assert "(from model.gms)" in proc.stdout
    assert "not found in" not in proc.stdout
