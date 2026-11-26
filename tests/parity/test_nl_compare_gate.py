"""nl_compare on the cascade compares the .nl of the case the MCP gate builds,
written by the core driver at the solve of each period (the other periods fixed,
as GAMS holds them), against the data/ GAMS .nl of the same variant (MAR-219).
It used to write the old single-period monolith (978 columns against GAMS's
2338) and look for the GAMS .nl in /tmp."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from equilibria.sources import find_reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

import nl_compare as NC  # noqa: E402  # ty: ignore[unresolved-import]

_NL = find_reference(
    "gtap7_3x3", "gams", form="nl", mode="altertax", ifsub=0, fbep="header"
)


def test_written_form_families_are_left_out_and_counted():
    stats = {"eq_pf": {"n_diff": 36}, "eq_pft": {"n_diff": 0}, "eq_x": {"n_diff": 0}}
    cells = [("eq_pf[USA,Land,Food]", "xf[USA,Land,Food]", 0.0, -1.0, 1.0, 1.0)]
    kept, left_out = NC.drop_written_form(stats, cells)
    assert kept == [] and left_out == {"eq_pf": 36}


def test_the_gate_case_has_its_references_on_gtap7_3x3():
    refs = NC.gate_references("gtap7_3x3", "altertax", 0)
    assert refs is not None and refs["nl"] == _NL


@pytest.mark.skipif(_NL is None, reason="GAMS .nl altertax ref absent")
@pytest.mark.needs_path
@pytest.mark.needs_gdxdump
@pytest.mark.parametrize("phase", ["check", "shock"])
def test_gtap7_3x3_altertax_is_clean(phase, tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/gtap/nl_compare.py",
            "--dataset",
            "gtap7_3x3",
            "--phase",
            phase,
            "--skip-gams",
            "--out-dir",
            str(tmp_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["status"] == "clean", out["headline"]
    per = out["meta"]["per_phase"][phase]
    assert per["n_common_cons"] > 500
    assert per["written_form_left_out"] == {"eq_pf": 36}
