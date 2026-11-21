"""seed_and_solve: a solve that never ran is a tool error, never "an equation differs";
and on gtap7_3x3 altertax, where the gates pass, the GAMS point stays (MAR-216)."""

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
    "seed_and_solve", ROOT / "scripts/gtap/seed_and_solve.py"
)
assert _spec is not None and _spec.loader is not None
SS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SS)

_FOLDER = find_reference(
    "gtap7_3x3", "gams", form="mcp", mode="altertax", ifsub=0, fbep="header"
)
REF = None if _FOLDER is None else _FOLDER / "out.gdx"


def _fake_out(code):
    return {
        "code": code,
        "resid": float("inf") if code is None else 1e-9,
        "median": 1e-8,
        "eq_resid": [(2.7, "eq_ytax", ("EU_28", "ft"))],
        "n_seed": 1,
        "n_cascade": 0,
        "sent_name": "pft",
        "sent_before": 1.0,
        "sent_after": 1.0,
    }


def test_a_solve_that_never_ran_is_a_tool_error(monkeypatch):
    monkeypatch.setattr(SS, "run_seed_and_solve_gtap", lambda *a, **k: _fake_out(None))
    out = SS._work("gtap7_3x3", Path("/ref/out.gdx"), "check", 3)
    assert out["status"] == "error"
    assert out["meta"]["error_kind"] == "solve_not_run"
    assert "DIFFERS" not in out["headline"]


def test_a_solve_that_ran_and_drifted_is_still_dirty(monkeypatch):
    monkeypatch.setattr(SS, "run_seed_and_solve_gtap", lambda *a, **k: _fake_out(2))
    out = SS._work("gtap7_3x3", Path("/ref/out.gdx"), "check", 3)
    assert out["status"] == "dirty"


@pytest.mark.skipif(
    REF is None or not REF.exists(), reason="GAMS MCP altertax ref absent"
)
@pytest.mark.needs_path
@pytest.mark.parametrize("period", ["check", "shock"])
def test_gams_altertax_point_stays_on_gtap7_3x3(period):
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/gtap/seed_and_solve.py",
            "--dataset",
            "gtap7_3x3",
            "--gdx",
            str(REF),
            "--period",
            period,
            "--top",
            "5",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    lines = proc.stdout.strip().splitlines()
    assert lines, f"no JSON on stdout (exit {proc.returncode}):\n{proc.stderr[-2000:]}"
    payload = json.loads(lines[-1])
    assert payload["status"] == "clean", payload["headline"]
