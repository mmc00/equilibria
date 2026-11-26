"""bench.py reads the MCP gate's own measure and shows, per period, which variables
moved toward or away from GAMS against the previous run."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
_spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts/gtap/bench.py")
assert _spec is not None and _spec.loader is not None
BENCH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BENCH)


class _FakeGate:
    @staticmethod
    def _solve_and_measure(*_args):
        per = {"match": 99.0, "code": 1, "total": 100, "var_div": {"pwmg": 1}}
        return {"base": per, "check": per, "shock": per}


def test_bench_reads_the_gate_measure(monkeypatch):
    monkeypatch.setattr(BENCH, "_gate", lambda: _FakeGate)
    out = BENCH.solve_and_match("gtap7_3x3", "altertax", 0, Path("/ref/out.gdx"))
    assert set(out) == {"check", "shock"}
    assert out["shock"] == {"match": 99.0, "code": 1, "var_div": {"pwmg": 1}}


def test_delta_counts_a_variable_new_or_gone(capsys):
    prev = {"shock": {"match": 98.0, "var_div": {"pf": 2, "xf": 1}}}
    cur = {"shock": {"match": 99.0, "var_div": {"pf": 1, "pm": 1}}}
    BENCH._delta(prev, cur, "shock")
    text = capsys.readouterr().out
    assert "vars TOWARD GAMS: 2 | vars AWAY: 1" in text
    assert "xf" in text and "pm" in text
