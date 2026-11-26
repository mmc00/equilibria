"""diff_holdfixed measures Python: at every solve, the cells GAMS holds fixed from
the previous period must be fixed in every period but the one being solved
(MAR-219). It used to read diff_altertax.py and always answer "Python freezes none"."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from equilibria.sources import find_reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
_spec = importlib.util.spec_from_file_location(
    "diff_holdfixed", ROOT / "scripts/gtap/diff_holdfixed.py"
)
assert _spec is not None and _spec.loader is not None
HF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(HF)

_FOLDER = find_reference(
    "gtap7_3x3", "gams", form="mcp", mode="altertax", ifsub=0, fbep="header"
)


class _Model:
    """Two variables over base/check/shock; ``free`` is {(var, period): free cells}."""

    periods = ("base", "check", "shock")

    def __init__(self, active, free):
        self._active, self._free = active, free

    def var_names(self):
        return ("pf", "xf")

    def row_names(self):
        return ("eq_pf",)

    def has_var(self, name):
        return name in self.var_names()

    def var_cells(self, name, period):
        return ((0,), (1,), (2,))

    def fixed_cells(self, name, period):
        n = self._free.get((name, period), 0)
        return frozenset(self.var_cells(name, period)[n:])

    def active_rows(self, name, period):
        return frozenset({(0,)}) if period == self._active else frozenset()


def test_a_period_frozen_outside_the_solved_one_is_clean():
    m = _Model("shock", {("pf", "shock"): 3, ("xf", "shock"): 3})
    assert HF.unfrozen_outside(m, ["pf", "xf", "pe"]) == ("shock", {})


def test_a_free_cell_in_another_period_is_reported():
    m = _Model("shock", {("pf", "shock"): 3, ("pf", "check"): 1})
    assert HF.unfrozen_outside(m, ["pf", "xf"]) == ("shock", {"check": {"pf": 1}})


def _measured(solves):
    return lambda *a, **k: {"solves": solves, "absent": ["axp"]}


def test_work_is_clean_when_every_solve_froze_the_other_periods(monkeypatch):
    monkeypatch.setattr(
        HF, "measure_holdfix", _measured([("check", {}), ("shock", {})])
    )
    out = HF._work(_args("shock"))
    assert out["status"] == "clean"
    assert out["meta"]["python_freezes_prior_stage"] is True


def test_work_is_dirty_when_a_solve_left_a_holdfix_cell_free(monkeypatch):
    monkeypatch.setattr(
        HF, "measure_holdfix", _measured([("shock", {"check": {"pf": 2}})])
    )
    out = HF._work(_args("shock"))
    assert out["status"] == "dirty"
    assert [v["entity"] for v in out["violations"]] == ["pf"]


def test_work_with_no_solve_of_the_period_is_a_tool_error(monkeypatch):
    monkeypatch.setattr(HF, "measure_holdfix", _measured([("check", {})]))
    out = HF._work(_args("shock"))
    assert out["status"] == "error"
    assert out["meta"]["error_kind"] == "solve_not_observed"


def _args(period):
    import argparse

    return argparse.Namespace(
        dataset="gtap7_3x3", gms=None, gdx=None, period=period, mode="altertax", ifsub=0
    )


def test_the_default_gms_is_the_data_reference():
    assert _FOLDER is not None
    gms, _ = HF._resolve_gms("gtap7_3x3", None, "altertax", 0)
    assert gms == _FOLDER / "model.gms"


@pytest.mark.skipif(_FOLDER is None, reason="GAMS MCP altertax ref absent")
@pytest.mark.needs_path
@pytest.mark.needs_gdxdump  # the seed reads the GAMS GDX through it
@pytest.mark.parametrize("period", ["check", "shock"])
def test_gtap7_3x3_altertax_is_clean(period):
    import json

    assert _FOLDER is not None

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/gtap/diff_holdfixed.py",
            "--dataset",
            "gtap7_3x3",
            "--period",
            period,
            "--gdx",
            str(_FOLDER / "out.gdx"),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["status"] == "clean", out["headline"]
